//! Full-brain parcellation by atlas propagation.
//!
//! Parcellating a subject brain means giving every voxel the identifier of the
//! anatomical region it belongs to. The approach here is the standard one:
//! rather than segmenting the subject from scratch, take a brain that has
//! already been labelled — by hand, or by a published atlas — deform it onto the
//! subject, and carry its labels along.
//!
//! ```text
//! atlas intensity ──► register to subject ──► deformation
//! atlas labels    ──────────────────────────────┴──► warp ──► Parcellation
//! ```
//!
//! # Why the labels are warped and never interpolated
//!
//! Label values are identifiers, not measurements. Region 17 and region 19 do
//! not average to region 18, and any interpolation that produced 18 would invent
//! an anatomical claim out of two unrelated ones — silently, since 18 is a valid
//! label. Every resampling here is nearest-neighbour for that reason. The cost
//! is a jagged boundary at the voxel scale, which is the honest representation
//! of what a label map can say.
//!
//! # Why one atlas is usually not enough
//!
//! A single atlas transfers not only its anatomy but its idiosyncrasies, and
//! wherever the registration is locally wrong, the labels are locally wrong with
//! no signal that they are. Registering several independently labelled brains
//! and fusing their votes is the standard remedy: an error must be shared by a
//! majority of the atlases to survive, and disagreement is measurable rather
//! than invisible.
//!
//! [`parcellate_with_atlas_set`] therefore returns the per-voxel agreement
//! alongside the labels. Low agreement marks where the result is a coin toss
//! between neighbouring parcels — usually the boundaries, which is where the
//! answer matters most for a connectome, since that is where streamlines end.
//!
//! # What this cannot do
//!
//! Atlas propagation transfers a *predefined* parcellation. It cannot discover a
//! region the atlas does not contain, and it cannot represent anatomy the
//! registration could not reach — a resected cavity, a large lesion, or a
//! malformation has no counterpart in a healthy atlas, and the labels warped
//! over it are meaningless rather than absent. The agreement map is the only
//! signal of that, and it is a weak one when every atlas is equally wrong.
//!
//! # References
//!
//! * Rohlfing, T., Brandt, R., Menzel, R. & Maurer, C. R. (2004). Evaluation of
//!   atlas selection strategies for atlas-based image segmentation with
//!   application to confocal microscopy images of bee brains. *NeuroImage*
//!   21(4):1428–1442. — why multi-atlas beats single-atlas.
//! * Avants, B. B., Epstein, C. L., Grossman, M. & Gee, J. C. (2008).
//!   Symmetric diffeomorphic image registration with cross-correlation.
//!   *Medical Image Analysis* 12(1):26–41. — the SyN registration used here.

use coeus_core::CpuAddressableStorage;
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_parcellation::{Parcellation, ParcellationError, ParcellationGrid};
use ritk_spatial::VolumeDims;

use crate::atlas::label_fusion::{joint_label_fusion, majority_vote, LabelFusionConfig};
use crate::deformable_field_ops::{
    compose_fields_into, scaling_and_squaring, warp_image, CpuFieldSmoother, CpuOrGpu,
    VelocityField, WarpInterpolation,
};
use crate::diffeomorphic::multires_syn::{
    InverseConsistency, MultiResSyNConfig, MultiResSyNRegistration,
};
use crate::error::RegistrationError;

/// A labelled reference brain: an intensity image and its parcellation.
///
/// The two must lie on the same voxel grid — the labels say what the intensities
/// show, so a mismatch means they describe different brains.
#[derive(Debug, Clone)]
pub struct LabelledAtlas {
    /// Intensity image, flat in `[nz, ny, nx]` order. This is what the
    /// registration matches against the subject.
    pub intensity: Vec<f32>,
    /// Region label per voxel, in the same order and of the same length.
    pub labels: Vec<u32>,
    /// Human-readable names keyed by label.
    pub region_names: Vec<(u32, String)>,
}

impl LabelledAtlas {
    /// Reject an atlas whose two volumes disagree or do not cover the grid.
    fn validate(&self, voxels: usize, index: usize) -> Result<(), RegistrationError> {
        if self.intensity.len() != voxels || self.labels.len() != voxels {
            return Err(RegistrationError::DimensionMismatch(format!(
                "atlas {index} has {} intensity and {} label voxels, but the \
                 subject grid has {voxels}",
                self.intensity.len(),
                self.labels.len()
            )));
        }
        Ok(())
    }
}

/// How a subject is parcellated from one or more atlases.
#[derive(Debug, Clone)]
pub struct AtlasParcellationConfig {
    /// Registration driving each atlas onto the subject.
    pub registration: MultiResSyNConfig,
    /// How several atlases' votes are combined. Ignored for a single atlas.
    pub fusion: LabelFusion,
}

impl Default for AtlasParcellationConfig {
    /// A three-level SyN registration and majority voting.
    ///
    /// The registration settings are the ANTs-style defaults the SyN
    /// implementation documents: three resolution levels with decreasing
    /// iteration counts, a two-voxel regularisation sigma, and a quarter-voxel
    /// gradient step. They are a starting point sized for a whole head at
    /// millimetre resolution, not a tuned optimum for any particular cohort.
    fn default() -> Self {
        Self {
            registration: MultiResSyNConfig {
                num_levels: 3,
                iterations_per_level: vec![40, 20, 10],
                sigma_smooth: 2.0,
                convergence_threshold: 1.0e-7,
                convergence_window: 10,
                n_squarings: 6,
                cc_window_radius: 2,
                gradient_step: 0.25,
                enforce_inverse_consistency: InverseConsistency::Relaxed,
            },
            fusion: LabelFusion::default(),
        }
    }
}

/// How multiple atlases' labels are combined into one.
#[derive(Debug, Clone, Default)]
pub enum LabelFusion {
    /// The label most atlases agree on, ties going to the smaller label.
    ///
    /// Uses only the labels, so it treats every atlas as equally trustworthy
    /// everywhere. That is the right assumption when the atlases are
    /// interchangeable and the wrong one when some registered better than
    /// others in a particular region.
    #[default]
    MajorityVote,

    /// Weighted voting, with each atlas's weight set by how well its warped
    /// intensities match the subject locally.
    ///
    /// Lets a well-registered atlas outvote a poorly registered one in the
    /// region where that is true, rather than globally. Costs a patch
    /// comparison and a small dense solve per voxel, so it is materially slower
    /// than voting.
    JointLabelFusion(LabelFusionConfig),
}

/// A parcellation, with how confidently each voxel was assigned.
#[derive(Debug, Clone)]
pub struct ParcellationResult {
    /// The subject's parcellation, on the subject's own grid.
    pub parcellation: Parcellation,
    /// Per-voxel agreement in `[0, 1]`, in the parcellation's storage order.
    ///
    /// For majority voting this is the fraction of atlases that voted for the
    /// winning label; for joint label fusion it is the summed weight behind it.
    /// A single atlas has nothing to disagree with, so every voxel reads `1`,
    /// which is a statement about the method rather than about the anatomy.
    pub agreement: Vec<f32>,
    /// Final cross-correlation of each atlas's registration to the subject, in
    /// the order the atlases were supplied.
    ///
    /// A value far below the others marks an atlas that did not register, whose
    /// labels are then noise in the vote.
    pub registration_quality: Vec<f64>,
}

/// Parcellate a subject from a single labelled atlas.
///
/// `subject` supplies both the intensity to register against and the grid the
/// result lands on. The atlas must be on that same grid — resample it first if
/// it is not, since a registration cannot recover a resampling.
///
/// # Errors
///
/// [`RegistrationError`] when the volumes disagree in size or the registration
/// fails, and [`RegistrationError::InvalidConfiguration`] when the warped labels
/// cannot form a parcellation — which happens when every voxel came back
/// background, meaning the atlas landed entirely outside the subject.
pub fn parcellate_with_atlas<B>(
    subject: &Image<f32, B, 3>,
    atlas: &LabelledAtlas,
    config: &AtlasParcellationConfig,
) -> Result<ParcellationResult, RegistrationError>
where
    B: Backend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    parcellate_with_atlas_set(subject, std::slice::from_ref(atlas), config)
}

/// Parcellate a subject from several labelled atlases, fusing their votes.
///
/// Each atlas is registered to the subject independently and its labels warped
/// onto the subject grid; the warped label maps are then fused under
/// `config.fusion`.
///
/// # Errors
///
/// As [`parcellate_with_atlas`], plus
/// [`RegistrationError::InvalidConfiguration`] when no atlas is supplied.
pub fn parcellate_with_atlas_set<B>(
    subject: &Image<f32, B, 3>,
    atlases: &[LabelledAtlas],
    config: &AtlasParcellationConfig,
) -> Result<ParcellationResult, RegistrationError>
where
    B: Backend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    if atlases.is_empty() {
        return Err(RegistrationError::InvalidConfiguration(
            "parcellation needs at least one atlas".into(),
        ));
    }

    let dims = image_dims(subject);
    let voxels = dims[0] * dims[1] * dims[2];
    for (index, atlas) in atlases.iter().enumerate() {
        atlas.validate(voxels, index)?;
    }

    let target = subject
        .data_slice()
        .map_err(|error| RegistrationError::ImageValidationError(error.to_string()))?;
    let spacing = registration_spacing(subject);

    let mut warped_labels: Vec<Vec<u32>> = Vec::with_capacity(atlases.len());
    let mut warped_intensities: Vec<Vec<f32>> = Vec::with_capacity(atlases.len());
    let mut registration_quality = Vec::with_capacity(atlases.len());

    let engine = MultiResSyNRegistration::new(config.registration.clone());
    for atlas in atlases {
        let mut factory = |level: [usize; 3]| -> CpuOrGpu<B> {
            CpuOrGpu::Cpu(CpuFieldSmoother::new(
                level,
                config.registration.sigma_smooth,
            ))
        };
        let result = engine.register_with(target, &atlas.intensity, dims, spacing, &mut factory)?;

        let displacement = atlas_to_subject_displacement(&result, dims, &config.registration);
        // Labels ride through as f32 only because the warp operates on f32
        // buffers; nearest-neighbour sampling means every value that comes out
        // is one that went in, so the round trip through the float is exact for
        // any label a 32-bit float represents exactly.
        let label_field: Vec<f32> = atlas.labels.iter().map(|label| *label as f32).collect();
        let warped = warp_image(
            &label_field,
            VolumeDims(dims),
            &displacement.z,
            &displacement.y,
            &displacement.x,
            WarpInterpolation::Nearest,
        );
        warped_labels.push(warped.iter().map(|value| round_to_label(*value)).collect());

        if matches!(config.fusion, LabelFusion::JointLabelFusion(_)) {
            warped_intensities.push(warp_image(
                &atlas.intensity,
                VolumeDims(dims),
                &displacement.z,
                &displacement.y,
                &displacement.x,
                WarpInterpolation::Trilinear,
            ));
        }
        registration_quality.push(result.final_cc);
    }

    let label_views: Vec<&[u32]> = warped_labels.iter().map(Vec::as_slice).collect();
    let fused = match &config.fusion {
        LabelFusion::MajorityVote => majority_vote(&label_views, dims)?,
        LabelFusion::JointLabelFusion(fusion_config) => {
            let intensity_views: Vec<&[f32]> =
                warped_intensities.iter().map(Vec::as_slice).collect();
            joint_label_fusion(target, &intensity_views, &label_views, dims, fusion_config)?
        }
    };

    let names = merged_region_names(atlases);
    let parcellation = parcellation_from_labels(fused.labels.into_boxed_slice(), subject, names)
        .map_err(|error| {
            RegistrationError::InvalidConfiguration(format!(
                "warped atlas labels do not form a parcellation: {error}"
            ))
        })?;

    Ok(ParcellationResult {
        parcellation,
        agreement: fused.confidence,
        registration_quality,
    })
}

/// Build a [`Parcellation`] on an image's grid from a label volume.
///
/// The bridge between the two crates' index conventions, which run in opposite
/// directions. An [`Image`] numbers its spatial axes outermost-first: axis 0 is
/// the *slowest*-varying index, so for a `[nz, ny, nx]` volume `spacing[0]` is
/// the slice thickness and `direction`'s first column is the slice normal. A
/// [`ParcellationGrid`] numbers them innermost-first, with axis 0 the fastest
/// index. Bridging therefore reverses all three: the shape, the spacing, and the
/// *columns* of the direction matrix.
///
/// Only the origin and the flat storage order pass through unchanged — both are
/// stated in absolute terms rather than per axis.
///
/// Reversing some but not all of them produces a grid that constructs, validates,
/// and answers every query, while placing voxels somewhere they are not. That is
/// why the equality it has to satisfy is written out rather than assumed:
///
/// ```text
/// origin + D·(s ⊙ [i₀, i₁, i₂])  =  origin + D_g·(s_g ⊙ [i₂, i₁, i₀])
/// ```
///
/// which holds for every index exactly when `s_g = reverse(s)` and column `c`
/// of `D_g` is column `2 − c` of `D`.
///
/// # Errors
///
/// [`ParcellationError`] when the label count does not cover the grid, the
/// geometry is degenerate, or every voxel is background.
pub fn parcellation_from_labels<B>(
    labels: Box<[u32]>,
    reference: &Image<f32, B, 3>,
    region_names: Vec<(u32, String)>,
) -> Result<Parcellation, ParcellationError>
where
    B: Backend,
{
    let grid = ParcellationGrid::from_image_order(
        reference.shape(),
        reference.spacing().to_array(),
        reference.origin().to_array(),
        reference.direction().to_row_major(),
    )?;
    Parcellation::new(labels, grid, region_names)
}

/// The field that resamples atlas labels onto the subject grid.
///
/// # What the returned field is
///
/// [`warp_image`] evaluates `out(p) = in(p + d(p))`, so `d` answers, for each
/// *output* voxel, where in the *input* to look. The output grid here is the
/// subject and the input is the atlas, so `d` is the pullback subject → atlas —
/// not the forward map that carries atlas anatomy onto the subject, which is its
/// inverse. The two differ by a sign and are equally plausible as field data,
/// which is why the direction is stated rather than assumed.
///
/// # Recovering it from a symmetric registration
///
/// SyN does not return one field between the images. It returns `v₁` and `v₂`
/// meeting at a midpoint, and both are pullbacks by the same convention as the
/// warp: `warped_fixed(p) = fixed(p + exp(v₁)(p))` means `exp(v₁)` maps a
/// midpoint location to its fixed-image counterpart, and likewise `exp(v₂)` maps
/// midpoint to moving. Reaching from subject to atlas therefore goes against
/// `v₁` and along `v₂`:
///
/// ```text
/// d = exp(v₂) ∘ exp(−v₁)
/// ```
///
/// Composition here follows `compose_fields`, whose `(φ₁, φ₂)` applies `φ₂`
/// first — so `φ₂ = exp(−v₁)` steps subject to midpoint and `φ₁ = exp(v₂)`
/// steps midpoint to atlas.
fn atlas_to_subject_displacement(
    result: &crate::diffeomorphic::SyNResult,
    dims: [usize; 3],
    config: &MultiResSyNConfig,
) -> VelocityField {
    let volume = VolumeDims(dims);

    // Subject → midpoint: against the fixed image's own pullback.
    let negated_forward = VelocityField::new(
        result.forward_field.z.iter().map(|v| -v).collect(),
        result.forward_field.y.iter().map(|v| -v).collect(),
        result.forward_field.x.iter().map(|v| -v).collect(),
    );
    let subject_to_midpoint = scaling_and_squaring(
        &negated_forward.z,
        &negated_forward.y,
        &negated_forward.x,
        volume,
        config.n_squarings,
    );

    // Midpoint → atlas: the moving image's pullback as returned.
    let midpoint_to_atlas = scaling_and_squaring(
        &result.inverse_field.z,
        &result.inverse_field.y,
        &result.inverse_field.x,
        volume,
        config.n_squarings,
    );

    let mut composed = VelocityField::zeros(dims[0] * dims[1] * dims[2]);
    compose_fields_into(
        crate::deformable_field_ops::VectorField {
            z: &midpoint_to_atlas.z,
            y: &midpoint_to_atlas.y,
            x: &midpoint_to_atlas.x,
        },
        crate::deformable_field_ops::VectorField {
            z: &subject_to_midpoint.z,
            y: &subject_to_midpoint.y,
            x: &subject_to_midpoint.x,
        },
        volume,
        crate::deformable_field_ops::VectorFieldMut {
            z: &mut composed.z,
            y: &mut composed.y,
            x: &mut composed.x,
        },
    );
    composed
}

/// Image shape as the `[nz, ny, nx]` the registration and fusion code expects.
fn image_dims<B: Backend>(image: &Image<f32, B, 3>) -> [usize; 3] {
    image.shape()
}

/// Voxel spacing in the `[nz, ny, nx]` order the registration expects.
///
/// Unreversed, unlike the parcellation bridge above, because the registration
/// shares the image's convention rather than the grid's: its `dims` are
/// `[nz, ny, nx]` and its field code reads `spacing[0]` as the slice thickness,
/// which is exactly what the image's axis 0 already is. Reversing here would
/// scale the deformation by the wrong extent on every anisotropic volume — a
/// silent geometric error rather than a failure — and the two directions are
/// easy to conflate, so both are stated where they are applied.
fn registration_spacing<B: Backend>(image: &Image<f32, B, 3>) -> [f64; 3] {
    image.spacing().to_array()
}

/// Nearest label to a warped float value, clamped at background.
///
/// A nearest-neighbour warp only ever copies values that were already present,
/// so the rounding recovers every label the warp saw; the clamp guards the
/// out-of-field border, which the warp fills by edge clamping.
///
/// Rounding goes through `f32::round` rather than the usual `+ 0.5` truncation
/// trick, which is not equivalent at the top of the range: `f32` carries 24
/// significant bits, so adding a half to 16777215 rounds the *sum* up to
/// 16777216 and the trick returns a label one higher than the one it was given.
fn round_to_label(value: f32) -> u32 {
    if value <= 0.0 {
        return 0;
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the value is positive and originates as a u32 label"
    )]
    let label = value.round() as u32;
    label
}

/// Region names from every atlas, deduplicated by label.
///
/// Atlases in one set share a labelling scheme by construction — that is what
/// makes fusing their votes meaningful — so the first name seen for a label is
/// as good as any other. A disagreement here would mean the atlases are not
/// actually a set, which the fusion could not detect anyway.
fn merged_region_names(atlases: &[LabelledAtlas]) -> Vec<(u32, String)> {
    let mut names: Vec<(u32, String)> = atlases
        .iter()
        .flat_map(|atlas| atlas.region_names.iter().cloned())
        .collect();
    names.sort_by_key(|(label, _)| *label);
    names.dedup_by_key(|(label, _)| *label);
    names
}

#[cfg(test)]
mod tests;
