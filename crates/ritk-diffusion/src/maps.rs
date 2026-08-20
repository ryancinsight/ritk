//! Whole-volume tensor fitting and the scalar maps derived from it.
//!
//! [`crate::dti::estimate_dti`] fits one voxel. Turning a DWI series into a
//! usable FA or MD map needs three further things, and every caller that has
//! tried has had to supply them itself:
//!
//! 1. **A background mask.** Outside the head the signal is noise, and a tensor
//!    fitted to noise is strongly anisotropic. Unmasked, a bright rim traces the
//!    skull and dominates the FA range, so the tissue the map exists to show is
//!    compressed into the bottom of the scale.
//! 2. **Degenerate-fit rejection.** Some voxels pass the intensity mask yet
//!    still yield a collapsed, rank-one tensor: one large eigenvalue with the
//!    other two near zero. Such a tensor is positive-definite, so a sign check
//!    accepts it, and its FA approaches 1. These are the source of impossible
//!    anisotropy speckled through an otherwise plausible map.
//! 3. **The fit loop itself**, including which voxels were fitted at all, since
//!    an unfitted voxel and a genuinely isotropic one are different claims.
//!
//! Those are properties of the estimator, not of any particular front end, so
//! they live here rather than in each caller.
//!
//! # What is retained
//!
//! [`DiffusionMaps`] keeps the eigen-decomposition — eigenvalues and the
//! principal eigenvector — together with the acquisition [`GradientFrame`],
//! rather than the six tensor elements. Every standard DTI scalar map derives
//! from the eigenvalues alone, so this answers each of them without a second
//! decomposition, at 49 bytes per voxel instead of the roughly 144 that
//! retaining whole tensors would cost. A caller needing the tensor elements
//! themselves calls [`crate::dti::estimate_dti`] for the voxel it cares about.
//!
//! # Definitions
//!
//! With eigenvalues `λ₁ ≥ λ₂ ≥ λ₃`:
//!
//! | Map | Definition | Meaning |
//! |-----|------------|---------|
//! | FA | `sqrt(3/2 · Σ(λᵢ - λ̄)² / Σλᵢ²)` | how directional the diffusion is, in `[0, 1]` |
//! | MD | `(λ₁ + λ₂ + λ₃) / 3` | average diffusivity, mm²/s |
//! | AD | `λ₁` | diffusivity along the principal axis |
//! | RD | `(λ₂ + λ₃) / 2` | diffusivity across it |
//!
//! FA and MD come from [`crate::dti::DiffusionTensor`]; AD and RD are defined
//! here because they are pure functions of the eigenvalues.

mod volume;

pub use volume::{DirectionInterpolation, DtiVolume};

use ritk_diffusion_scheme::{GradientFrame, GradientScheme};
use thiserror::Error;

use crate::dti::{DtiConfig, estimate_dti, invariants};

/// Percentile of the reference signal that sets the masking scale.
///
/// Referencing a high percentile rather than the maximum keeps a single hot
/// voxel — a spike artefact, a vessel — from setting the scale and masking out
/// the brain along with the background.
const REFERENCE_PERCENTILE: usize = 98;

/// Fitting failed for a voxel, so it carries no orientation.
const UNFITTED: [f64; 3] = [0.0; 3];

/// Why a whole-volume fit could not be attempted.
///
/// Per-voxel fit failures are not errors: they are expected wherever the signal
/// is noise or the tensor collapses, and are reported through
/// [`DiffusionMaps::mask`] rather than aborting the volume.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DiffusionMapsError {
    /// No volumes were supplied.
    #[error("a diffusion series needs at least one volume")]
    NoVolumes,
    /// Volume count and scheme length differ.
    #[error("series has {volume_count} volumes but the scheme declares {acquisition_count}")]
    VolumeCountMismatch {
        /// Number of supplied volumes.
        volume_count: usize,
        /// Number of scheme entries.
        acquisition_count: usize,
    },
    /// Volumes differ in voxel count, so they are not one series.
    #[error("volume {index} has {length} voxels but volume 0 has {expected}")]
    VolumeLengthMismatch {
        /// Acquisition-order index of the offending volume.
        index: usize,
        /// Its voxel count.
        length: usize,
        /// The voxel count established by volume 0.
        expected: usize,
    },
    /// The scheme declares no reference volume, so no mask can be built.
    #[error("gradient scheme has no unweighted reference volume to build a mask from")]
    NoReferenceVolume,
    /// A configured bound is not a usable number.
    #[error("{parameter} must be finite and nonnegative, got {value}")]
    InvalidConfiguration {
        /// Name of the offending configuration field.
        parameter: &'static str,
        /// The rejected value.
        value: f64,
    },
    /// The fitted maps use a frame that cannot be placed on an image-index grid.
    #[error("DTI volume placement requires ImageAxis gradients, got {frame:?}")]
    UnsupportedGradientFrame {
        /// Coordinate frame retained by the fitted maps.
        frame: GradientFrame,
    },
}

/// Masking and rejection policy for a whole-volume fit.
///
/// The defaults are physical rather than tuned: they come from what water in
/// tissue can do, not from what makes a particular image look right.
#[derive(Debug, Clone, Copy)]
pub struct DiffusionMapsConfig {
    /// Mask threshold as a fraction of the reference signal's upper percentile.
    ///
    /// A voxel whose reference signal falls below
    /// `fraction × percentile(reference)` is not fitted. Set to zero to fit
    /// every voxel.
    pub background_fraction: f64,

    /// Smallest admissible eigenvalue, in mm²/s.
    ///
    /// Radial diffusivity in coherent white matter is around 2–3 × 10⁻⁴ mm²/s;
    /// no tissue restricts water two orders below that. A smallest eigenvalue
    /// under this floor is a collapsed, rank-one fit rather than an anisotropic
    /// voxel.
    pub diffusivity_floor: f64,

    /// Largest admissible eigenvalue, in mm²/s.
    ///
    /// Free water at body temperature diffuses at about 3.0 × 10⁻³ mm²/s, and
    /// no tissue compartment exceeds free water. An eigenvalue above this is a
    /// fit artefact, not a measurement.
    pub diffusivity_ceiling: f64,

    /// Per-voxel tensor-fitting configuration.
    pub dti: DtiConfig,
}

impl Default for DiffusionMapsConfig {
    fn default() -> Self {
        Self {
            background_fraction: 0.12,
            diffusivity_floor: 1.0e-5,
            diffusivity_ceiling: 3.2e-3,
            dti: DtiConfig::default(),
        }
    }
}

impl DiffusionMapsConfig {
    /// Reject a configuration that cannot describe a physical bound.
    fn validate(&self) -> Result<(), DiffusionMapsError> {
        let checks = [
            ("background_fraction", self.background_fraction),
            ("diffusivity_floor", self.diffusivity_floor),
            ("diffusivity_ceiling", self.diffusivity_ceiling),
        ];
        for (parameter, value) in checks {
            if !value.is_finite() || value < 0.0 {
                return Err(DiffusionMapsError::InvalidConfiguration { parameter, value });
            }
        }
        Ok(())
    }

    /// Whether an eigenvalue triple describes a physically admissible tensor.
    ///
    /// Eigenvalues arrive in descending order, so the largest and smallest are
    /// the ends of the slice.
    fn admits(&self, eigenvalues: &[f64; 3]) -> bool {
        eigenvalues[2] >= self.diffusivity_floor && eigenvalues[0] <= self.diffusivity_ceiling
    }
}

/// Eigen-decomposed tensor field over a volume, with the voxels that were fitted.
///
/// Unfitted voxels carry zero eigenvalues and a zero principal eigenvector, so
/// every derived map reads zero there. [`Self::mask`] distinguishes that from a
/// voxel genuinely measured as isotropic.
#[derive(Debug, Clone)]
pub struct DiffusionMaps {
    eigenvalues: Vec<[f64; 3]>,
    principal: Vec<[f64; 3]>,
    mask: Vec<bool>,
    frame: GradientFrame,
}

#[cfg(test)]
impl DiffusionMaps {
    /// Assemble maps directly, for cases a fit cannot produce.
    ///
    /// A diffusion tensor is sign-invariant, so fitting can never yield two
    /// voxels holding `v` and `−v` for the same fibre — the solver picks one
    /// sign deterministically per tensor. That case still arises across
    /// neighbouring voxels with different tensors, and it is exactly what
    /// sign-invariant interpolation exists to survive, so the test that proves
    /// it has to build the state directly.
    pub(crate) fn from_parts(
        eigenvalues: Vec<[f64; 3]>,
        principal: Vec<[f64; 3]>,
        mask: Vec<bool>,
        frame: GradientFrame,
    ) -> Self {
        assert!(
            eigenvalues.len() == principal.len() && principal.len() == mask.len(),
            "invariant: every per-voxel field covers the same voxels"
        );
        Self {
            eigenvalues,
            principal,
            mask,
            frame,
        }
    }
}

impl DiffusionMaps {
    /// Number of voxels in the volume.
    #[must_use]
    pub fn len(&self) -> usize {
        self.mask.len()
    }

    /// Whether the volume has no voxels.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mask.is_empty()
    }

    /// Which voxels were fitted successfully.
    #[must_use]
    pub fn mask(&self) -> &[bool] {
        &self.mask
    }

    /// Count of voxels that yielded an admissible tensor.
    #[must_use]
    pub fn fitted_count(&self) -> usize {
        self.mask.iter().filter(|fitted| **fitted).count()
    }

    /// Eigenvalues per voxel, descending.
    #[must_use]
    pub fn eigenvalues(&self) -> &[[f64; 3]] {
        &self.eigenvalues
    }

    /// Principal eigenvector per voxel — the local fibre orientation.
    #[must_use]
    pub fn principal_eigenvector(&self) -> &[[f64; 3]] {
        &self.principal
    }

    /// Coordinate frame of the gradient directions used to fit the maps.
    #[must_use]
    pub const fn frame(&self) -> GradientFrame {
        self.frame
    }

    /// Fractional anisotropy, in `[0, 1]`.
    #[must_use]
    pub fn fractional_anisotropy(&self) -> Vec<f64> {
        self.derive(invariants::fractional_anisotropy)
    }

    /// Fractional anisotropy at one voxel.
    ///
    /// The whole-volume accessors allocate, which a per-voxel query — a
    /// tractography step, say — cannot afford to do at every point.
    ///
    /// # Panics
    ///
    /// If `voxel` is out of range.
    #[must_use]
    pub fn fractional_anisotropy_at(&self, voxel: usize) -> f64 {
        if self.mask[voxel] {
            invariants::fractional_anisotropy(self.eigenvalues[voxel])
        } else {
            0.0
        }
    }

    /// Mean diffusivity, in mm²/s.
    #[must_use]
    pub fn mean_diffusivity(&self) -> Vec<f64> {
        self.derive(invariants::mean_diffusivity)
    }

    /// Axial diffusivity `λ₁`, in mm²/s.
    #[must_use]
    pub fn axial_diffusivity(&self) -> Vec<f64> {
        self.derive(invariants::axial_diffusivity)
    }

    /// Radial diffusivity `(λ₂ + λ₃) / 2`, in mm²/s.
    #[must_use]
    pub fn radial_diffusivity(&self) -> Vec<f64> {
        self.derive(invariants::radial_diffusivity)
    }

    /// Relative anisotropy — see [`invariants::relative_anisotropy`].
    #[must_use]
    pub fn relative_anisotropy(&self) -> Vec<f64> {
        self.derive(invariants::relative_anisotropy)
    }

    /// Mode of anisotropy in `[−1, 1]` — see [`invariants::mode`].
    ///
    /// Read alongside [`Self::fractional_anisotropy`]: FA gives how far the
    /// tensor is from isotropic, mode gives which way it departed. A voxel with
    /// high FA and mode near `−1` is a plane, not a fibre, and is the signature
    /// of crossing bundles inside one voxel.
    #[must_use]
    pub fn mode(&self) -> Vec<f64> {
        self.derive(invariants::mode)
    }

    /// Westin linear measure `cₗ` — see [`invariants::westin_measures`].
    #[must_use]
    pub fn linear_measure(&self) -> Vec<f64> {
        self.derive(|eigenvalues| invariants::westin_measures(eigenvalues).0)
    }

    /// Westin planar measure `cₚ` — see [`invariants::westin_measures`].
    #[must_use]
    pub fn planar_measure(&self) -> Vec<f64> {
        self.derive(|eigenvalues| invariants::westin_measures(eigenvalues).1)
    }

    /// Westin spherical measure `cₛ` — see [`invariants::westin_measures`].
    ///
    /// Unfitted voxels read zero like every other map, rather than the
    /// spherical limit of one that the bare invariant returns for a null
    /// tensor: a voxel that was never fitted makes no claim about shape.
    #[must_use]
    pub fn spherical_measure(&self) -> Vec<f64> {
        self.derive(|eigenvalues| invariants::westin_measures(eigenvalues).2)
    }

    /// Direction-encoded colour per voxel — see
    /// [`invariants::colour_by_orientation`].
    ///
    /// Unfitted voxels are black.
    #[must_use]
    pub fn colour_by_orientation(&self) -> Vec<[f64; 3]> {
        self.eigenvalues
            .iter()
            .zip(&self.principal)
            .zip(&self.mask)
            .map(|((eigenvalues, principal), fitted)| {
                if *fitted {
                    invariants::colour_by_orientation(*eigenvalues, *principal)
                } else {
                    [0.0; 3]
                }
            })
            .collect()
    }

    /// Apply a scalar function of the eigenvalues over the volume.
    ///
    /// Unfitted voxels short-circuit to zero rather than evaluating the
    /// function on zeros, so a definition with a zero denominator cannot leak a
    /// NaN into the background.
    fn derive(&self, measure: impl Fn([f64; 3]) -> f64) -> Vec<f64> {
        self.eigenvalues
            .iter()
            .zip(&self.mask)
            .map(|(eigenvalues, fitted)| if *fitted { measure(*eigenvalues) } else { 0.0 })
            .collect()
    }
}

/// Fit one tensor per voxel across a diffusion series and derive its maps.
///
/// `volumes` holds one slice per acquisition, in scheme order, each covering
/// the same voxels in the same layout. Any element type that converts to `f64`
/// is accepted, so a caller holding `f32` image data passes it directly.
///
/// Voxels below the background threshold, and voxels whose fit is not
/// physically admissible under `config`, are excluded and read zero in every
/// map. A per-voxel fit failure never aborts the volume.
///
/// # Errors
///
/// [`DiffusionMapsError`] when the series and scheme disagree, the volumes are
/// not one consistent series, the scheme declares no reference volume, or a
/// configured bound is not a usable number.
pub fn fit_diffusion_maps<T>(
    scheme: &GradientScheme,
    volumes: &[&[T]],
    config: &DiffusionMapsConfig,
) -> Result<DiffusionMaps, DiffusionMapsError>
where
    T: Copy + Into<f64>,
{
    config.validate()?;

    let Some(first) = volumes.first() else {
        return Err(DiffusionMapsError::NoVolumes);
    };
    if volumes.len() != scheme.len() {
        return Err(DiffusionMapsError::VolumeCountMismatch {
            volume_count: volumes.len(),
            acquisition_count: scheme.len(),
        });
    }
    let voxels = first.len();
    for (index, volume) in volumes.iter().enumerate() {
        if volume.len() != voxels {
            return Err(DiffusionMapsError::VolumeLengthMismatch {
                index,
                length: volume.len(),
                expected: voxels,
            });
        }
    }

    // Resolved once: `b0_indices` allocates, and the fit loop runs per voxel.
    let references = scheme.b0_indices(config.dti.b0_threshold());
    if references.is_empty() {
        return Err(DiffusionMapsError::NoReferenceVolume);
    }
    let floor = mask_floor(volumes, &references, config);

    let mut eigenvalues = vec![UNFITTED; voxels];
    let mut principal = vec![UNFITTED; voxels];
    let mut mask = vec![false; voxels];
    let mut signals = vec![0.0_f64; volumes.len()];

    for voxel in 0..voxels {
        for (slot, volume) in signals.iter_mut().zip(volumes) {
            *slot = volume[voxel].into();
        }
        if reference_of(&signals, &references) < floor {
            continue;
        }
        // A per-voxel failure is expected, not exceptional: it means this voxel
        // carries no tensor, which the mask already records.
        let Ok(tensor) = estimate_dti(scheme, &signals, config.dti) else {
            continue;
        };
        if !config.admits(tensor.eigenvalues()) {
            continue;
        }
        eigenvalues[voxel] = *tensor.eigenvalues();
        principal[voxel] = tensor.principal_eigenvector();
        mask[voxel] = true;
    }

    Ok(DiffusionMaps {
        eigenvalues,
        principal,
        mask,
        frame: scheme.frame(),
    })
}

/// Mean of the reference volumes at one voxel.
///
/// Averaging every b = 0 volume rather than picking one matters because the
/// mask is applied per voxel: a single noisy reference speckles the mask along
/// tissue boundaries, where the threshold decision is closest.
fn reference_of(signals: &[f64], references: &[usize]) -> f64 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "a diffusion series has far fewer volumes than f64's exact-integer range"
    )]
    let count = references.len() as f64;
    references.iter().map(|index| signals[*index]).sum::<f64>() / count
}

/// Intensity below which a voxel is treated as background.
///
/// Returns zero when masking is disabled, which admits every voxel since
/// reference signals are nonnegative.
fn mask_floor<T>(volumes: &[&[T]], references: &[usize], config: &DiffusionMapsConfig) -> f64
where
    T: Copy + Into<f64>,
{
    if config.background_fraction == 0.0 {
        return 0.0;
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "a diffusion series has far fewer volumes than f64's exact-integer range"
    )]
    let count = references.len() as f64;
    // Accumulated one reference volume at a time rather than one voxel at a
    // time: each pass walks a volume sequentially, where gathering across
    // volumes per voxel would stride through every one of them.
    let mut reference = vec![0.0_f64; volumes[0].len()];
    for index in references {
        for (slot, value) in reference.iter_mut().zip(volumes[*index]) {
            *slot += (*value).into();
        }
    }
    for slot in &mut reference {
        *slot /= count;
    }

    reference.sort_by(f64::total_cmp);
    let rank = reference.len() * REFERENCE_PERCENTILE / 100;
    let upper = reference[rank.min(reference.len().saturating_sub(1))];
    upper * config.background_fraction
}

#[cfg(test)]
mod tests;
