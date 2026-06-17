//! Grayscale morphological filters for 3-D images.
//!
//! # Mathematical Specification
//!
//! Grayscale morphology extends binary morphology to scalar-valued images by
//! replacing set union/intersection with max/min over a structuring element B:
//!
//! - **Erosion**: `(E_B f)(x) = min_{b ∈ B} f(x + b)`
//! - **Dilation**: `(D_B f)(x) = max_{b ∈ B} f(x - b)`
//!
//! The structuring element used here is a cubic neighbourhood of half-width
//! `radius`, i.e. B = { b ∈ ℤ³ : |b_i| ≤ r for all i }. Boundary handling
//! uses replicate (clamp) padding.
//!
//! # Derived Operations
//!
//! - **Opening**: `O_B = D_B ∘ E_B` — removes bright features smaller than B.
//! - **Closing**: `C_B = E_B ∘ D_B` — removes dark features smaller than B.
//!
//! # Complexity
//!
//! O(N · (2r+1)³) where N is the total voxel count and r is the radius.
//!
//! # References
//!
//! - Serra, J. (1982). *Image Analysis and Mathematical Morphology*. Academic Press.
//! - Soille, P. (2003). *Morphological Image Analysis*, 2nd ed. Springer.

pub mod grayscale_dilation;
pub mod grayscale_erosion;

pub use grayscale_dilation::GrayscaleDilation;
pub use grayscale_erosion::GrayscaleErosion;

pub mod hit_or_miss;
pub mod label_morphology;
pub mod top_hat;

pub use hit_or_miss::HitOrMissTransform;
pub use label_morphology::{
    LabelClosing, LabelDilation, LabelErosion, LabelOpening, MorphologicalReconstruction,
    ReconstructionMode,
};
pub use top_hat::{BlackTopHatFilter, WhiteTopHatFilter};

pub mod binary_closing;
pub mod binary_dilate;
pub mod binary_erode;
pub mod binary_fillhole;
pub mod binary_opening;

pub use binary_closing::BinaryMorphologicalClosing;
pub use binary_dilate::BinaryDilateFilter;
pub use binary_erode::BinaryErodeFilter;
pub use binary_fillhole::BinaryFillholeFilter;
pub use binary_opening::BinaryMorphologicalOpening;

pub mod grayscale_closing;
pub mod grayscale_fillhole;
pub mod grayscale_gradient;
pub mod grayscale_grind_peak;
pub mod grayscale_opening;

pub use grayscale_closing::GrayscaleClosingFilter;
pub use grayscale_fillhole::GrayscaleFillholeFilter;
pub use grayscale_gradient::GrayscaleMorphologicalGradientFilter;
pub use grayscale_grind_peak::GrayscaleGrindPeakFilter;
pub use grayscale_opening::GrayscaleOpeningFilter;

pub mod grayscale_geodesic;
pub use grayscale_geodesic::{GrayscaleGeodesicDilationFilter, GrayscaleGeodesicErosionFilter};

pub mod h_transform;
pub use h_transform::{HConcaveFilter, HConvexFilter, HMaximaFilter, HMinimaFilter};

pub mod regional_extrema;
pub use regional_extrema::{
    RegionalMaximaFilter, RegionalMinimaFilter, ValuedRegionalMaximaFilter,
    ValuedRegionalMinimaFilter,
};

pub mod reconstruction_opening_closing;
pub use reconstruction_opening_closing::{
    ClosingByReconstructionFilter, OpeningByReconstructionFilter,
};

pub mod morphological_laplace;
pub use morphological_laplace::MorphologicalLaplacian;

pub mod binary_contour;
pub mod connectivity;
pub mod label_contour;
pub mod voting_binary;

pub use binary_contour::BinaryContourImageFilter;
pub use connectivity::Connectivity;
pub use label_contour::LabelContourImageFilter;
pub use voting_binary::VotingBinaryImageFilter;

pub mod iterate_structure;
pub use iterate_structure::{iterate_structure, iterate_structure_with_origin, BoolStructure};

pub mod types;
pub use types::ForegroundValue;

/// True if voxel `(iz, iy, ix)` lies on the image border, **ignoring degenerate
/// (size-1) axes**.
///
/// A naive `iz == 0 || iz == nz-1 || …` test marks *every* voxel of a `z = 1`
/// (2-D) volume as border, because `iz == 0` is always true — which silently
/// turns border-seeded reconstructions (fill-hole, grind-peak) into the
/// identity on 2-D images, diverging from ITK/SimpleITK. Excluding size-1 axes
/// makes the border the frame of the genuinely-present dimensions (the 2-D frame
/// for a `z = 1` slab). See the `z = 1` degenerate-axis trap.
#[inline]
pub(crate) fn on_image_border(iz: usize, iy: usize, ix: usize, dims: [usize; 3]) -> bool {
    let [nz, ny, nx] = dims;
    (nz > 1 && (iz == 0 || iz == nz - 1))
        || (ny > 1 && (iy == 0 || iy == ny - 1))
        || (nx > 1 && (ix == 0 || ix == nx - 1))
}

/// Replicate-pad a flat `Z×Y×X` volume by `r` voxels on every face (edge-clamp).
///
/// ITK's composed grayscale opening/closing pads the input by the SE radius
/// before the erode/dilate pair and crops afterward (the "safe border"). Without
/// it, the second operation of the pair reads edge-clamped intermediate values
/// instead of the true padded ones, so the border band (within `r` of an edge)
/// diverges from `sitk.GrayscaleMorphological{Opening,Closing}`. Replicating a
/// degenerate (size-1) axis is harmless — the duplicated planes are identical,
/// so the min/max over them is unchanged.
pub(crate) fn pad_replicate_3d(data: &[f32], dims: [usize; 3], r: usize) -> (Vec<f32>, [usize; 3]) {
    if r == 0 {
        return (data.to_vec(), dims);
    }
    let [nz, ny, nx] = dims;
    let pdims = [nz + 2 * r, ny + 2 * r, nx + 2 * r];
    let [pz, py, px] = pdims;
    let mut out = vec![0.0_f32; pz * py * px];
    let clamp = |v: isize, n: usize| v.clamp(0, n as isize - 1) as usize;
    for z in 0..pz {
        let sz = clamp(z as isize - r as isize, nz);
        for y in 0..py {
            let sy = clamp(y as isize - r as isize, ny);
            for x in 0..px {
                let sx = clamp(x as isize - r as isize, nx);
                out[z * py * px + y * px + x] = data[sz * ny * nx + sy * nx + sx];
            }
        }
    }
    (out, pdims)
}

/// Crop the central `r`-voxel border off a padded `Z×Y×X` volume (inverse of
/// [`pad_replicate_3d`]).
pub(crate) fn crop_border_3d(data: &[f32], pdims: [usize; 3], r: usize) -> (Vec<f32>, [usize; 3]) {
    if r == 0 {
        return (data.to_vec(), pdims);
    }
    let [pz, py, px] = pdims;
    let dims = [pz - 2 * r, py - 2 * r, px - 2 * r];
    let [nz, ny, nx] = dims;
    let mut out = Vec::with_capacity(nz * ny * nx);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                out.push(data[(z + r) * py * px + (y + r) * px + (x + r)]);
            }
        }
    }
    (out, dims)
}

// ── Shared morphological primitive ───────────────────────────────────────────────────────────

/// Generic 3-D morphological neighbourhood scan.
///
/// Iterates a cubic kernel of side `2*radius+1` around each voxel, collecting
/// the fold result of `reduce` over all neighbour values.
///
/// # Invariants
/// - Output length equals `nz * ny * nx`.
/// - Boundary voxels are handled by clamping to the nearest valid index.
pub(super) fn morphological_scan_3d(
    data: &[f32],
    dims: [usize; 3],
    radius: usize,
    init: f32,
    reduce: impl Fn(f32, f32) -> f32,
) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let r = radius as isize;
    let mut output = vec![0.0_f32; nz * ny * nx];

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let mut acc = init;

                for dz in -r..=r {
                    for dy in -r..=r {
                        for dx in -r..=r {
                            let zz = (iz as isize + dz).clamp(0, nz as isize - 1) as usize;
                            let yy = (iy as isize + dy).clamp(0, ny as isize - 1) as usize;
                            let xx = (ix as isize + dx).clamp(0, nx as isize - 1) as usize;
                            acc = reduce(acc, data[zz * ny * nx + yy * nx + xx]);
                        }
                    }
                }

                output[iz * ny * nx + iy * nx + ix] = acc;
            }
        }
    }

    output
}
