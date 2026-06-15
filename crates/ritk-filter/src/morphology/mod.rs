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
pub mod grayscale_opening;

pub use grayscale_closing::GrayscaleClosingFilter;
pub use grayscale_fillhole::GrayscaleFillholeFilter;
pub use grayscale_gradient::GrayscaleMorphologicalGradientFilter;
pub use grayscale_opening::GrayscaleOpeningFilter;

pub mod grayscale_geodesic;
pub use grayscale_geodesic::{GrayscaleGeodesicDilationFilter, GrayscaleGeodesicErosionFilter};

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
