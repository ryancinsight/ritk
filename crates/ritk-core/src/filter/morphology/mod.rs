//! Grayscale morphological filters for 3-D images.
//!
//! # Mathematical Specification
//!
//! Grayscale morphology extends binary morphology to scalar-valued images by
//! replacing set union/intersection with max/min over a structuring element B:
//!
//! - **Erosion**:  `(E_B f)(x) = min_{b ∈ B} f(x + b)`
//! - **Dilation**: `(D_B f)(x) = max_{b ∈ B} f(x - b)`
//!
//! The structuring element used here is a cubic neighbourhood of half-width
//! `radius`, i.e. B = { b ∈ ℤ³ : |b_i| ≤ r for all i }. Boundary handling
//! uses replicate (clamp) padding.
//!
//! # Derived Operations
//!
//! - **Opening**:  `O_B = D_B ∘ E_B` — removes bright features smaller than B.
//! - **Closing**:  `C_B = E_B ∘ D_B` — removes dark features smaller than B.
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

pub mod binary_erode;
pub mod binary_dilate;
pub mod binary_closing;
pub mod binary_opening;
pub mod binary_fillhole;

pub use binary_erode::BinaryErodeFilter;
pub use binary_dilate::BinaryDilateFilter;
pub use binary_closing::BinaryMorphologicalClosing;
pub use binary_opening::BinaryMorphologicalOpening;
pub use binary_fillhole::BinaryFillholeFilter;

pub mod grayscale_closing;
pub mod grayscale_opening;
pub mod grayscale_fillhole;
pub mod grayscale_gradient;

pub use grayscale_closing::GrayscaleClosingFilter;
pub use grayscale_opening::GrayscaleOpeningFilter;
pub use grayscale_fillhole::GrayscaleFillholeFilter;
pub use grayscale_gradient::GrayscaleMorphologicalGradientFilter;

pub mod grayscale_geodesic;
pub use grayscale_geodesic::{GrayscaleGeodesicDilationFilter, GrayscaleGeodesicErosionFilter};
