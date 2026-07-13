//! Image comparison metrics for segmentation and image quality evaluation.
//!
//! Provides spatial overlap, surface distance, and image quality metrics:
//! - [`dice_coefficient`]: volumetric overlap between two binary masks.
//! - [`hausdorff_distance`]: maximum surface-to-surface distance.
//! - [`mean_surface_distance`]: symmetric mean surface-to-surface distance.
//! - [`psnr`]: Peak Signal-to-Noise Ratio between two images.
//! - [`ssim`]: Structural Similarity Index (global, Wang et al. 2004).
//!
//! # Mathematical Specification
//!
//! ## Dice Coefficient
//! Given binary masks P (prediction) and G (ground truth):
//!
//!   Dice(P, G) = 2*|P intersect G| / (|P| + |G|)
//!
//! where |.| denotes the voxel count. Returns 1.0 when both masks are empty.
//!
//! ## Hausdorff Distance
//! Given boundary sets dP and dG in physical space:
//!
//!   HD(P, G) = max( max_{p in dP} min_{g in dG} d(p,g),
//!                   max_{g in dG} min_{p in dP} d(g,p) )
//!
//! where d is Euclidean distance in physical coordinates.
//!
//! ## Mean Surface Distance
//! Directed MSD from set A to set B:
//!
//!   MSD(A->B) = (1/|A|) * sum_{a in A} min_{b in B} d(a,b)
//!
//! Symmetric MSD is the mean of the two directed distances. It returns 0.0
//! when both boundary sets are empty.
//!
//! ## Peak Signal-to-Noise Ratio
//!
//!   MSE = (1/N) * sum (I_i - R_i)^2
//!   PSNR = 10 * log10(MAX^2 / MSE)
//!
//! Returns +infinity when MSE = 0.
//!
//! ## Structural Similarity Index
//! Global SSIM follows Wang et al., IEEE Trans. Image Process. 13(4), 2004:
//!
//!   SSIM(x, y) = (2*mu_x*mu_y + C1)(2*sigma_xy + C2)
//!                / ((mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2))
//!
//! This implementation computes a single global SSIM over all voxels.

pub mod native;
mod overlap;
mod quality;
mod surface;

pub use overlap::{dice_coefficient, similarity_index};
pub use quality::{psnr, ssim};
pub use surface::{hausdorff_distance, mean_surface_distance};

#[cfg(test)]
mod tests;
