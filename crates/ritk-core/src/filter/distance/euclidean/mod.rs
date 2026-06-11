//! Exact Euclidean distance transform using the Meijster–Roerdink–Hesselink (2000) algorithm.
//!
//! # Mathematical Specification
//!
//! Given a binary image `B : ℤ³ → {0,1}` (1 = foreground), the **unsigned** Euclidean
//! distance transform is:
//!
//! `EDT(x) = min_{y ∈ S} ||x − y||₂`
//!
//! where `S = { y : B(y) = 1 }` and distances are in physical units (mm).
//!
//! The **signed** transform uses the convention:
//!
//! - `x ∉ S` (background): `SEDT(x) = +EDT(x)` (positive = outside)
//! - `x ∈ S` (foreground): `SEDT(x) = −EDT_bg(x)` (negative = inside, where `EDT_bg` is
//!   distance to nearest background voxel)
//!
//! # Algorithm
//!
//! Meijster, A., Roerdink, J.B.T.M., Hesselink, W.H. (2000). "A General Algorithm for
//! Computing Distance Transforms in Linear Time." *Mathematical Morphology and its
//! Applications to Image and Signal Processing*, Springer, pp. 331–340.
//!
//! The algorithm decomposes the 3-D EDT into three sequential 1-D passes, each of which
//! applies a parabolic lower-envelope sweep in O(N). Total complexity: O(N) time, O(N) space.
//!
//! # ITK / SimpleITK Parity
//!
//! | Filter | ITK class |
//! |----------------------------------------|------------------------------------------|
//! | `DistanceTransformImageFilter` | `DanielssonDistanceMapImageFilter` |
//! | `SignedDistanceTransformImageFilter` | `SignedMaurerDistanceMapImageFilter` |

mod core;
mod signed;
mod unsigned;

pub use signed::SignedDistanceTransformImageFilter;
pub use unsigned::DistanceTransformImageFilter;
#[cfg(test)]
pub(crate) use core::edt_3d;

#[cfg(test)]
#[path = "../tests_euclidean.rs"]
mod tests_euclidean;
