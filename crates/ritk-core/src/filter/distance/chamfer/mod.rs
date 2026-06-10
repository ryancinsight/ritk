//! Chamfer distance transform (CDT) for 3-D binary volumes.
//!
//! # Mathematical Specification
//!
//! Given a binary image `B : ℤ³ → {0,1}` (1 = foreground), the **chamfer**
//! distance transform is an integer-weighted approximation of the true
//! Euclidean distance transform, computed via two 3-D raster scans
//! (forward + backward) over a 3×3×3 neighbourhood mask.
//!
//! ## Supported metrics
//!
//! | Metric      | Math definition                       | Voxel-step cost |
//! |-------------|---------------------------------------|-----------------|
//! | `Chessboard`| L∞: `max(\|dz\|, \|dy\|, \|dx\|)`     | 1               |
//! | `Taxicab`   | L1: `\|dz\| + \|dy\| + \|dx\|`         | 1 (exact on 3-tap) |
//!
//! Both metrics give an integer-valued distance map with O(N) time and
//! O(1) auxiliary storage (the output buffer itself is reused as the
//! workspace — no additional allocation).
//!
//! # Algorithm
//!
//! Two raster scans over a 3×3×3 mask.
//!
//! **Pass 1 (forward)**: `out(z,y,x) = min_{(a,b,c) ∈ S⁻} out(z+a, y+b, x+c) + w(a,b,c)`
//! where `S⁻ = {−1, 0}³ \ {(0,0,0)}` covers the *predecessor* half of the mask
//! (positions already computed). Foreground voxels are seeded with 0.
//! Background voxels are seeded with `+∞` (encoded as `i32::MAX`).
//!
//! **Pass 2 (backward)**: the *successor* half `S⁺ = {0, +1}³ \ {(0,0,0)}`
//! (positions still to be visited on the way back). The minimum of
//! `pass1` and `pass2` gives the chamfer distance.
//!
//! For `Chessboard` (L∞), the 3-tap successor half is sufficient. For
//! `Taxicab` (L1) the full 13-tap half-mask is required for exactness
//! on diagonals.
//!
//! # Properties
//!
//! - **Exact** for taxicab (L1) on a uniform grid: the chamfer distance
//!   equals `|dz| + |dy| + |dx|`.
//! - **Exact** for chessboard (L∞) on a uniform grid: the chamfer distance
//!   equals `max(|dz|, |dy|, |dx|)`.
//!
//! # Spacing
//!
//! Physical spacing is supported. The voxel-step weight along axis `a` is
//! `w_a = round(s_a / s_min)`, so the chamfer distance approximates the
//! physical L1 / L∞ distance in millimetres up to integer rounding.
//!
//! # Comparison to exact EDT
//!
//! | Property              | CDT (this)        | EDT (`euclidean.rs`)         |
//! |-----------------------|-------------------|------------------------------|
//! | Distance metric       | L1 or L∞ (integer)| L2 (float, mm)               |
//! | Time                  | O(N)              | O(N)                         |
//! | Per-voxel cost        | ~24 integer adds  | 1-D phase + parabolic sweep  |
//! | Result type           | `i32` (scaled mm) | `f32` (exact mm)             |
//! | Best use              | Skeletonization,  | Surface extraction,          |
//! |                       | fast thresholds   | level sets, registration     |
//!
//! # References
//!
//! - Borgefors, G. (1984). "Distance transformations in arbitrary
//!   dimensions." *Computer Vision, Graphics, and Image Processing*,
//!   27(3), 321–345.
//! - Borgefors, G. (1986). "Distance transformations in digital images."
//!   *Computer Vision, Graphics, and Image Processing*, 34(3), 344–371.

pub mod kernel;
pub mod transform;

pub use kernel::{
    cdt_3d, cdt_3d_dispatch, chamfer_distance_transform_3d, chamfer_distance_transform_3d_generic,
    ChamferKernel, ChamferMetric, Chessboard, Taxicab, INF,
};
pub use transform::ChamferDistanceTransform;

#[cfg(test)]
mod tests;
