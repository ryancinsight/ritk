//! Intensity-weighted center-of-mass initialization for multi-modal registration.
//!
//! Computes the physical center of mass of a 3D image weighted by shifted
//! intensities, providing a robust initial translation estimate for global
//! MI registration without any gradient computation.
//!
//! # Algorithm
//!
//! ```text
//! 1. Shift intensities by global minimum: w_i = intensity_i − min ≥ 0
//!    (handles CT Hounsfield Units which can be strongly negative)
//! 2. Compute intensity-weighted centroid:
//!    CoM = Σ w_i · p_i / Σ w_i
//!    where p_i is the physical coordinate of voxel i
//! 3. Fall back to geometric centre if Σ w_i < 1e-10 (blank / uniform image)
//! ```
//!
//! # Coordinate Convention
//!
//! All returned coordinates follow RITK's `[z, y, x]` ordering in physical
//! units (mm), consistent with `Image::origin()` and `Image::spacing()`.

use burn::tensor::backend::Backend;
use ritk_core::image::Image;

// ─── Center-of-Mass ──────────────────────────────────────────────────────────

/// Compute the intensity-weighted center of mass of a 3-D image.
///
/// Returns physical coordinates in RITK `[z, y, x]` order (mm).
///
/// # Algorithm
///
/// 1. Find the global minimum intensity `min_val`.
/// 2. For each voxel `(iz, iy, ix)`, compute weight `w = (I − min_val).max(0)`.
/// 3. Accumulate `weighted_sum[d] += w · phys_coord[d]` and `total_weight += w`.
/// 4. Return `weighted_sum / total_weight`.
/// 5. **Fallback**: if `total_weight < 1e-10` (blank or uniform image), return
///    the unweighted geometric centre `origin[d] + (n[d]−1)/2 · spacing[d]`.
///
/// # Physical Coordinate Mapping
///
/// Voxel `(iz, iy, ix)` maps to physical space as:
/// ```text
/// phys_z = origin[0] + iz · spacing[0]
/// phys_y = origin[1] + iy · spacing[1]
/// phys_x = origin[2] + ix · spacing[2]
/// ```
///
/// # Example
///
/// ```rust,ignore
/// let com = compute_center_of_mass(&ct_image);
/// println!("CoM [z, y, x] = [{:.1}, {:.1}, {:.1}] mm", com[0], com[1], com[2]);
/// ```
pub fn compute_center_of_mass<B: Backend>(image: &Image<B, 3>) -> [f64; 3] {
    let data = image.data_slice();
    let shape = image.shape();
    let origin = image.origin();
    let spacing = image.spacing();
    let direction = image.direction();

    let nz = shape[0];
    let ny = shape[1];
    let nx = shape[2];

    // Global minimum — enables negative-intensity support (e.g. CT HU values).
    let min_val = data.iter().copied().fold(f32::INFINITY, f32::min) as f64;

    // Weighted-mean voxel index per axis `[d0=z, d1=y, d2=x]`.
    let mut wi = [0.0_f64; 3];
    let mut total_weight = 0.0_f64;
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let w = (data[iz * ny * nx + iy * nx + ix] as f64 - min_val).max(0.0);
                wi[0] += w * iz as f64;
                wi[1] += w * iy as f64;
                wi[2] += w * ix as f64;
                total_weight += w;
            }
        }
    }

    let mean_idx = if total_weight < 1e-10 {
        // Blank/uniform image — unweighted geometric centre index.
        [
            (nz as f64 - 1.0) * 0.5,
            (ny as f64 - 1.0) * 0.5,
            (nx as f64 - 1.0) * 0.5,
        ]
    } else {
        [
            wi[0] / total_weight,
            wi[1] / total_weight,
            wi[2] / total_weight,
        ]
    };

    // Map the mean voxel index to a PHYSICAL world point in LPS `[px, py, pz]`
    // (matching `Image::index_to_world_tensor` and `RigidTransform`'s world space):
    //   world[c] = origin[c] + Σ_axis mean_idx[axis] · spacing[axis] · direction[(c, axis)]
    // The previous implementation paired the physical origin with index extents in
    // the wrong axis order and ignored the direction matrix, returning a scrambled
    // centre that placed the rotation centre / CoM seed in the wrong frame.
    let mut world = [0.0_f64; 3];
    for (c, wc) in world.iter_mut().enumerate() {
        let mut acc = origin[c];
        for (axis, &mi) in mean_idx.iter().enumerate() {
            acc += mi * spacing[axis] * direction[(c, axis)];
        }
        *wc = acc;
    }
    world
}

// ─── Translation Initialisation ──────────────────────────────────────────────

/// Compute the translation that aligns the center of mass of the moving image
/// with that of the fixed image.
///
/// Returns `[tx, ty, tz]` in mm — a PHYSICAL world-space (LPS) vector matching
/// [`compute_center_of_mass`], `Image::index_to_world_tensor`, and the world
/// space `RigidTransform` operates in. Adding it to the initial translation
/// parameter places the moving CoM at the fixed CoM before optimisation begins.
///
/// # Formula
///
/// ```text
/// translation[i] = CoM_fixed[i] − CoM_moving[i],  i ∈ {x, y, z} (world LPS)
/// ```
///
/// # Notes
///
/// This is a zeroth-order initialisation: it is exact when the two images
/// represent the same anatomy in different modalities and the objects occupy
/// a similar fraction of the field of view. For strongly asymmetric images
/// (e.g. whole-body CT vs. brain MRI) the estimate may be unreliable; in that
/// case use `init_strategy = InitStrategy::Manual` in `CmaMiConfig` and supply an explicit
/// initial translation.
pub fn translation_from_centers_of_mass<B: Backend>(
    fixed: &Image<B, 3>,
    moving: &Image<B, 3>,
) -> [f64; 3] {
    let com_fixed = compute_center_of_mass(fixed);
    let com_moving = compute_center_of_mass(moving);
    [
        com_fixed[0] - com_moving[0],
        com_fixed[1] - com_moving[1],
        com_fixed[2] - com_moving[2],
    ]
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;
    use ritk_core::image::Image;
    use ritk_spatial::{Direction, Point, Spacing};

    type B = NdArray<f32>;

    /// Helper: build a 3-D image from a flat `f32` vec with explicit metadata.
    fn make_image(
        data: Vec<f32>,
        shape: [usize; 3],
        spacing: [f64; 3],
        origin: [f64; 3],
    ) -> Image<B, 3> {
        let device = Default::default();
        let tensor = Tensor::<B, 3>::from_data(TensorData::new(data, Shape::new(shape)), &device);
        Image::new(
            tensor,
            Point::new(origin),
            Spacing::new(spacing),
            Direction::identity(),
        )
    }

    // ── compute_center_of_mass ────────────────────────────────────────────────

    /// A 3×3×3 cube of uniform `1.0` values: after min-shift all weights are
    /// zero, so the fallback geometric centre is used.
    ///
    /// Geometric centre index = (3−1)/2 = 1 per axis;
    /// with spacing 2 and origin 0: phys = 0 + 1·2 = **2.0**.
    #[test]
    fn test_com_uniform_cube_at_origin() {
        let data = vec![1.0f32; 27];
        let image = make_image(data, [3, 3, 3], [2.0, 2.0, 2.0], [0.0, 0.0, 0.0]);
        let com = compute_center_of_mass(&image);
        assert!(
            (com[0] - 2.0).abs() < 1e-9,
            "CoM z = {:.6} (expected 2.0)",
            com[0]
        );
        assert!(
            (com[1] - 2.0).abs() < 1e-9,
            "CoM y = {:.6} (expected 2.0)",
            com[1]
        );
        assert!(
            (com[2] - 2.0).abs() < 1e-9,
            "CoM x = {:.6} (expected 2.0)",
            com[2]
        );
    }

    /// 3×3×3 zeros except a single bright voxel at `(iz=2, iy=2, ix=2) = 100`.
    ///
    /// Only that voxel carries non-zero weight, so the CoM equals its
    /// physical position: `0 + 2·1 = 2.0` on every axis.
    #[test]
    fn test_com_single_bright_voxel() {
        let mut data = vec![0.0f32; 27];
        // iz=2, iy=2, ix=2  →  flat index = 2·9 + 2·3 + 2 = 26
        data[26] = 100.0;
        let image = make_image(data, [3, 3, 3], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]);
        let com = compute_center_of_mass(&image);
        assert!(
            (com[0] - 2.0).abs() < 1e-9,
            "CoM z = {:.6} (expected 2.0)",
            com[0]
        );
        assert!(
            (com[1] - 2.0).abs() < 1e-9,
            "CoM y = {:.6} (expected 2.0)",
            com[1]
        );
        assert!(
            (com[2] - 2.0).abs() < 1e-9,
            "CoM x = {:.6} (expected 2.0)",
            com[2]
        );
    }

    /// 2×1×1 image with values `[−10, 10]` verifies negative-intensity handling.
    ///
    /// `min_val = −10`.  After shift: weights = `[0, 20]`.
    /// ```text
    /// CoM_z = (0·0 + 20·1) / 20 = 1.0
    /// ```
    #[test]
    fn test_com_negative_intensity_shift() {
        // shape [nz=2, ny=1, nx=1]
        let data = vec![-10.0f32, 10.0];
        let image = make_image(data, [2, 1, 1], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]);
        let com = compute_center_of_mass(&image);
        assert!(
            (com[0] - 1.0).abs() < 1e-9,
            "CoM z = {:.6} (expected 1.0)",
            com[0]
        );
    }

    // ── translation_from_centers_of_mass ─────────────────────────────────────

    /// Fixed: uniform 3×3×3 → geometric fallback → CoM = (1, 1, 1).
    /// Moving: all zeros except `(0,0,0) = 100` → CoM = (0, 0, 0).
    ///
    /// Expected translation = CoM_fixed − CoM_moving = **(1, 1, 1)**.
    #[test]
    fn test_translation_from_coms() {
        let fixed_data = vec![1.0f32; 27];
        let fixed = make_image(fixed_data, [3, 3, 3], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]);

        let mut moving_data = vec![0.0f32; 27];
        moving_data[0] = 100.0; // voxel (iz=0, iy=0, ix=0)
        let moving = make_image(moving_data, [3, 3, 3], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]);

        let t = translation_from_centers_of_mass(&fixed, &moving);
        assert!(
            (t[0] - 1.0).abs() < 1e-9,
            "t[0] = {:.6} (expected 1.0)",
            t[0]
        );
        assert!(
            (t[1] - 1.0).abs() < 1e-9,
            "t[1] = {:.6} (expected 1.0)",
            t[1]
        );
        assert!(
            (t[2] - 1.0).abs() < 1e-9,
            "t[2] = {:.6} (expected 1.0)",
            t[2]
        );
    }
}
