//! Binary erosion morphological operation.
//!
//! # Mathematical Specification
//!
//! Binary erosion with a box structuring element of half-width `radius` r:
//!
//!   (M ⊖ B)(p) = 1  iff  ∀q ∈ N_r(p): M(q) = 1
//!
//! where N_r(p) is the set of voxels within Chebyshev distance r of p
//! (the axis-aligned hypercube of side 2r+1 centred at p).
//!
//! Out-of-bounds neighbours are treated as background (0), which means
//! any foreground voxel within `r` voxels of the image boundary is eroded.
//!
//! # Complexity
//! O(n · (2r+1)^D) where n is the total voxel count.
//!
//! # Supported dimensionalities
//! D = 1, 2, 3.  For D outside this set the function panics with a clear message.

use crate::image::Image;
use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};

/// Binary erosion with a box structuring element of half-width `radius` voxels.
///
/// For each voxel p, output[p] = 1.0 iff every voxel within the axis-aligned
/// hypercube of half-width `radius` centred at p is foreground (value > 0.5).
///
/// Out-of-bounds neighbours are treated as background, so any foreground voxel
/// within `radius` of the image boundary is removed.
pub struct BinaryErosion {
    /// Half-width of the box structuring element in voxels.
    /// Radius 0 → structuring element = {p} → erosion is the identity.
    /// Radius 1 → 3^D neighbourhood.
    pub radius: usize,
}

impl BinaryErosion {
    /// Create a `BinaryErosion` with the given structuring-element radius.
    pub fn new(radius: usize) -> Self {
        Self { radius }
    }

    /// Apply erosion to a binary mask image.
    ///
    /// Supports D = 1, 2, 3.  Panics for other dimensionalities.
    pub fn apply<B: Backend, const D: usize>(&self, mask: &Image<B, D>) -> Image<B, D> {
        let shape: [usize; D] = mask.shape();
        let device = mask.data().device();

        let mask_data = mask.data().clone().into_data();
        let flat = mask_data.as_slice::<f32>().expect("f32 mask tensor data");

        let output = erode_nd(flat, &shape, self.radius);

        let tensor = Tensor::<B, D>::from_data(TensorData::new(output, Shape::new(shape)), &device);

        Image::new(
            tensor,
            mask.origin().clone(),
            mask.spacing().clone(),
            mask.direction().clone(),
        )
    }
}

impl Default for BinaryErosion {
    fn default() -> Self {
        Self::new(1)
    }
}

impl<B: Backend, const D: usize> super::MorphologicalOperation<B, D> for BinaryErosion {
    fn apply(&self, mask: &Image<B, D>) -> Image<B, D> {
        self.apply(mask)
    }
}

// ── Core CPU-side erosion ─────────────────────────────────────────────────────

/// Apply binary erosion on a flat row-major array for shapes of rank 1, 2, or 3.
///
/// Panics for ranks other than 1, 2, 3.
pub(super) fn erode_nd(flat: &[f32], shape: &[usize], radius: usize) -> Vec<f32> {
    match shape.len() {
        1 => erode_1d(flat, shape[0], radius),
        2 => erode_2d(flat, shape[0], shape[1], radius),
        3 => erode_3d(flat, shape[0], shape[1], shape[2], radius),
        d => panic!("BinaryErosion: unsupported dimensionality D={d}; only D=1,2,3 are supported"),
    }
}

// ── D = 1 ─────────────────────────────────────────────────────────────────────

fn erode_1d(flat: &[f32], nx: usize, radius: usize) -> Vec<f32> {
    let r = radius as isize;
    let mut output = vec![0.0_f32; nx];

    for ix in 0..nx {
        if flat[ix] <= 0.5 {
            continue;
        }
        let all_fg = ((-r)..=r).all(|dx| {
            let nb = ix as isize + dx;
            if nb < 0 || nb >= nx as isize {
                return false; // out-of-bounds → background
            }
            flat[nb as usize] > 0.5
        });
        if all_fg {
            output[ix] = 1.0;
        }
    }

    output
}

// ── D = 2 ─────────────────────────────────────────────────────────────────────

fn erode_2d(flat: &[f32], ny: usize, nx: usize, radius: usize) -> Vec<f32> {
    let r = radius as isize;
    let mut output = vec![0.0_f32; ny * nx];

    for iy in 0..ny {
        for ix in 0..nx {
            let center = iy * nx + ix;
            if flat[center] <= 0.5 {
                continue;
            }
            let all_fg = 'outer: {
                for dy in (-r)..=r {
                    for dx in (-r)..=r {
                        let ny_i = iy as isize + dy;
                        let nx_i = ix as isize + dx;
                        if ny_i < 0 || ny_i >= ny as isize || nx_i < 0 || nx_i >= nx as isize {
                            break 'outer false;
                        }
                        if flat[ny_i as usize * nx + nx_i as usize] <= 0.5 {
                            break 'outer false;
                        }
                    }
                }
                true
            };
            if all_fg {
                output[center] = 1.0;
            }
        }
    }

    output
}

// ── D = 3 ─────────────────────────────────────────────────────────────────────

fn erode_3d(flat: &[f32], nz: usize, ny: usize, nx: usize, radius: usize) -> Vec<f32> {
    let r = radius as isize;
    let mut output = vec![0.0_f32; nz * ny * nx];

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let center = iz * ny * nx + iy * nx + ix;
                if flat[center] <= 0.5 {
                    continue;
                }
                let all_fg = 'outer: {
                    for dz in (-r)..=r {
                        for dy in (-r)..=r {
                            for dx in (-r)..=r {
                                let nz_i = iz as isize + dz;
                                let ny_i = iy as isize + dy;
                                let nx_i = ix as isize + dx;
                                if nz_i < 0
                                    || nz_i >= nz as isize
                                    || ny_i < 0
                                    || ny_i >= ny as isize
                                    || nx_i < 0
                                    || nx_i >= nx as isize
                                {
                                    break 'outer false;
                                }
                                let nb =
                                    nz_i as usize * ny * nx + ny_i as usize * nx + nx_i as usize;
                                if flat[nb] <= 0.5 {
                                    break 'outer false;
                                }
                            }
                        }
                    }
                    true
                };
                if all_fg {
                    output[center] = 1.0;
                }
            }
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn make_mask_1d(data: Vec<f32>) -> Image<TestBackend, 1> {
        let n = data.len();
        let device = Default::default();
        let tensor =
            Tensor::<TestBackend, 1>::from_data(TensorData::new(data, Shape::new([n])), &device);
        Image::new(
            tensor,
            Point::new([0.0]),
            Spacing::new([1.0]),
            Direction::identity(),
        )
    }

    fn make_mask_3d(data: Vec<f32>, dims: [usize; 3]) -> Image<TestBackend, 3> {
        let device = Default::default();
        let tensor =
            Tensor::<TestBackend, 3>::from_data(TensorData::new(data, Shape::new(dims)), &device);
        Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    fn values_3d(image: &Image<TestBackend, 3>) -> Vec<f32> {
        image
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec()
    }

    fn values_1d(image: &Image<TestBackend, 1>) -> Vec<f32> {
        image
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec()
    }

    fn count_fg_3d(image: &Image<TestBackend, 3>) -> usize {
        values_3d(image).iter().filter(|&&v| v > 0.5).count()
    }

    // ── radius = 0 is identity ────────────────────────────────────────────────

    #[test]
    fn test_radius0_is_identity_3d() {
        // Structuring element {p} → output = input for any binary mask.
        let data: Vec<f32> = (0u8..27)
            .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
            .collect();
        let mask = make_mask_3d(data.clone(), [3, 3, 3]);
        let result = BinaryErosion::new(0).apply(&mask);
        assert_eq!(
            values_3d(&result),
            data,
            "radius=0 erosion must be identity"
        );
    }

    #[test]
    fn test_radius0_is_identity_1d() {
        let data = vec![1.0, 0.0, 1.0, 1.0, 0.0];
        let mask = make_mask_1d(data.clone());
        let result = BinaryErosion::new(0).apply(&mask);
        assert_eq!(
            values_1d(&result),
            data,
            "radius=0 erosion must be identity"
        );
    }

    // ── All-foreground large image: interior survives ─────────────────────────

    #[test]
    fn test_all_fg_5x5x5_erosion_r1_keeps_3x3x3_interior() {
        // 5×5×5 all-foreground: r=1 removes the outer shell → 3×3×3 = 27 voxels survive.
        let mask = make_mask_3d(vec![1.0_f32; 125], [5, 5, 5]);
        let result = BinaryErosion::new(1).apply(&mask);
        assert_eq!(
            count_fg_3d(&result),
            27,
            "5×5×5 all-fg erosion r=1 must keep 3×3×3 = 27 voxels"
        );
    }

    #[test]
    fn test_all_fg_7x7x7_erosion_r2_keeps_3x3x3_interior() {
        // 7×7×7 all-fg: r=2 removes 2-voxel shell → 3×3×3 = 27 survive.
        let mask = make_mask_3d(vec![1.0_f32; 343], [7, 7, 7]);
        let result = BinaryErosion::new(2).apply(&mask);
        assert_eq!(
            count_fg_3d(&result),
            27,
            "7×7×7 all-fg erosion r=2 must keep 3×3×3 = 27 voxels"
        );
    }

    // ── Single isolated voxel is fully eroded ─────────────────────────────────

    #[test]
    fn test_single_voxel_eroded_to_empty() {
        // Isolated single foreground voxel in 3×3×3 → fully eroded (boundary).
        let mut values = vec![0.0_f32; 27];
        values[13] = 1.0; // center (1,1,1)
        let mask = make_mask_3d(values, [3, 3, 3]);
        let result = BinaryErosion::new(1).apply(&mask);
        assert_eq!(
            count_fg_3d(&result),
            0,
            "isolated single voxel must be fully eroded"
        );
    }

    // ── Anti-extensivity invariant: eroded ⊆ input ───────────────────────────

    #[test]
    fn test_erosion_is_anti_extensive() {
        let values: Vec<f32> = (0u8..27)
            .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
            .collect();
        let mask = make_mask_3d(values.clone(), [3, 3, 3]);
        let result = BinaryErosion::new(1).apply(&mask);
        let result_vals = values_3d(&result);

        for (i, (&orig, &out)) in values.iter().zip(result_vals.iter()).enumerate() {
            if out > 0.5 {
                assert!(
                    orig > 0.5,
                    "erosion introduced foreground at index {} where input was background",
                    i
                );
            }
        }
    }

    // ── All-background stays all-background ───────────────────────────────────

    #[test]
    fn test_all_background_stays_empty() {
        let mask = make_mask_3d(vec![0.0_f32; 27], [3, 3, 3]);
        let result = BinaryErosion::new(1).apply(&mask);
        assert_eq!(
            count_fg_3d(&result),
            0,
            "all-background mask must remain all-background after erosion"
        );
    }

    // ── 1D erosion: known cases ───────────────────────────────────────────────

    #[test]
    fn test_1d_erosion_r1_known_output() {
        // Input:  [0, 1, 1, 1, 1, 1, 0]
        // The foreground run is [1..=5].  r=1 neighbourhood:
        //   i=1: needs i=0 (bg) → eroded.
        //   i=2: needs i=1 (fg), i=3 (fg) → survives.
        //   i=3: needs i=2 (fg), i=4 (fg) → survives.
        //   i=4: needs i=3 (fg), i=5 (fg) → survives.
        //   i=5: needs i=6 (bg) → eroded.
        // Expected: [0, 0, 1, 1, 1, 0, 0]
        let data = vec![0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        let mask = make_mask_1d(data);
        let result = BinaryErosion::new(1).apply(&mask);
        let out = values_1d(&result);
        let expected = vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
        assert_eq!(out, expected, "1D r=1 erosion output mismatch");
    }

    #[test]
    fn test_1d_all_foreground_erosion_r1() {
        // [1,1,1,1,1] → boundary voxels eroded → [0,1,1,1,0].
        let mask = make_mask_1d(vec![1.0_f32; 5]);
        let result = BinaryErosion::new(1).apply(&mask);
        let out = values_1d(&result);
        assert_eq!(out, vec![0.0, 1.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_1d_single_voxel_eroded() {
        // Single foreground voxel at edge → boundary neighbours missing → eroded.
        let mask = make_mask_1d(vec![1.0]);
        let result = BinaryErosion::new(1).apply(&mask);
        assert_eq!(values_1d(&result), vec![0.0]);
    }

    // ── Output strictly binary ────────────────────────────────────────────────

    #[test]
    fn test_output_strictly_binary_3d() {
        let values: Vec<f32> = (0u8..27)
            .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
            .collect();
        let mask = make_mask_3d(values, [3, 3, 3]);
        let result = BinaryErosion::new(1).apply(&mask);
        for &v in values_3d(&result).iter() {
            assert!(
                v == 0.0 || v == 1.0,
                "output must be strictly binary, got {v}"
            );
        }
    }

    // ── Metadata preservation ─────────────────────────────────────────────────

    #[test]
    fn test_preserves_spatial_metadata() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let tensor = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![1.0f32; 27], Shape::new([3, 3, 3])),
            &device,
        );
        let origin = Point::new([1.0, 2.0, 3.0]);
        let spacing = Spacing::new([0.5, 0.5, 0.5]);
        let direction = Direction::identity();
        let mask: Image<TestBackend, 3> = Image::new(tensor, origin, spacing, direction);

        let result = BinaryErosion::new(1).apply(&mask);

        assert_eq!(result.origin(), &origin, "origin must be preserved");
        assert_eq!(result.spacing(), &spacing, "spacing must be preserved");
        assert_eq!(
            result.direction(),
            &direction,
            "direction must be preserved"
        );
        assert_eq!(result.shape(), [3, 3, 3], "shape must be preserved");
    }

    // ── Idempotency: erode twice ≡ erode once with larger radius (not equal,
    //    but monotone: second erosion is subset of first erosion) ──────────────

    #[test]
    fn test_double_erosion_subset_of_single_erosion() {
        // E(E(M)) ⊆ E(M) for any mask M (monotone).
        let mask = make_mask_3d(vec![1.0_f32; 125], [5, 5, 5]);
        let once = BinaryErosion::new(1).apply(&mask);
        let twice = BinaryErosion::new(1).apply(&once);

        let once_vals = values_3d(&once);
        let twice_vals = values_3d(&twice);

        for (i, (&once_v, &twice_v)) in once_vals.iter().zip(twice_vals.iter()).enumerate() {
            if twice_v > 0.5 {
                assert!(
                    once_v > 0.5,
                    "double erosion result at index {} not subset of single erosion",
                    i
                );
            }
        }
    }
}
