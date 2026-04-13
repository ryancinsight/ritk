//! Binary dilation morphological operation.
//!
//! # Mathematical Specification
//!
//! Binary dilation with a box structuring element of half-width `radius` r:
//!
//!   (M ⊕ B)(p) = 1  iff  ∃q ∈ N_r(p): M(q) = 1
//!
//! where N_r(p) is the set of voxels within Chebyshev distance r of p
//! (the axis-aligned hypercube of side 2r+1 centred at p).
//!
//! Out-of-bounds neighbours are ignored (treated as non-contributors), so the
//! structuring element is clipped at image boundaries.
//!
//! # Complexity
//! O(n · (2r+1)^D) where n is the total voxel count.
//!
//! # Supported dimensionalities
//! D = 1, 2, 3.  For D outside this set the function panics with a clear message.

use crate::image::Image;
use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};

/// Binary dilation with a box structuring element of half-width `radius` voxels.
///
/// For each voxel p, output[p] = 1.0 iff at least one voxel within the
/// axis-aligned hypercube of half-width `radius` centred at p is foreground
/// (value > 0.5).
///
/// Out-of-bounds positions in the structuring element are skipped (they do not
/// contribute to dilation), so boundary voxels may still be set if an in-bounds
/// neighbour is foreground.
pub struct BinaryDilation {
    /// Half-width of the box structuring element in voxels.
    /// Radius 0 → structuring element = {p} → dilation is the identity.
    /// Radius 1 → 3^D neighbourhood.
    pub radius: usize,
}

impl BinaryDilation {
    /// Create a `BinaryDilation` with the given structuring-element radius.
    pub fn new(radius: usize) -> Self {
        Self { radius }
    }

    /// Apply dilation to a binary mask image.
    ///
    /// Supports D = 1, 2, 3.  Panics for other dimensionalities.
    pub fn apply<B: Backend, const D: usize>(&self, mask: &Image<B, D>) -> Image<B, D> {
        let shape: [usize; D] = mask.shape();
        let device = mask.data().device();

        let mask_data = mask.data().clone().into_data();
        let flat = mask_data.as_slice::<f32>().expect("f32 mask tensor data");

        let output = dilate_nd(flat, &shape, self.radius);

        let tensor = Tensor::<B, D>::from_data(TensorData::new(output, Shape::new(shape)), &device);

        Image::new(
            tensor,
            mask.origin().clone(),
            mask.spacing().clone(),
            mask.direction().clone(),
        )
    }
}

impl Default for BinaryDilation {
    fn default() -> Self {
        Self::new(1)
    }
}

impl<B: Backend, const D: usize> super::MorphologicalOperation<B, D> for BinaryDilation {
    fn apply(&self, mask: &Image<B, D>) -> Image<B, D> {
        self.apply(mask)
    }
}

// ── Core CPU-side dilation ────────────────────────────────────────────────────

/// Apply binary dilation on a flat row-major array for shapes of rank 1, 2, or 3.
///
/// Panics for ranks other than 1, 2, 3.
pub(super) fn dilate_nd(flat: &[f32], shape: &[usize], radius: usize) -> Vec<f32> {
    match shape.len() {
        1 => dilate_1d(flat, shape[0], radius),
        2 => dilate_2d(flat, shape[0], shape[1], radius),
        3 => dilate_3d(flat, shape[0], shape[1], shape[2], radius),
        d => panic!("BinaryDilation: unsupported dimensionality D={d}; only D=1,2,3 are supported"),
    }
}

// ── D = 1 ─────────────────────────────────────────────────────────────────────

fn dilate_1d(flat: &[f32], nx: usize, radius: usize) -> Vec<f32> {
    let r = radius as isize;
    let mut output = vec![0.0_f32; nx];

    for ix in 0..nx {
        let any_fg = ((-r)..=r).any(|dx| {
            let nb = ix as isize + dx;
            if nb < 0 || nb >= nx as isize {
                return false; // out-of-bounds → skip
            }
            flat[nb as usize] > 0.5
        });
        if any_fg {
            output[ix] = 1.0;
        }
    }

    output
}

// ── D = 2 ─────────────────────────────────────────────────────────────────────

fn dilate_2d(flat: &[f32], ny: usize, nx: usize, radius: usize) -> Vec<f32> {
    let r = radius as isize;
    let mut output = vec![0.0_f32; ny * nx];

    for iy in 0..ny {
        for ix in 0..nx {
            let any_fg = 'outer: {
                for dy in (-r)..=r {
                    for dx in (-r)..=r {
                        let ny_i = iy as isize + dy;
                        let nx_i = ix as isize + dx;
                        if ny_i < 0 || ny_i >= ny as isize || nx_i < 0 || nx_i >= nx as isize {
                            continue; // out-of-bounds → skip
                        }
                        if flat[ny_i as usize * nx + nx_i as usize] > 0.5 {
                            break 'outer true;
                        }
                    }
                }
                false
            };
            if any_fg {
                output[iy * nx + ix] = 1.0;
            }
        }
    }

    output
}

// ── D = 3 ─────────────────────────────────────────────────────────────────────

fn dilate_3d(flat: &[f32], nz: usize, ny: usize, nx: usize, radius: usize) -> Vec<f32> {
    let r = radius as isize;
    let mut output = vec![0.0_f32; nz * ny * nx];

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let any_fg = 'outer: {
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
                                    continue; // out-of-bounds → skip
                                }
                                let nb =
                                    nz_i as usize * ny * nx + ny_i as usize * nx + nx_i as usize;
                                if flat[nb] > 0.5 {
                                    break 'outer true;
                                }
                            }
                        }
                    }
                    false
                };
                if any_fg {
                    output[iz * ny * nx + iy * nx + ix] = 1.0;
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

    // ── radius = 0 is identity ─────────────────────────────────────────────────

    #[test]
    fn test_radius0_is_identity_3d() {
        // Structuring element {p} → output = input for any binary mask.
        let data: Vec<f32> = (0u8..27)
            .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
            .collect();
        let mask = make_mask_3d(data.clone(), [3, 3, 3]);
        let result = BinaryDilation::new(0).apply(&mask);
        assert_eq!(
            values_3d(&result),
            data,
            "radius=0 dilation must be identity"
        );
    }

    #[test]
    fn test_radius0_is_identity_1d() {
        let data = vec![1.0, 0.0, 1.0, 1.0, 0.0];
        let mask = make_mask_1d(data.clone());
        let result = BinaryDilation::new(0).apply(&mask);
        assert_eq!(
            values_1d(&result),
            data,
            "radius=0 dilation must be identity"
        );
    }

    // ── Single isolated voxel grows to neighbourhood ──────────────────────────

    #[test]
    fn test_single_center_voxel_5x5x5_dilates_to_box_r1() {
        // Center voxel (2,2,2) in 5×5×5: dilation r=1 → 3×3×3 box = 27 voxels.
        let mut values = vec![0.0_f32; 125];
        values[2 * 25 + 2 * 5 + 2] = 1.0; // index of (2,2,2) in [5,5,5]
        let mask = make_mask_3d(values, [5, 5, 5]);
        let result = BinaryDilation::new(1).apply(&mask);
        assert_eq!(
            count_fg_3d(&result),
            27,
            "single center voxel r=1 dilation must produce 3×3×3 = 27 foreground voxels"
        );
    }

    #[test]
    fn test_single_corner_voxel_3x3x3_dilates_to_corner_box() {
        // Corner voxel (0,0,0) in 3×3×3: dilation r=1 clips to 2×2×2 = 8 voxels
        // (the full 3×3×3 neighbour box exceeds the image boundary on 3 sides).
        let mut values = vec![0.0_f32; 27];
        values[0] = 1.0;
        let mask = make_mask_3d(values, [3, 3, 3]);
        let result = BinaryDilation::new(1).apply(&mask);
        // The r=1 box around (0,0,0) in a [3,3,3] image is [0..=1, 0..=1, 0..=1] = 8 voxels.
        assert_eq!(
            count_fg_3d(&result),
            8,
            "corner voxel r=1 dilation must cover 2×2×2 = 8 voxels"
        );
    }

    // ── Extensivity invariant: input ⊆ dilated ────────────────────────────────

    #[test]
    fn test_dilation_is_extensive() {
        // Every foreground voxel in the input must remain foreground after dilation.
        let values: Vec<f32> = (0u8..27)
            .map(|i| if i % 5 == 0 { 1.0 } else { 0.0 })
            .collect();
        let mask = make_mask_3d(values.clone(), [3, 3, 3]);
        let result = BinaryDilation::new(1).apply(&mask);
        let result_vals = values_3d(&result);

        for (i, (&orig, &out)) in values.iter().zip(result_vals.iter()).enumerate() {
            if orig > 0.5 {
                assert!(
                    out > 0.5,
                    "dilation removed foreground voxel at index {}",
                    i
                );
            }
        }
    }

    // ── All-foreground input stays all-foreground ─────────────────────────────

    #[test]
    fn test_all_foreground_stays_all_foreground() {
        // Dilating a fully-foreground mask changes nothing.
        let mask = make_mask_3d(vec![1.0_f32; 27], [3, 3, 3]);
        let result = BinaryDilation::new(1).apply(&mask);
        assert_eq!(
            count_fg_3d(&result),
            27,
            "all-foreground mask must remain fully foreground after dilation"
        );
    }

    // ── All-background stays all-background ───────────────────────────────────

    #[test]
    fn test_all_background_stays_all_background() {
        let mask = make_mask_3d(vec![0.0_f32; 27], [3, 3, 3]);
        let result = BinaryDilation::new(1).apply(&mask);
        assert_eq!(
            count_fg_3d(&result),
            0,
            "all-background mask must remain all-background after dilation"
        );
    }

    // ── 1D dilation: known cases ──────────────────────────────────────────────

    #[test]
    fn test_1d_dilation_r1_known_output() {
        // Input:  [0, 0, 1, 0, 0]
        // r=1 neighbourhood:
        //   i=0: neighbours {0,1} → bg → 0
        //   i=1: neighbours {0,1,2} → i=2 is fg → 1
        //   i=2: neighbours {1,2,3} → i=2 is fg → 1
        //   i=3: neighbours {2,3,4} → i=2 is fg → 1
        //   i=4: neighbours {3,4}   → bg → 0
        // Expected: [0, 1, 1, 1, 0]
        let data = vec![0.0, 0.0, 1.0, 0.0, 0.0];
        let mask = make_mask_1d(data);
        let result = BinaryDilation::new(1).apply(&mask);
        let out = values_1d(&result);
        let expected = vec![0.0, 1.0, 1.0, 1.0, 0.0];
        assert_eq!(out, expected, "1D r=1 dilation output mismatch");
    }

    #[test]
    fn test_1d_dilation_r2_known_output() {
        // Input:  [0, 0, 0, 1, 0, 0, 0]
        // r=2: all voxels within distance 2 of index 3 are set.
        // Expected: [0, 1, 1, 1, 1, 1, 0]
        let data = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let mask = make_mask_1d(data);
        let result = BinaryDilation::new(2).apply(&mask);
        let out = values_1d(&result);
        let expected = vec![0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        assert_eq!(out, expected, "1D r=2 dilation output mismatch");
    }

    #[test]
    fn test_1d_dilation_single_voxel_at_boundary() {
        // Single voxel at index 0: r=1 dilates to indices {0, 1}.
        let data = vec![1.0, 0.0, 0.0, 0.0];
        let mask = make_mask_1d(data);
        let result = BinaryDilation::new(1).apply(&mask);
        let out = values_1d(&result);
        let expected = vec![1.0, 1.0, 0.0, 0.0];
        assert_eq!(out, expected, "boundary single voxel dilation mismatch");
    }

    // ── Double dilation is superset of single dilation ────────────────────────

    #[test]
    fn test_double_dilation_superset_of_single_dilation() {
        // D(D(M)) ⊇ D(M) (monotone).
        let mut values = vec![0.0_f32; 125];
        values[62] = 1.0; // center of 5×5×5
        let mask = make_mask_3d(values, [5, 5, 5]);

        let once = BinaryDilation::new(1).apply(&mask);
        let twice = BinaryDilation::new(1).apply(&once);

        let once_vals = values_3d(&once);
        let twice_vals = values_3d(&twice);

        for (i, (&once_v, &twice_v)) in once_vals.iter().zip(twice_vals.iter()).enumerate() {
            if once_v > 0.5 {
                assert!(
                    twice_v > 0.5,
                    "double dilation removed voxel at index {} that was present after single dilation",
                    i
                );
            }
        }
    }

    // ── Output strictly binary ────────────────────────────────────────────────

    #[test]
    fn test_output_strictly_binary_3d() {
        let values: Vec<f32> = (0u8..27)
            .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
            .collect();
        let mask = make_mask_3d(values, [3, 3, 3]);
        let result = BinaryDilation::new(1).apply(&mask);
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

        let result = BinaryDilation::new(1).apply(&mask);

        assert_eq!(result.origin(), &origin, "origin must be preserved");
        assert_eq!(result.spacing(), &spacing, "spacing must be preserved");
        assert_eq!(
            result.direction(),
            &direction,
            "direction must be preserved"
        );
        assert_eq!(result.shape(), [3, 3, 3], "shape must be preserved");
    }

    // ── Dilation then erosion (closing): foreground grows then shrinks ────────

    #[test]
    fn test_dilation_increases_or_preserves_foreground_count() {
        // Dilation must never decrease the number of foreground voxels.
        let values: Vec<f32> = (0u8..27)
            .map(|i| if i % 4 == 0 { 1.0 } else { 0.0 })
            .collect();
        let orig_count = values.iter().filter(|&&v| v > 0.5).count();
        let mask = make_mask_3d(values, [3, 3, 3]);
        let result = BinaryDilation::new(1).apply(&mask);
        let result_count = count_fg_3d(&result);

        assert!(
            result_count >= orig_count,
            "dilation must not decrease foreground count: before={} after={}",
            orig_count,
            result_count
        );
    }
}
