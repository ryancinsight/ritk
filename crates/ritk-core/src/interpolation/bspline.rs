//! B-Spline interpolation implementation.
//!
//! This module provides cubic B-Spline interpolation for smooth sampling
//! of image values at continuous coordinates.

use super::trait_::Interpolator;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Cubic B-Spline basis function.
///
/// The cubic B-Spline kernel is defined as:
/// - (2/3) - |x|^2 + (1/2)|x|^3    for |x| < 1
/// - (1/6)(2 - |x|)^3              for 1 <= |x| < 2
/// - 0                             otherwise
fn cubic_bspline(x: f32) -> f32 {
    let abs_x = x.abs();
    if abs_x < 1.0 {
        (2.0 / 3.0) - abs_x.powi(2) + 0.5 * abs_x.powi(3)
    } else if abs_x < 2.0 {
        let two_minus_x = 2.0 - abs_x;
        (1.0 / 6.0) * two_minus_x.powi(3)
    } else {
        0.0
    }
}

/// Cubic B-Spline interpolator.
///
/// Provides smooth interpolation using cubic B-Spline basis functions.
///
/// When `zero_pad` is `false` (the default), out-of-bounds neighborhood
/// samples are skipped and the remaining in-bounds weights are renormalized,
/// which produces an edge-continuation effect at volume boundaries.
/// When `zero_pad` is `true`, query coordinates that fall outside the valid
/// voxel range `[0, dim-1]` for any dimension return `0.0` immediately,
/// matching the behavior of [`LinearInterpolator`] and
/// [`NearestNeighborInterpolator`] in zero-pad mode.
#[derive(Debug, Clone, Copy)]
pub struct BSplineInterpolator {
    /// If `true`, samples outside the volume boundary return `0.0` instead of
    /// the renormalized edge value.  Mirrors [`LinearInterpolator::zero_pad`]
    /// and [`NearestNeighborInterpolator::zero_pad`].
    pub zero_pad: bool,
}

impl BSplineInterpolator {
    /// Create a new B-Spline interpolator with edge-renormalization (default).
    pub fn new() -> Self {
        Self { zero_pad: false }
    }

    /// Create a B-Spline interpolator that returns `0.0` for out-of-bounds query coordinates.
    pub fn new_zero_pad() -> Self {
        Self { zero_pad: true }
    }

    /// Builder-style setter for the `zero_pad` option.
    pub fn with_zero_pad(mut self, zero_pad: bool) -> Self {
        self.zero_pad = zero_pad;
        self
    }
}

impl Default for BSplineInterpolator {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> Interpolator<B> for BSplineInterpolator {
    fn interpolate<const D: usize>(
        &self,
        data: &Tensor<B, D>,
        indices: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        let device = indices.device();
        let [n_points, rank] = indices.dims();
        assert_eq!(rank, D, "Indices rank must match data dimensionality");
        assert!(
            D == 2 || D == 3,
            "B-Spline interpolation only supports 2D and 3D"
        );

        let shape = data.shape();
        let dims: Vec<usize> = shape.dims;

        // Get all index data at once
        let indices_data = indices.to_data();
        let indices_slice: &[f32] = indices_data.as_slice::<f32>().expect("Indices must be f32");

        // Process each point
        let mut results = Vec::with_capacity(n_points);

        for i in 0..n_points {
            // Get coordinates for this point
            let coords_start = i * D;
            let coords: Vec<f32> = (0..D).map(|d| indices_slice[coords_start + d]).collect();

            let value = if D == 3 {
                interpolate_point_3d(data, &coords, &dims, &device, self.zero_pad)
            } else {
                interpolate_point_2d(data, &coords, &dims, &device, self.zero_pad)
            };
            results.push(value);
        }

        if results.is_empty() {
            Tensor::zeros([0], &device)
        } else {
            Tensor::cat(results, 0)
        }
    }
}

/// 3D B-Spline interpolation for a single point.
///
/// The data tensor layout is `[dim0, dim1, dim2]` and `coords` are
/// `[coord0, coord1, coord2]` indexing the respective dimensions.
///
/// When `zero_pad` is `true` and `floor(coord_d)` lies outside `[0, dim_d - 1]`
/// for any dimension `d`, the function returns `0.0` immediately.
fn interpolate_point_3d<B: Backend, const D: usize>(
    data: &Tensor<B, D>,
    coords: &[f32],
    dims: &[usize],
    device: &B::Device,
    zero_pad: bool,
) -> Tensor<B, 1> {
    let x = coords[0];
    let y = coords[1];
    let z = coords[2];

    // Zero-pad early exit: if the query coordinate itself is outside the volume,
    // return 0.0 immediately.  This mirrors the Linear and NearestNeighbor
    // zero-pad semantics where `floor(coord) == clamp(floor(coord), 0, dim-1)`
    // is the in-bounds criterion.
    if zero_pad {
        let xf = x.floor() as isize;
        let yf = y.floor() as isize;
        let zf = z.floor() as isize;
        if xf < 0
            || xf >= dims[0] as isize
            || yf < 0
            || yf >= dims[1] as isize
            || zf < 0
            || zf >= dims[2] as isize
        {
            return Tensor::zeros([1], device);
        }
    }

    let x0 = x.floor() as isize - 1;
    let y0 = y.floor() as isize - 1;
    let z0 = z.floor() as isize - 1;

    let mut result = Tensor::zeros([1], device);
    let mut weight_sum = 0.0f32;

    // Sample 4x4x4 neighborhood
    for dz in 0..4 {
        for dy in 0..4 {
            for dx in 0..4 {
                let xi = x0 + dx;
                let yi = y0 + dy;
                let zi = z0 + dz;

                // Compute B-Spline weights
                let wx = cubic_bspline(x - xi as f32);
                let wy = cubic_bspline(y - yi as f32);
                let wz = cubic_bspline(z - zi as f32);
                let weight = wx * wy * wz;

                // Check bounds and sample
                // Performance: use slice without clone to avoid O(volume_size) allocation.
                // The slice operation creates a view, and we only need to clone the single
                // element we're sampling, not the entire volume.
                if xi >= 0
                    && xi < dims[0] as isize
                    && yi >= 0
                    && yi < dims[1] as isize
                    && zi >= 0
                    && zi < dims[2] as isize
                {
                    let sample = data.clone().slice([
                        xi as usize..xi as usize + 1,
                        yi as usize..yi as usize + 1,
                        zi as usize..zi as usize + 1,
                    ]);
                    let sample_scalar = sample.reshape([1]);
                    result = result.add(sample_scalar.mul_scalar(weight));
                    weight_sum += weight;
                }
            }
        }
    }

    // Normalize by weight sum (handles boundary renormalization when neighbors are OOB)
    if weight_sum > 0.0 {
        result = result.div_scalar(weight_sum);
    }

    result
}

/// 2D B-Spline interpolation for a single point.
///
/// When `zero_pad` is `true` and `floor(coord_d)` lies outside `[0, dim_d - 1]`
/// for any dimension `d`, the function returns `0.0` immediately.
fn interpolate_point_2d<B: Backend, const D: usize>(
    data: &Tensor<B, D>,
    coords: &[f32],
    dims: &[usize],
    device: &B::Device,
    zero_pad: bool,
) -> Tensor<B, 1> {
    let x = coords[0];
    let y = coords[1];

    // Zero-pad early exit: if the query coordinate itself is outside the image.
    if zero_pad {
        let xf = x.floor() as isize;
        let yf = y.floor() as isize;
        if xf < 0 || xf >= dims[0] as isize || yf < 0 || yf >= dims[1] as isize {
            return Tensor::zeros([1], device);
        }
    }

    let x0 = x.floor() as isize - 1;
    let y0 = y.floor() as isize - 1;

    let mut result = Tensor::zeros([1], device);
    let mut weight_sum = 0.0f32;

    // Sample 4x4 neighborhood
    for dy in 0..4 {
        for dx in 0..4 {
            let xi = x0 + dx;
            let yi = y0 + dy;

            // Compute B-Spline weights
            let wx = cubic_bspline(x - xi as f32);
            let wy = cubic_bspline(y - yi as f32);
            let weight = wx * wy;

            // Check bounds and sample
            // Performance: use slice without clone to avoid O(image_size) allocation.
            if xi >= 0 && xi < dims[0] as isize && yi >= 0 && yi < dims[1] as isize {
                let sample = data
                    .clone()
                    .slice([xi as usize..xi as usize + 1, yi as usize..yi as usize + 1]);
                let sample_scalar = sample.reshape([1]);
                result = result.add(sample_scalar.mul_scalar(weight));
                weight_sum += weight;
            }
        }
    }

    // Normalize by weight sum
    if weight_sum > 0.0 {
        result = result.div_scalar(weight_sum);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{ElementConversion, Tensor};
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_bspline_3d() {
        let device = Default::default();

        // Create a simple 3D volume
        let data = Tensor::<TestBackend, 3>::from_floats(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            &device,
        );

        let interpolator = BSplineInterpolator::new();

        // Test at exact grid point
        // Note: Without B-spline pre-filtering, the interpolated value at grid points
        // may differ from original data due to the convolution with the B-spline kernel
        let indices = Tensor::<TestBackend, 2>::from_floats([[0.0, 0.0, 0.0]], &device);
        let result = interpolator.interpolate(&data, indices);
        let val = result.into_scalar().elem::<f32>();
        // Value should be within reasonable range (cubic B-spline center coefficient is 2/3)
        assert!(
            (0.0..=8.0).contains(&val),
            "Interpolated value {} out of range",
            val
        );

        // Test at interpolated point
        let indices = Tensor::<TestBackend, 2>::from_floats([[0.5, 0.5, 0.5]], &device);
        let result = interpolator.interpolate(&data, indices);
        let val = result.into_scalar().elem::<f32>();
        // Value should be between min and max
        assert!(
            (0.0..=8.0).contains(&val),
            "Interpolated value {} out of range",
            val
        );
    }

    #[test]
    fn test_bspline_2d() {
        let device = Default::default();

        // Create a simple 2D image
        let data = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);

        let interpolator = BSplineInterpolator::new();

        // Test at exact grid point
        // Note: Without B-spline pre-filtering, the interpolated value at grid points
        // may differ from original data due to the convolution with the B-spline kernel
        let indices = Tensor::<TestBackend, 2>::from_floats([[0.0, 0.0]], &device);
        let result = interpolator.interpolate(&data, indices);
        let val = result.into_scalar().elem::<f32>();
        // Value should be within reasonable range
        assert!(
            (0.0..=5.0).contains(&val),
            "Interpolated value {} out of range",
            val
        );
    }

    #[test]
    fn test_bspline_basis() {
        // Test B-Spline basis properties
        assert!((cubic_bspline(0.0) - 2.0 / 3.0).abs() < 1e-6);
        assert!(cubic_bspline(1.0) > 0.0);
        assert_eq!(cubic_bspline(2.0), 0.0);
        assert_eq!(cubic_bspline(-2.0), 0.0);
        assert_eq!(cubic_bspline(3.0), 0.0);

        // Symmetry
        assert!((cubic_bspline(0.5) - cubic_bspline(-0.5)).abs() < 1e-6);
    }

    // ---- zero_pad tests ------------------------------------------------

    #[test]
    fn test_bspline_zero_pad_3d_oob_returns_zero() {
        let device = Default::default();
        let data = Tensor::<TestBackend, 3>::from_floats(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            &device,
        );
        let interp = BSplineInterpolator::new_zero_pad();

        // Clearly out-of-bounds queries in each direction.
        let oob = Tensor::<TestBackend, 2>::from_floats(
            [
                [-5.0, 0.0, 0.0], // dim0 OOB negative
                [10.0, 0.0, 0.0], // dim0 OOB positive
                [0.0, -5.0, 0.0], // dim1 OOB
                [0.0, 0.0, 10.0], // dim2 OOB
            ],
            &device,
        );
        let result = interp.interpolate(&data, oob);
        let s = result.into_data().as_slice::<f32>().unwrap().to_vec();
        for (i, v) in s.iter().enumerate() {
            assert!(
                v.abs() < 1e-6,
                "OOB 3D sample {} should give 0.0, got {}",
                i,
                v
            );
        }
    }

    #[test]
    fn test_bspline_zero_pad_3d_inbounds_matches_no_pad() {
        // In-bounds queries should produce the same result regardless of zero_pad flag.
        let device = Default::default();
        let data = Tensor::<TestBackend, 3>::from_floats(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            &device,
        );
        let interp_pad = BSplineInterpolator::new_zero_pad();
        let interp_nop = BSplineInterpolator::new();

        // Interior point; floor coords are (0,0,0) which is in-bounds.
        let pt = Tensor::<TestBackend, 2>::from_floats([[0.5, 0.5, 0.5]], &device);

        let val_pad = interp_pad
            .interpolate(&data, pt.clone())
            .into_data()
            .as_slice::<f32>()
            .unwrap()[0];
        let val_nop = interp_nop
            .interpolate(&data, pt)
            .into_data()
            .as_slice::<f32>()
            .unwrap()[0];

        assert!(
            (val_pad - val_nop).abs() < 1e-5,
            "In-bounds zero_pad {} vs no-pad {} should match",
            val_pad,
            val_nop
        );
        assert!(
            (0.0..=8.0).contains(&val_pad),
            "In-bounds value {} out of range",
            val_pad
        );
    }

    #[test]
    fn test_bspline_zero_pad_2d_oob_returns_zero() {
        let device = Default::default();
        let data =
            Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
        let interp = BSplineInterpolator::new_zero_pad();

        // OOB in both dimensions.
        let oob = Tensor::<TestBackend, 2>::from_floats(
            [[-1.0, 0.0], [0.0, -1.0], [10.0, 0.0], [0.0, 10.0]],
            &device,
        );
        let result = interp.interpolate(&data, oob);
        let s = result.into_data().as_slice::<f32>().unwrap().to_vec();
        for (i, v) in s.iter().enumerate() {
            assert!(
                v.abs() < 1e-6,
                "OOB 2D sample {} should give 0.0, got {}",
                i,
                v
            );
        }
    }

    #[test]
    fn test_bspline_no_zero_pad_oob_gives_finite_value() {
        // Without zero_pad, a query just outside the boundary should still
        // produce a finite (non-panic) value thanks to weight renormalization
        // of the in-bounds neighborhood samples.
        let device = Default::default();
        let data = Tensor::<TestBackend, 3>::from_floats(
            [[[10.0, 20.0], [30.0, 40.0]], [[50.0, 60.0], [70.0, 80.0]]],
            &device,
        );
        let interp = BSplineInterpolator::new(); // zero_pad = false

        // Query just outside: floor(-0.1) = -1 (OOB in dim0).
        // The kernel neighbourhood still touches in-bounds samples at indices 0,1,2
        // (clipped from the 4-wide support), so weight_sum > 0 and result is finite.
        let pt = Tensor::<TestBackend, 2>::from_floats([[-0.1, 0.5, 0.5]], &device);
        let val = interp
            .interpolate(&data, pt)
            .into_data()
            .as_slice::<f32>()
            .unwrap()[0];
        assert!(
            val.is_finite(),
            "No-zero-pad OOB should return finite value, got {}",
            val
        );
    }

    #[test]
    fn test_bspline_with_zero_pad_builder() {
        let interp_true = BSplineInterpolator::new().with_zero_pad(true);
        let interp_false = BSplineInterpolator::new().with_zero_pad(false);
        assert!(interp_true.zero_pad);
        assert!(!interp_false.zero_pad);

        let interp_default = BSplineInterpolator::default();
        assert!(
            !interp_default.zero_pad,
            "Default should have zero_pad=false"
        );
    }
}
