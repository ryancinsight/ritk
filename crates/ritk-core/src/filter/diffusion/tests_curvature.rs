use super::*;
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
        use crate::spatial::{Direction, Point, Spacing};
        let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
        let td = TensorData::new(vals, Shape::new(dims));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        )
    }

    fn image_vals(img: &Image<B, 3>) -> Vec<f32> {
        img.data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec()
    }

    // ── Test 1: constant image must be unchanged ───────────────────────────────

    #[test]
    fn test_constant_image_unchanged() {
        let dims = [8, 8, 8];
        let vals = vec![5.0f32; 8 * 8 * 8];
        let img = make_image(vals.clone(), dims);

        let filter = CurvatureAnisotropicDiffusionFilter::new(CurvatureConfig::default());
        let out = filter.apply(&img).unwrap();
        let out_vals = image_vals(&out);

        let max_diff = out_vals
            .iter()
            .zip(vals.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-5,
            "constant image must be unchanged; max diff = {max_diff}"
        );
    }

    // ── Test 2: linear field must be unchanged ─────────────────────────────────
    // A linear ramp I(x,y,z) = ax+by+cz has zero curvature everywhere.

    #[test]
    fn test_linear_field_unchanged() {
        let [nz, ny, nx] = [10usize, 10, 10];
        let vals: Vec<f32> = (0..nz * ny * nx)
            .map(|i| {
                let ix = (i % nx) as f32;
                let iy = ((i / nx) % ny) as f32;
                let iz = (i / (ny * nx)) as f32;
                0.3 * ix + 0.5 * iy + 0.7 * iz
            })
            .collect();
        let img = make_image(vals.clone(), [nz, ny, nx]);

        let filter = CurvatureAnisotropicDiffusionFilter::new(CurvatureConfig {
            num_iterations: 10,
            time_step: 1.0 / 16.0,
        });
        let out = filter.apply(&img).unwrap();
        let out_vals = image_vals(&out);

        // Interior voxels only (boundaries use one-sided stencils which introduce small errors)
        let mut max_interior_diff = 0.0f32;
        for iz in 1..nz - 1 {
            for iy in 1..ny - 1 {
                for ix in 1..nx - 1 {
                    let i = iz * ny * nx + iy * nx + ix;
                    let diff = (out_vals[i] - vals[i]).abs();
                    max_interior_diff = max_interior_diff.max(diff);
                }
            }
        }
        assert!(
            max_interior_diff < 1e-3,
            "linear field interior should be unchanged; max diff = {max_interior_diff}"
        );
    }

    // ── Test 3: mean conservation ──────────────────────────────────────────────

    #[test]
    fn test_mean_conservation() {
        let dims = [12, 12, 12];
        let n = 12 * 12 * 12;
        // Sinusoidal image with non-trivial curvature.
        let vals: Vec<f32> = (0..n)
            .map(|i| {
                let ix = (i % 12) as f32 / 12.0;
                let iy = ((i / 12) % 12) as f32 / 12.0;
                let iz = (i / 144) as f32 / 12.0;
                (std::f32::consts::PI * ix).sin()
                    * (std::f32::consts::PI * iy).cos()
                    * (std::f32::consts::PI * iz).sin()
                    + 5.0
            })
            .collect();
        let mean_in: f32 = vals.iter().sum::<f32>() / n as f32;

        let img = make_image(vals, dims);
        let filter = CurvatureAnisotropicDiffusionFilter::new(CurvatureConfig {
            num_iterations: 20,
            time_step: 1.0 / 16.0,
        });
        let out = filter.apply(&img).unwrap();
        let out_vals = image_vals(&out);
        let mean_out: f32 = out_vals.iter().sum::<f32>() / n as f32;

        let rel_err = ((mean_out - mean_in) / mean_in).abs();
        assert!(
            rel_err < 1e-2,
            "mean should be approximately conserved; rel err = {rel_err}"
        );
    }

    // ── Test 4: spherical blob smoothed (gradient reduced) ────────────────────

    #[test]
    fn test_spherical_blob_smoothed() {
        let [nz, ny, nx] = [24usize, 24, 24];
        let n = nz * ny * nx;
        let vals: Vec<f32> = (0..n)
            .map(|i| {
                let ix = (i % nx) as f32 - 12.0;
                let iy = ((i / nx) % ny) as f32 - 12.0;
                let iz = (i / (ny * nx)) as f32 - 12.0;
                let r = (ix * ix + iy * iy + iz * iz).sqrt();
                if r < 6.0 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();

        // Compute max gradient magnitude before filtering.
        let max_grad_in = max_gradient_magnitude(&vals, [nz, ny, nx]);

        let img = make_image(vals, [nz, ny, nx]);
        let filter = CurvatureAnisotropicDiffusionFilter::new(CurvatureConfig {
            num_iterations: 10,
            time_step: 1.0 / 16.0,
        });
        let out = filter.apply(&img).unwrap();
        let out_vals = image_vals(&out);
        let max_grad_out = max_gradient_magnitude(&out_vals, [nz, ny, nx]);

        assert!(
            max_grad_out < max_grad_in,
            "spherical blob should be smoothed; max grad before={max_grad_in:.4} after={max_grad_out:.4}"
        );
    }

    // ── Test 5: stability — outputs finite and within intensity range ──────────

    #[test]
    fn test_stability_small_timestep() {
        let [nz, ny, nx] = [10usize, 10, 10];
        let n = nz * ny * nx;
        let vals: Vec<f32> = (0..n)
            .map(|i| {
                let ix = i % nx;
                let iy = (i / nx) % ny;
                let iz = i / (ny * nx);
                ((ix + iy * 3 + iz * 7) % 17) as f32 * 10.0
            })
            .collect();
        let v_min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let v_max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let img = make_image(vals, [nz, ny, nx]);
        let filter = CurvatureAnisotropicDiffusionFilter::new(CurvatureConfig {
            num_iterations: 50,
            time_step: 1.0 / 16.0,
        });
        let out = filter.apply(&img).unwrap();
        let out_vals = image_vals(&out);

        for &v in &out_vals {
            assert!(v.is_finite(), "output contains non-finite value: {v}");
        }
        // Allow a small overshoot margin due to finite-difference approximation.
        let margin = (v_max - v_min) * 0.05;
        for &v in &out_vals {
            assert!(
                v >= v_min - margin && v <= v_max + margin,
                "output value {v} outside input range [{}, {}] (+margin)",
                v_min - margin,
                v_max + margin
            );
        }
    }

    // ── Helper: max gradient magnitude via central differences ─────────────────

    fn max_gradient_magnitude(data: &[f32], dims: [usize; 3]) -> f32 {
        let [nz, ny, nx] = dims;
        let idx = |iz: usize, iy: usize, ix: usize| iz * ny * nx + iy * nx + ix;
        let mut max_g = 0.0f32;
        for iz in 1..nz - 1 {
            for iy in 1..ny - 1 {
                for ix in 1..nx - 1 {
                    let gz = (data[idx(iz + 1, iy, ix)] - data[idx(iz - 1, iy, ix)]) * 0.5;
                    let gy = (data[idx(iz, iy + 1, ix)] - data[idx(iz, iy - 1, ix)]) * 0.5;
                    let gx = (data[idx(iz, iy, ix + 1)] - data[idx(iz, iy, ix - 1)]) * 0.5;
                    let g = (gz * gz + gy * gy + gx * gx).sqrt();
                    max_g = max_g.max(g);
                }
            }
        }
        max_g
    }