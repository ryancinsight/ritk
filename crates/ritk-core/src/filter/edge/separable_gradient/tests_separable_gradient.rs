use super::*;

/// Naive (unsplit) 1-D convolution — the original combined-loop logic.
fn convolve_1d_axis_naive(
    data: &[f32],
    dims: [usize; 3],
    axis: usize,
    kernel: &[f32; 3],
) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let mut out = vec![0.0_f32; n];
    let stride: usize = match axis {
        0 => ny * nx,
        1 => nx,
        2 => 1,
        _ => unreachable!(),
    };
    let dim_len = dims[axis];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let base = iz * ny * nx + iy * nx + ix;
                let pos = match axis {
                    0 => iz,
                    1 => iy,
                    2 => ix,
                    _ => unreachable!(),
                };
                let mut sum = 0.0_f32;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let offset = ki as isize - 1;
                    let neighbor = (pos as isize + offset).clamp(0, dim_len as isize - 1) as usize;
                    let neighbor_flat = (base as isize
                        + (neighbor as isize - pos as isize) * stride as isize)
                        as usize;
                    sum += kv * data[neighbor_flat];
                }
                out[base] = sum;
            }
        }
    }
    out
}

/// Differential test: boundary/interior split matches naive reference
/// for all 3 axes, multiple sizes, both derivative and smoothing kernels.
#[test]
fn test_convolve_split_matches_naive() {
    let deriv: [f32; 3] = [-1.0, 0.0, 1.0];
    let smooth_sobel: [f32; 3] = [1.0, 2.0, 1.0];
    let smooth_prewitt: [f32; 3] = [1.0, 1.0, 1.0];

    for &dims in &[
        [4, 4, 4],
        [3, 5, 7],
        [1, 1, 16],
        [8, 1, 1],
        [1, 10, 1],
        [2, 2, 2],
    ] {
        let n = dims[0] * dims[1] * dims[2];
        let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
        for axis in 0..3 {
            for kernel in [&deriv, &smooth_sobel, &smooth_prewitt] {
                let out_split = convolve_1d_axis(&data, dims, axis, kernel);
                let out_naive = convolve_1d_axis_naive(&data, dims, axis, kernel);
                for i in 0..n {
                    assert!(
                        (out_split[i] - out_naive[i]).abs() < 1e-6,
                        "convolve mismatch: dims={dims:?} axis={axis} kernel={:?} \
                         idx={i} split={} naive={}",
                        kernel,
                        out_split[i],
                        out_naive[i]
                    );
                }
            }
        }
    }
}

/// Single-element axis: convolution with replicate padding clamps all 3 taps
/// to the same value, so the output is (k[-1]+k[0]+k[1])·x.
#[test]
fn test_convolve_single_element_axis() {
    let dims = [1, 1, 1];
    let data = vec![42.0];
    let deriv: [f32; 3] = [-1.0, 0.0, 1.0];
    let smooth: [f32; 3] = [1.0, 2.0, 1.0];
    let out_d = convolve_1d_axis(&data, dims, 2, &deriv);
    let out_s = convolve_1d_axis(&data, dims, 2, &smooth);
    assert_eq!(out_d[0], 0.0, "derivative of single element must be zero");
    assert_eq!(
        out_s[0],
        42.0 * 4.0,
        "smoothing of single element = kernel_sum * x = 4 * 42 = 168"
    );
}

/// Sobel and Prewitt on a constant image yield zero gradient.
#[test]
fn test_constant_image_zero_gradient() {
    let dims = [3, 3, 3];
    let data = vec![5.0_f32; 27];

    let (sz, sy, sx) = gradient_components::<SobelKernel>(&data, dims, &[1.0; 3].into());
    let (pz, py, px) = gradient_components::<PrewittKernel>(&data, dims, &[1.0; 3].into());

    for (i, &v) in sz.iter().enumerate() {
        assert!(v.abs() < 1e-5, "Sobel gz[{}] = {} != 0", i, v);
    }
    for (i, &v) in sy.iter().enumerate() {
        assert!(v.abs() < 1e-5, "Sobel gy[{}] = {} != 0", i, v);
    }
    for (i, &v) in sx.iter().enumerate() {
        assert!(v.abs() < 1e-5, "Sobel gx[{}] = {} != 0", i, v);
    }
    for (i, &v) in pz.iter().enumerate() {
        assert!(v.abs() < 1e-5, "Prewitt gz[{}] = {} != 0", i, v);
    }
    for (i, &v) in py.iter().enumerate() {
        assert!(v.abs() < 1e-5, "Prewitt gy[{}] = {} != 0", i, v);
    }
    for (i, &v) in px.iter().enumerate() {
        assert!(v.abs() < 1e-5, "Prewitt gx[{}] = {} != 0", i, v);
    }
}
