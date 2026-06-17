use super::*;

fn apply_first_derivative_1d_naive(data: &[f32], dims: [usize; 3], dim: usize, out: &mut [f32]) {
    let lp = line_params(dims, dim);
    for li in 0..lp.num_lines {
        let base = line_base(dims, dim, li);
        for i in 0..lp.len {
            let val = if lp.len == 1 {
                0.0
            } else if i == 0 {
                data[base + lp.stride] - data[base]
            } else if i == lp.len - 1 {
                data[base + i * lp.stride] - data[base + (i - 1) * lp.stride]
            } else {
                (data[base + (i + 1) * lp.stride] - data[base + (i - 1) * lp.stride]) * 0.5
            };
            out[base + i * lp.stride] = val;
        }
    }
}

fn apply_second_derivative_1d_naive(data: &[f32], dims: [usize; 3], dim: usize, out: &mut [f32]) {
    let lp = line_params(dims, dim);
    for li in 0..lp.num_lines {
        let base = line_base(dims, dim, li);
        for i in 0..lp.len {
            let val = if lp.len < 3 {
                0.0
            } else if i == 0 {
                let x0 = data[base];
                let x1 = data[base + lp.stride];
                let x2 = data[base + 2 * lp.stride];
                x2 - 2.0 * x1 + x0
            } else if i == lp.len - 1 {
                let xn1 = data[base + i * lp.stride];
                let xn2 = data[base + (i - 1) * lp.stride];
                let xn3 = data[base + (i - 2) * lp.stride];
                xn1 - 2.0 * xn2 + xn3
            } else {
                let xp = data[base + (i + 1) * lp.stride];
                let xc = data[base + i * lp.stride];
                let xm = data[base + (i - 1) * lp.stride];
                xp - 2.0 * xc + xm
            };
            out[base + i * lp.stride] = val;
        }
    }
}

#[test]
fn test_first_derivative_split_matches_naive() {
    for &dims in &[[4, 4, 4], [3, 5, 7], [1, 1, 16], [8, 1, 1], [1, 10, 1]] {
        let n = dims[0] * dims[1] * dims[2];
        let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
        for dim in 0..3 {
            let mut out_split = vec![0.0_f32; n];
            let mut out_naive = vec![0.0_f32; n];
            apply_first_derivative_1d_into(&data, dims, dim, &mut out_split);
            apply_first_derivative_1d_naive(&data, dims, dim, &mut out_naive);
            for i in 0..n {
                assert!(
                    (out_split[i] - out_naive[i]).abs() < 1e-7,
                    "first derivative mismatch: dims={dims:?} dim={dim} idx={i} \
                     split={} naive={}",
                    out_split[i],
                    out_naive[i]
                );
            }
        }
    }
}

#[test]
fn test_second_derivative_split_matches_naive() {
    for &dims in &[
        [4, 4, 4],
        [3, 5, 7],
        [1, 1, 16],
        [8, 1, 1],
        [1, 10, 1],
        [1, 1, 2],
    ] {
        let n = dims[0] * dims[1] * dims[2];
        let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
        for dim in 0..3 {
            let mut out_split = vec![0.0_f32; n];
            let mut out_naive = vec![0.0_f32; n];
            apply_second_derivative_1d_into(&data, dims, dim, &mut out_split);
            apply_second_derivative_1d_naive(&data, dims, dim, &mut out_naive);
            for i in 0..n {
                assert!(
                    (out_split[i] - out_naive[i]).abs() < 1e-7,
                    "second derivative mismatch: dims={dims:?} dim={dim} idx={i} \
                     split={} naive={}",
                    out_split[i],
                    out_naive[i]
                );
            }
        }
    }
}

/// Deriche smoothing has unit DC gain: a constant volume is returned unchanged
/// (every voxel, every axis, across sigmas).
#[test]
fn test_deriche_smooth_preserves_constant() {
    for &dims in &[[4, 4, 4], [3, 5, 7], [2, 2, 16]] {
        let n = dims[0] * dims[1] * dims[2];
        let data = vec![3.7_f32; n];
        for &sigma in &[0.5, 1.0, 2.0, 5.0] {
            let coeffs = DericheCoefficients::from_sigma(sigma);
            for dim in 0..3 {
                let out = apply_deriche_1d(&data, dims, dim, &coeffs, sigma);
                for (i, &v) in out.iter().enumerate() {
                    assert!(
                        (v - 3.7).abs() < 1e-4,
                        "constant not preserved: dims={dims:?} dim={dim} sigma={sigma} idx={i} got {v}"
                    );
                }
            }
        }
    }
}

/// Deriche coefficients are DC-normalised: the causal+anticausal passes sum to
/// unit gain, i.e. (N0+N1+N2+N3 + M1+M2+M3+M4) / (1+D1+D2+D3+D4) = 1.
#[test]
fn test_deriche_unit_dc_gain() {
    for &sigma in &[0.5, 1.0, 2.0, 5.0, 10.0] {
        let c = DericheCoefficients::from_sigma(sigma);
        let sn: f64 = c.n.iter().sum();
        let sm: f64 = c.m.iter().sum();
        let sd: f64 = 1.0 + c.d.iter().sum::<f64>();
        let dc = (sn + sm) / sd;
        assert!(
            (dc - 1.0).abs() < 1e-12,
            "Deriche DC gain {dc} ≠ 1 at sigma={sigma}"
        );
    }
}

#[test]
fn test_first_derivative_single_element() {
    let dims = [1, 1, 1];
    let data = vec![42.0];
    let mut out = vec![0.0; 1];
    apply_first_derivative_1d_into(&data, dims, 2, &mut out);
    assert_eq!(out[0], 0.0, "single-element first derivative must be zero");
}

#[test]
fn test_second_derivative_two_elements() {
    let dims = [1, 1, 2];
    let data = vec![1.0, 2.0];
    let mut out = vec![0.0; 2];
    apply_second_derivative_1d_into(&data, dims, 2, &mut out);
    assert_eq!(
        out[0], 0.0,
        "2-element second derivative must be zero at [0]"
    );
    assert_eq!(
        out[1], 0.0,
        "2-element second derivative must be zero at [1]"
    );
}
