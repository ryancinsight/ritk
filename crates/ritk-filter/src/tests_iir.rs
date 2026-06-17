use super::*;

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
