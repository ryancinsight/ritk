//! Tests for local cross-correlation primitives.

use super::*;

// -- Window statistics tests --

#[test]
fn window_cc_stats_constant_images() {
    let dims = [5usize, 5, 5];
    let a = vec![3.0_f32; 125];
    let b = vec![7.0_f32; 125];
    let (mu_i, mu_j, num, vi, vj, cnt) = window_cc_stats(&a, &b, dims, 2, 2, 2, 1);
    assert!((mu_i - 3.0).abs() < 1e-10, "mu_i = {mu_i}");
    assert!((mu_j - 7.0).abs() < 1e-10, "mu_j = {mu_j}");
    assert!(num.abs() < 1e-10, "num = {num}");
    assert!(vi.abs() < 1e-10, "var_i = {vi}");
    assert!(vj.abs() < 1e-10, "var_j = {vj}");
    assert!(cnt > 0, "count = {cnt}");
}

#[test]
fn window_cc_stats_identical_non_constant() {
    let dims = [6usize, 6, 6];
    let image: Vec<f32> = (0..216).map(|i| i as f32).collect();
    let (_, _, num, di2, dj2, _) = window_cc_stats(&image, &image, dims, 3, 3, 3, 1);
    // For identical images: num = var_i = var_j, so CC = 1.0
    let d = (di2 * dj2).sqrt();
    assert!(d > 1e-10, "denom = {d}");
    let cc = num / d;
    assert!(
        (cc - 1.0).abs() < 1e-10,
        "CC of identical local patches = {cc}"
    );
}

// -- Force computation tests --

#[test]
fn mean_local_cc_constant_images_safe() {
    // Constant images have zero variance â†’ CC should be 0, not NaN.
    let dims = [5usize, 5, 5];
    let n = 5 * 5 * 5;
    let a = vec![3.0_f32; n];
    let b = vec![3.0_f32; n];
    let cc = mean_local_cc(&a, &b, dims, 1);
    assert!(
        cc.is_finite(),
        "CC of constant images must be finite, got {cc}"
    );
    assert!(
        cc.abs() < 1e-10,
        "CC of constant images must be 0, got {cc}"
    );
}

#[test]
fn cc_forces_zero_on_constant_images() {
    // var_i < 1e-10 guard must return zero forces for constant I.
    let dims = [4usize, 4, 4];
    let n = 4 * 4 * 4;
    let a = vec![5.0_f32; n];
    let b = vec![3.0_f32; n];
    let gi = vec![1.0_f32; n];
    let (fz, fy, fx) = cc_forces(&a, &b, &gi, &gi, &gi, dims, 1);
    for &v in fz.iter().chain(fy.iter()).chain(fx.iter()) {
        assert!(v.abs() < 1e-6, "constant-I force must be zero, got {v}");
    }
}

#[test]
fn cc_forces_nonzero_for_shifted_images() {
    // A linearly increasing image vs its shifted version: forces must be
    // non-trivially large (algorithm sees the local intensity gradient).
    let dims = [8usize, 8, 10];
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let fixed: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let shifted: Vec<f32> = (0..n).map(|i| (i + nx) as f32).collect();
    let gi_x: Vec<f32> = vec![1.0_f32; n];
    let gi_zero: Vec<f32> = vec![0.0_f32; n];
    let (_, _, fx) = cc_forces(&fixed, &shifted, &gi_zero, &gi_zero, &gi_x, dims, 1);
    let rms_fx: f64 = field_rms(&fx);
    assert!(
        rms_fx > 0.0,
        "x-forces must be non-zero for an x-gradient image"
    );
}

#[test]
fn mean_local_cc_identical_images_returns_one() {
    let dims = [6usize, 6, 6];
    let n = 6 * 6 * 6;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let cc = mean_local_cc(&a, &a, dims, 1);
    assert!(
        (cc - 1.0).abs() < 1e-8,
        "CC of identical images must be 1.0, got {cc}"
    );
}

#[test]
fn cc_forces_identical_images_bounded() {
    // CC forces on identical images are bounded (at optimum, gradient is small).
    let dims = [6usize, 6, 6];
    let n = 216;
    let image: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let grad = crate::deformable_field_ops::compute_gradient(&image, dims.into(), [1.0, 1.0, 1.0]);
    let (fz, fy, fx) = cc_forces(&image, &image, &grad.z, &grad.y, &grad.x, dims, 1);
    let rms = |f: &[f32]| -> f64 {
        (f.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / n as f64).sqrt()
    };
    assert!(
        rms(&fz) < 10.0 && rms(&fy) < 10.0 && rms(&fx) < 10.0,
        "CC forces on identical images should be bounded"
    );
}

#[test]
fn mean_local_cc_identical_non_constant_images() {
    let dims = [6usize, 6, 6];
    let image: Vec<f32> = (0..216).map(|i| i as f32).collect();
    let cc = mean_local_cc(&image, &image, dims, 1);
    assert!(
        cc > 0.99,
        "CC of identical non-constant images should be â‰ˆ 1.0, got {cc}"
    );
}

#[test]
fn mean_local_cc_constant_images_is_zero() {
    let dims = [5usize, 5, 5];
    let a = vec![3.0_f32; 125];
    let cc = mean_local_cc(&a, &a, dims, 1);
    assert!(cc.is_finite(), "CC must be finite, got {cc}");
    assert!(
        cc.abs() < 1e-6,
        "CC of constant images should be 0, got {cc}"
    );
}

#[test]
fn cc_forces_into_matches_cc_forces() {
    let dims = [6usize, 6, 6];
    let n = 6 * 6 * 6;
    let fixed: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let moving: Vec<f32> = (0..n).map(|i| (i + 6) as f32).collect();
    let gi = crate::deformable_field_ops::compute_gradient(&fixed, dims.into(), [1.0, 1.0, 1.0]);
    let (fz_alloc, fy_alloc, fx_alloc) = cc_forces(&fixed, &moving, &gi.z, &gi.y, &gi.x, dims, 1);
    let mut fz = vec![0.0_f32; n];
    let mut fy = vec![0.0_f32; n];
    let mut fx = vec![0.0_f32; n];
    cc_forces_into(
        &fixed, &moving, &gi.z, &gi.y, &gi.x, dims, 1, &mut fz, &mut fy, &mut fx,
    );
    for i in 0..n {
        assert!(
            (fz[i] - fz_alloc[i]).abs() < 1e-5,
            "fz[{i}] mismatch: into={}, alloc={}",
            fz[i],
            fz_alloc[i]
        );
        assert!(
            (fy[i] - fy_alloc[i]).abs() < 1e-5,
            "fy[{i}] mismatch: into={}, alloc={}",
            fy[i],
            fy_alloc[i]
        );
        assert!(
            (fx[i] - fx_alloc[i]).abs() < 1e-5,
            "fx[{i}] mismatch: into={}, alloc={}",
            fx[i],
            fx_alloc[i]
        );
    }
}

#[test]
fn reversed_shared_sats_match_independent_build() {
    let dims = [6usize, 6, 6];
    let n = dims.iter().product();
    let fixed: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013) % 1.0).collect();
    let moving: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.021) + 0.17) % 1.0).collect();
    let gradient =
        crate::deformable_field_ops::compute_gradient(&moving, dims.into(), [1.0, 1.0, 1.0]);
    let sats = CcSats::build(&fixed, &moving, dims, 1);
    let mut shared = [vec![0.0_f32; n], vec![0.0_f32; n], vec![0.0_f32; n]];
    let [shared_z, shared_y, shared_x] = &mut shared;
    cc_forces_from_sats_into::<true>(
        &moving,
        &fixed,
        &gradient.z,
        &gradient.y,
        &gradient.x,
        dims,
        &sats,
        shared_z,
        shared_y,
        shared_x,
    );
    let mut independent = [vec![0.0_f32; n], vec![0.0_f32; n], vec![0.0_f32; n]];
    let [independent_z, independent_y, independent_x] = &mut independent;
    cc_forces_into(
        &moving,
        &fixed,
        &gradient.z,
        &gradient.y,
        &gradient.x,
        dims,
        1,
        independent_z,
        independent_y,
        independent_x,
    );
    assert_eq!(shared, independent);
}

#[test]
fn bidirectional_fusion_matches_independent_passes() {
    let dims = [6usize, 6, 6];
    let n = dims.iter().product();
    let fixed: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013) % 1.0).collect();
    let moving: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.021) + 0.17) % 1.0).collect();
    let fixed_gradient =
        crate::deformable_field_ops::compute_gradient(&fixed, dims.into(), [1.0; 3]);
    let moving_gradient =
        crate::deformable_field_ops::compute_gradient(&moving, dims.into(), [1.0; 3]);
    let sats = CcSats::build(&fixed, &moving, dims, 1);
    let mut fused_i = [vec![0.0_f32; n], vec![0.0_f32; n], vec![0.0_f32; n]];
    let mut fused_j = [vec![0.0_f32; n], vec![0.0_f32; n], vec![0.0_f32; n]];
    let [fused_iz, fused_iy, fused_ix] = &mut fused_i;
    let [fused_jz, fused_jy, fused_jx] = &mut fused_j;
    let mut slice_cc = vec![(0.0_f64, 0usize); dims[0]];
    let fused_cc = bidirectional_cc_from_sats_into(
        &fixed,
        &moving,
        crate::deformable_field_ops::VectorField {
            z: &fixed_gradient.z,
            y: &fixed_gradient.y,
            x: &fixed_gradient.x },
        crate::deformable_field_ops::VectorField {
            z: &moving_gradient.z,
            y: &moving_gradient.y,
            x: &moving_gradient.x },
        dims,
        &sats,
        crate::deformable_field_ops::VectorFieldMut {
            z: fused_iz,
            y: fused_iy,
            x: fused_ix },
        crate::deformable_field_ops::VectorFieldMut {
            z: fused_jz,
            y: fused_jy,
            x: fused_jx },
        &mut slice_cc,
    );

    let mut reference_i = [vec![0.0_f32; n], vec![0.0_f32; n], vec![0.0_f32; n]];
    let [reference_iz, reference_iy, reference_ix] = &mut reference_i;
    cc_forces_into(
        &fixed,
        &moving,
        &fixed_gradient.z,
        &fixed_gradient.y,
        &fixed_gradient.x,
        dims,
        1,
        reference_iz,
        reference_iy,
        reference_ix,
    );
    let mut reference_j = [vec![0.0_f32; n], vec![0.0_f32; n], vec![0.0_f32; n]];
    let [reference_jz, reference_jy, reference_jx] = &mut reference_j;
    cc_forces_into(
        &moving,
        &fixed,
        &moving_gradient.z,
        &moving_gradient.y,
        &moving_gradient.x,
        dims,
        1,
        reference_jz,
        reference_jy,
        reference_jx,
    );
    assert_eq!(fused_i, reference_i);
    assert_eq!(fused_j, reference_j);
    let reference_cc = mean_local_cc(&fixed, &moving, dims, 1);
    // At most n bounded CC terms are reordered. gamma_n < 5e-14 here;
    // 1e-12 includes the final division rounding while remaining diagnostic.
    assert!((fused_cc - reference_cc).abs() < 1e-12);
}
