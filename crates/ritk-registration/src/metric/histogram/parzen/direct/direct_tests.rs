use super::types::{BinRange, ParzenConfig, StackWeights};
use super::*;
use burn::tensor::Tensor;
use burn_ndarray::NdArray;

type B = NdArray<f32>;

fn device() -> <B as burn::tensor::backend::Backend>::Device {
    Default::default()
}

#[test]
fn accumulate_sample_direct_vs_sparse_weights() {
    // ARCH-317-01: Verify that the monomorphized direct-path accumulate
    // (using SampleWindow with pre-computed StackWeights) produces the same
    // histogram entries as the sparse-cache path (using SparseWFixedEntry).
    let num_bins = 32;
    let sigma_sq = 1.0_f32;
    let fix_cfg = ParzenConfig::new(sigma_sq);
    let mov_cfg = ParzenConfig::new(sigma_sq);

    let f_val = 15.3_f32;
    let m_val = 20.7_f32;
    let f_primary = f_val.floor() as i32;
    let m_primary = m_val.floor() as i32;
    let f_range = BinRange::new(f_primary, fix_cfg.half_width, num_bins);
    let m_range = BinRange::new(m_primary, mov_cfg.half_width, num_bins);

    // Direct-path: build SampleWindow with pre-computed StackWeights
    let f_weights = StackWeights::new(f_val, f_range.lo, f_range.hi, fix_cfg.inv_2sigma_sq);
    let m_weights = StackWeights::new(m_val, m_range.lo, m_range.hi, mov_cfg.inv_2sigma_sq);
    let window = SampleWindow {
        f_range,
        m_range,
        f_val,
        m_val,
        f_weights,
        m_weights,
    };
    let mut hist_direct = vec![0.0f32; num_bins * num_bins];
    accumulate_sample_direct(&mut hist_direct, num_bins, &window);

    // Sparse-path: build SparseWFixedEntry and use accumulate_sample_sparse
    let sparse_weights: Vec<SparseWFixedEntry> = f_range
        .iter()
        .map(|a| {
            let diff_f = f_val - a as f32;
            let w_f = (diff_f * diff_f * fix_cfg.inv_2sigma_sq).exp();
            SparseWFixedEntry::new(a, w_f)
        })
        .collect();
    let mut hist_sparse = vec![0.0f32; num_bins * num_bins];
    accumulate_sample_sparse(
        &mut hist_sparse,
        num_bins,
        m_range,
        &m_weights,
        sparse_weights.iter().copied(),
    );

    // Both must produce identical histograms
    for (i, (d, s)) in hist_direct.iter().zip(hist_sparse.iter()).enumerate() {
        let diff = (d - s).abs();
        assert!(
            diff < 1e-10,
            "accumulate_sample mismatch at bin {i}: direct={d}, sparse={s}, diff={diff}"
        );
    }
}

#[test]
fn direct_matches_dense_histogram() {
    let dev = device();
    let num_bins = 32;
    let sigma_in_bins = 1.0_f32;
    let sigma_sq = sigma_in_bins * sigma_in_bins;

    let n = 100;
    let fixed_vec: Vec<f32> = (0..n).map(|i| i as f32 % 30.0).collect();
    let moving_vec: Vec<f32> = (0..n).map(|i| (i * 3 + 7) as f32 % 30.0).collect();

    let direct_data =
        compute_joint_histogram_direct(&fixed_vec, &moving_vec, num_bins, sigma_sq, sigma_sq, None, None);
    let direct_slice = direct_data.as_slice::<f32>().unwrap();

    let fixed_tensor =
        Tensor::<B, 1>::from_data(TensorData::new(fixed_vec.clone(), Shape::new([n])), &dev);
    let moving_tensor =
        Tensor::<B, 1>::from_data(TensorData::new(moving_vec.clone(), Shape::new([n])), &dev);

    let bins_exp =
        burn::tensor::Tensor::<B, 1, burn::tensor::Int>::arange(0..num_bins as i64, &dev)
            .float()
            .reshape([1, num_bins]);
    let f_exp = fixed_tensor.clone().reshape([n, 1]);
    let m_exp = moving_tensor.clone().reshape([n, 1]);
    let diff_f = f_exp - bins_exp.clone();
    let sq_f = diff_f.clone() * diff_f;
    let w_fixed = (sq_f * (-0.5 / sigma_sq)).exp();
    let diff_m = m_exp - bins_exp;
    let sq_m = diff_m.clone() * diff_m;
    let w_moving = (sq_m * (-0.5 / sigma_sq)).exp();
    let dense_hist = w_fixed.transpose().matmul(w_moving);

    let dense_data = dense_hist.into_data();
    let dense_slice = dense_data.as_slice::<f32>().unwrap();

    for (i, (d, s)) in dense_slice.iter().zip(direct_slice.iter()).enumerate() {
        let diff = (d - s).abs();
        let max_val = d.abs().max(s.abs()).max(1e-10);
        let rel_err = diff / max_val;
        assert!(
            rel_err < 0.01 || diff < 0.01,
            "mismatch at bin {i}: dense={d}, direct={s}, diff={diff}, rel_err={rel_err}"
        );
    }
}

#[test]
fn direct_with_oob_mask() {
    let num_bins = 16;
    let sigma_sq = 1.0_f32;
    let n = 10;
    let fixed_vec: Vec<f32> = (0..n).map(|i| i as f32 * 1.5).collect();
    let moving_vec: Vec<f32> = (0..n).map(|i| i as f32 * 1.2).collect();
    let all_oob = vec![0.0f32; n];

    let hist = compute_joint_histogram_direct(
        &fixed_vec,
        &moving_vec,
        num_bins,
        sigma_sq,
        sigma_sq,
            Some(&all_oob),
            None,
    );
    let sum: f32 = hist.as_slice::<f32>().unwrap().iter().sum();
    assert!(sum < 1e-6, "all-OOB histogram must be zero, got sum={sum}");
}

#[test]
fn sparse_from_cache_matches_direct() {
    let num_bins = 32;
    let sigma_in_bins = 1.0_f32;
    let sigma_sq = sigma_in_bins * sigma_in_bins;
    let n = 100;
    let fixed_vec: Vec<f32> = (0..n).map(|i| i as f32 % 30.0).collect();
    let moving_vec: Vec<f32> = (0..n).map(|i| (i * 3 + 7) as f32 % 30.0).collect();

    let direct_data =
        compute_joint_histogram_direct(&fixed_vec, &moving_vec, num_bins, sigma_sq, sigma_sq, None, None);
    let direct_slice = direct_data.as_slice::<f32>().unwrap();

    let sparse_w_fixed = build_sparse_w_fixed_transposed(&fixed_vec, num_bins, sigma_sq, None);
    let sparse_data = compute_joint_histogram_from_cache_sparse(
        &sparse_w_fixed,
        &moving_vec,
        num_bins,
        sigma_sq,
        None,
        None,
    );
    let sparse_slice = sparse_data.as_slice::<f32>().unwrap();

    for (i, (d, s)) in direct_slice.iter().zip(sparse_slice.iter()).enumerate() {
        let diff = (d - s).abs();
        let max_val = d.abs().max(s.abs()).max(1e-10);
        let rel_err = diff / max_val;
        assert!(
            rel_err < 0.01 || diff < 0.01,
            "mismatch at bin {i}: direct={d}, sparse={s}, diff={diff}, rel_err={rel_err}"
        );
    }
}

#[test]
fn sparse_from_cache_with_oob_mask() {
    let num_bins = 16;
    let sigma_sq = 1.0_f32;
    let n = 10;
    let fixed_vec: Vec<f32> = (0..n).map(|i| i as f32 * 1.5).collect();
    let moving_vec: Vec<f32> = (0..n).map(|i| i as f32 * 1.2).collect();
    let all_oob = vec![0.0f32; n];

    let sparse_w_fixed =
        build_sparse_w_fixed_transposed(&fixed_vec, num_bins, sigma_sq, Some(&all_oob));

    for (i, entry) in sparse_w_fixed.iter().enumerate() {
        assert!(
            entry.is_empty(),
            "OOB sample {i} should have empty sparse entry, got {} elements",
            entry.len()
        );
    }

    let hist = compute_joint_histogram_from_cache_sparse(
        &sparse_w_fixed,
        &moving_vec,
        num_bins,
        sigma_sq,
        Some(&all_oob),
        None,
    );
    let sum: f32 = hist.as_slice::<f32>().unwrap().iter().sum();
    assert!(
        sum < 1e-6,
        "all-OOB sparse histogram must be zero, got sum={sum}"
    );
}

#[test]
fn direct_large_volume_matches_dense() {
    let dev = device();
    let num_bins = 32;
    let sigma_in_bins = 1.0_f32;
    let sigma_sq = sigma_in_bins * sigma_in_bins;
    let n = 1000;
    let fixed_vec: Vec<f32> = (0..n).map(|i| (i as f32 * 0.03) % 30.0).collect();
    let moving_vec: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) as f32 * 0.02) % 30.0).collect();

    let direct_data =
        compute_joint_histogram_direct(&fixed_vec, &moving_vec, num_bins, sigma_sq, sigma_sq, None, None);
    let direct_slice = direct_data.as_slice::<f32>().unwrap();

    let fixed_tensor =
        Tensor::<B, 1>::from_data(TensorData::new(fixed_vec.clone(), Shape::new([n])), &dev);
    let moving_tensor =
        Tensor::<B, 1>::from_data(TensorData::new(moving_vec.clone(), Shape::new([n])), &dev);

    let bins_exp =
        burn::tensor::Tensor::<B, 1, burn::tensor::Int>::arange(0..num_bins as i64, &dev)
            .float()
            .reshape([1, num_bins]);
    let f_exp = fixed_tensor.clone().reshape([n, 1]);
    let m_exp = moving_tensor.clone().reshape([n, 1]);
    let diff_f = f_exp - bins_exp.clone();
    let sq_f = diff_f.clone() * diff_f;
    let w_fixed = (sq_f * (-0.5 / sigma_sq)).exp();
    let diff_m = m_exp - bins_exp;
    let sq_m = diff_m.clone() * diff_m;
    let w_moving = (sq_m * (-0.5 / sigma_sq)).exp();
    let dense_hist = w_fixed.transpose().matmul(w_moving);

    let dense_data = dense_hist.into_data();
    let dense_slice = dense_data.as_slice::<f32>().unwrap();

    for (i, (d, s)) in dense_slice.iter().zip(direct_slice.iter()).enumerate() {
        let diff = (d - s).abs();
        let max_val = d.abs().max(s.abs()).max(1e-10);
        let rel_err = diff / max_val;
        assert!(
            rel_err < 0.25 || diff < 0.5,
            "large-volume mismatch at bin {i}: dense={d}, direct={s}, diff={diff}, rel_err={rel_err}"
        );
    }
}

#[test]
fn sparse_cache_large_volume_matches_direct() {
    let num_bins = 32;
    let sigma_in_bins = 1.0_f32;
    let sigma_sq = sigma_in_bins * sigma_in_bins;
    let n = 500;
    let fixed_vec: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05) % 30.0).collect();
    let moving_vec: Vec<f32> = (0..n)
        .map(|i| ((i * 11 + 5) as f32 * 0.03) % 30.0)
        .collect();

    let direct_data =
        compute_joint_histogram_direct(&fixed_vec, &moving_vec, num_bins, sigma_sq, sigma_sq, None, None);
    let direct_slice = direct_data.as_slice::<f32>().unwrap();

    let sparse_w_fixed = build_sparse_w_fixed_transposed(&fixed_vec, num_bins, sigma_sq, None);
    let sparse_data = compute_joint_histogram_from_cache_sparse(
        &sparse_w_fixed,
        &moving_vec,
        num_bins,
        sigma_sq,
        None,
        None,
    );
    let sparse_slice = sparse_data.as_slice::<f32>().unwrap();

    for (i, (d, s)) in direct_slice.iter().zip(sparse_slice.iter()).enumerate() {
        let diff = (d - s).abs();
        let max_val = d.abs().max(s.abs()).max(1e-10);
        let rel_err = diff / max_val;
        assert!(
            rel_err < 0.01 || diff < 0.01,
            "large-volume sparse mismatch at bin {i}: direct={d}, sparse={s}, diff={diff}, rel_err={rel_err}"
        );
    }
}

#[test]
fn direct_oob_partial_mask() {
    let num_bins = 16;
    let sigma_sq = 1.0_f32;
    let n = 20;
    let fixed_vec: Vec<f32> = (0..n).map(|i| i as f32 * 1.5).collect();
    let moving_vec: Vec<f32> = (0..n).map(|i| i as f32 * 1.2).collect();

    let partial_mask: Vec<f32> = (0..n).map(|i| if i < n / 2 { 1.0 } else { 0.0 }).collect();

    let hist_partial = compute_joint_histogram_direct(
        &fixed_vec,
        &moving_vec,
        num_bins,
        sigma_sq,
        sigma_sq,
        Some(&partial_mask),
        None,
    );

    let half_n = n / 2;
    let hist_half = compute_joint_histogram_direct(
        &fixed_vec[..half_n],
        &moving_vec[..half_n],
        num_bins,
        sigma_sq,
        sigma_sq,
        None,
        None,
    );

    let partial_slice = hist_partial.as_slice::<f32>().unwrap();
    let half_slice = hist_half.as_slice::<f32>().unwrap();

    for (i, (p, h)) in partial_slice.iter().zip(half_slice.iter()).enumerate() {
        let diff = (p - h).abs();
        assert!(
            diff < 1e-6,
            "partial OOB mask mismatch at bin {i}: partial={p}, half={h}, diff={diff}"
        );
    }

    let hist_full =
        compute_joint_histogram_direct(&fixed_vec, &moving_vec, num_bins, sigma_sq, sigma_sq, None, None);
    let sum_partial: f32 = partial_slice.iter().sum();
    let sum_full: f32 = hist_full.as_slice::<f32>().unwrap().iter().sum();
    assert!(
        sum_partial < sum_full,
        "partial-mask sum ({sum_partial}) must be less than full sum ({sum_full})"
    );
    assert!(sum_partial > 0.0, "partial-mask sum must be > 0");
}
