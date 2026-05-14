//! Shannon entropy estimators via histogram binning.
//!
//! All functions use nearest-bin hard-assignment histograms.
//! Inputs are `f32` voxel intensities; outputs are `f64` entropy values.
//! Using `f64` for entropy is analytically required: `-p * p.ln()` accumulations
//! over large histograms lose precision in `f32`.

use anyhow::{bail, Result};

/// Return `(min, max)` of a non-empty `f32` slice.
///
/// Both values equal 0.0 when `data` is empty (caller must validate length).
#[inline]
pub(super) fn min_max(data: &[f32]) -> (f32, f32) {
    data.iter().fold(
        (f32::INFINITY, f32::NEG_INFINITY),
        |(mn, mx), &v| (mn.min(v), mx.max(v)),
    )
}

/// Bin `data` into `num_bins` uniform bins and return the normalized histogram.
fn build_marginal_hist(data: &[f32], num_bins: usize) -> Vec<f64> {
    let (mn, mx) = min_max(data);
    let range = (mx - mn) as f64;
    let scale = if range < f64::EPSILON {
        0.0
    } else {
        (num_bins - 1) as f64 / range
    };
    let total = data.len() as f64;
    let mut hist = vec![0.0_f64; num_bins];
    for &v in data {
        let bin = ((v as f64 - mn as f64) * scale)
            .clamp(0.0, (num_bins - 1) as f64) as usize;
        hist[bin] += 1.0;
    }
    for p in hist.iter_mut() {
        *p /= total;
    }
    hist
}

/// H(X) = -Σ p·ln(p) for non-zero `p`.
fn entropy_from_hist(hist: &[f64]) -> f64 {
    hist.iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum()
}

/// Marginal Shannon entropy H(X) = -Σ p·ln(p).
///
/// # Errors
/// Returns an error when `data` is empty or `num_bins < 2`.
pub fn marginal_entropy(data: &[f32], num_bins: usize) -> Result<f64> {
    if data.is_empty() {
        bail!("data must not be empty");
    }
    if num_bins < 2 {
        bail!("num_bins must be >= 2, got {}", num_bins);
    }
    let hist = build_marginal_hist(data, num_bins);
    Ok(entropy_from_hist(&hist))
}

/// Joint entropy H(X,Y) = -Σ p(x,y)·ln(p(x,y)).
///
/// # Errors
/// Returns an error when lengths differ, `a` is empty, or `num_bins < 2`.
pub fn joint_entropy(a: &[f32], b: &[f32], num_bins: usize) -> Result<f64> {
    if a.len() != b.len() {
        bail!("channel lengths differ: {} vs {}", a.len(), b.len());
    }
    if a.is_empty() {
        bail!("inputs must not be empty");
    }
    if num_bins < 2 {
        bail!("num_bins must be >= 2, got {}", num_bins);
    }
    let (a_min, a_max) = min_max(a);
    let (b_min, b_max) = min_max(b);
    let a_range = (a_max - a_min) as f64;
    let b_range = (b_max - b_min) as f64;
    let scale_a = if a_range < f64::EPSILON { 0.0 } else { (num_bins - 1) as f64 / a_range };
    let scale_b = if b_range < f64::EPSILON { 0.0 } else { (num_bins - 1) as f64 / b_range };
    let total = a.len() as f64;
    let mut joint = vec![0.0_f64; num_bins * num_bins];
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let ia = ((ai as f64 - a_min as f64) * scale_a).clamp(0.0, (num_bins - 1) as f64) as usize;
        let ib = ((bi as f64 - b_min as f64) * scale_b).clamp(0.0, (num_bins - 1) as f64) as usize;
        joint[ia * num_bins + ib] += 1.0;
    }
    for p in joint.iter_mut() {
        *p /= total;
    }
    Ok(entropy_from_hist(&joint))
}

/// Returns `(histogram, total_samples)` where `histogram` is the unnormalized counts
/// or probabilities. Here we return probabilities to match existing logic.
pub fn build_joint_hist_n(channels: &[&[f32]], num_bins: usize) -> Result<Vec<f64>> {
    let n = channels.len();
    if n == 0 {
        bail!("channels must not be empty");
    }
    if num_bins < 2 {
        bail!("num_bins must be >= 2, got {}", num_bins);
    }
    let len = channels[0].len();
    if len == 0 {
        bail!("channels must contain at least one sample");
    }
    for (i, ch) in channels.iter().enumerate() {
        if ch.len() != len {
            bail!("channel {} length {} != channel 0 length {}", i, ch.len(), len);
        }
    }
    let joint_size = num_bins.saturating_pow(n as u32);
    if joint_size > 4_194_304 {
        bail!(
            "joint histogram {}^{} = {} exceeds 4_194_304 limit; reduce num_bins or n",
            num_bins, n, joint_size
        );
    }
    let ranges: Vec<(f32, f32)> = channels.iter().map(|ch| min_max(ch)).collect();
    let mut joint = vec![0.0_f64; joint_size];
    let total = len as f64;
    for sample_idx in 0..len {
        let mut idx = 0usize;
        for (ch_idx, ch) in channels.iter().enumerate() {
            let (mn, mx) = ranges[ch_idx];
            let range = (mx - mn) as f64;
            let scale = if range < f64::EPSILON { 0.0 } else { (num_bins - 1) as f64 / range };
            let bin = ((ch[sample_idx] as f64 - mn as f64) * scale)
                .clamp(0.0, (num_bins - 1) as f64) as usize;
            idx = idx * num_bins + bin;
        }
        joint[idx] += 1.0;
    }
    for p in joint.iter_mut() {
        *p /= total;
    }
    Ok(joint)
}

/// N-way joint entropy H(X₁,...,Xₙ) = -Σ p·ln(p) over the N-dimensional histogram.
///
/// Joint histogram size is `num_bins^n`; enforced ≤ 4_194_304 entries.
///
/// # Errors
/// Returns an error on empty channels, length mismatch, `num_bins < 2`,
/// or when `num_bins^n > 4_194_304`.
pub fn joint_entropy_n(channels: &[&[f32]], num_bins: usize) -> Result<f64> {
    let joint = build_joint_hist_n(channels, num_bins)?;
    Ok(entropy_from_hist(&joint))
}

/// Marginalize an N-dimensional histogram by summing over a specified axis.
/// The resulting histogram has N-1 dimensions.
pub fn marginalize_hist(hist: &[f64], num_bins: usize, current_n: usize, axis_to_drop: usize) -> Vec<f64> {
    let out_size = num_bins.pow((current_n - 1) as u32);
    let mut out = vec![0.0_f64; out_size];
    
    for (idx, &p) in hist.iter().enumerate() {
        if p == 0.0 { continue; }
        
        let mut out_idx = 0;
        let mut out_multiplier = 1;
        let mut temp_idx = idx;
        
        for d in (0..current_n).rev() {
            let coord = temp_idx % num_bins;
            temp_idx /= num_bins;
            
            if d != axis_to_drop {
                out_idx += coord * out_multiplier;
                out_multiplier *= num_bins;
            }
        }
        out[out_idx] += p;
    }
    out
}

/// Calculate entropy directly from an already computed histogram
pub fn entropy_from_hist_pub(hist: &[f64]) -> f64 {
    entropy_from_hist(hist)
}
