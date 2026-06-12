//! Mutual information estimators via histogram binning.
//!
//! # Definitions
//!
//! I(X;Y) = H(X) + H(Y) − H(X,Y)   (Shannon 1948)
//!
//! NMI(X,Y) = (H(X) + H(Y)) / H(X,Y)   (Studholme et al. 1999)
//!
//! I_mattes(X;Y) = bilinear soft-binning MI   (Mattes et al. 2003)
//!
//! I(X;Y|Z) = H(X,Z) + H(Y,Z) − H(X,Y,Z) − H(Z)   (conditional MI)
//!
//! II(X;Y;Z) = I(X;Y) − I(X;Y|Z)   (interaction information, Mcgill 1954)
//!           = I(X;Y) + I(X;Z) − I(X;Y,Z) (equivalent formulation)

use anyhow::{bail, Result};

use super::entropy::{joint_entropy, joint_entropy_n, marginal_entropy, min_max};

/// Standard bivariate mutual information I(X;Y) = H(X) + H(Y) − H(X,Y).
///
/// Returns `max(I, 0.0)` — negative values are numerical artefacts from
/// finite-bin histograms where I(X;Y) ≈ 0.
///
/// # Errors
/// Propagates errors from [`joint_entropy`] and [`marginal_entropy`].
pub fn mutual_information(a: &[f32], b: &[f32], num_bins: usize) -> Result<f64> {
    if a.len() != b.len() {
        bail!("channel lengths differ: {} vs {}", a.len(), b.len());
    }
    if a.is_empty() {
        bail!("inputs must not be empty");
    }
    let h_a = marginal_entropy(a, num_bins)?;
    let h_b = marginal_entropy(b, num_bins)?;
    let h_ab = joint_entropy(a, b, num_bins)?;
    Ok((h_a + h_b - h_ab).max(0.0))
}

/// Normalized mutual information NMI(X,Y) = (H(X) + H(Y)) / H(X,Y).
///
/// Returns `1.0` when H(X,Y) < ε (both signals are identical constants).
/// NMI ∈ [1.0, 2.0]: value 1 means independent; value 2 means identical.
///
/// # Errors
/// Propagates errors from [`joint_entropy`] and [`marginal_entropy`].
pub fn normalized_mutual_information(a: &[f32], b: &[f32], num_bins: usize) -> Result<f64> {
    if a.len() != b.len() {
        bail!("channel lengths differ: {} vs {}", a.len(), b.len());
    }
    if a.is_empty() {
        bail!("inputs must not be empty");
    }
    let h_a = marginal_entropy(a, num_bins)?;
    let h_b = marginal_entropy(b, num_bins)?;
    let h_ab = joint_entropy(a, b, num_bins)?;
    if h_ab < f64::EPSILON {
        return Ok(1.0);
    }
    Ok((h_a + h_b) / h_ab)
}

/// Symmetric uncertainty U(X;Y) = 2·I(X;Y) / (H(X)+H(Y)).
///
/// Returns U ∈ [0.0, 1.0]: 0.0 for independent signals, 1.0 for identical.
/// Returns 0.0 when H(X)+H(Y) < ε (both channels constant).
///
/// This is the symmetric-uncertainty coefficient of Liu & Setiono (1996),
/// equivalent to the "normalized MI" variant used by SimpleITK's
/// `NormalizedMutualInformationFilter`.
///
/// # Errors
/// Propagates errors from [`mutual_information`] and [`marginal_entropy`].
pub fn symmetric_uncertainty(a: &[f32], b: &[f32], num_bins: usize) -> Result<f64> {
    if a.len() != b.len() {
        bail!("channel lengths differ: {} vs {}", a.len(), b.len());
    }
    if a.is_empty() {
        bail!("inputs must not be empty");
    }
    let mi = mutual_information(a, b, num_bins)?;
    let h_a = marginal_entropy(a, num_bins)?;
    let h_b = marginal_entropy(b, num_bins)?;
    let denom = h_a + h_b;
    if denom < 1e-15 {
        return Ok(0.0);
    }
    Ok((2.0 * mi / denom).clamp(0.0, 1.0))
}

/// Bilinear soft-assignment mutual information (Mattes et al. 2003).
///
/// Each sample distributes probability weight to the 4 neighboring histogram
/// cells via bilinear interpolation instead of hard nearest-bin assignment.
/// Soft-binning produces smoother gradients for image registration at the
/// cost of ~4× more histogram updates per sample.
///
/// Returns `max(I, 0.0)` — negative values are numerical artefacts.
///
/// # Errors
/// Returns an error when lengths differ, `a` is empty, or `num_bins < 2`.
pub fn mutual_information_mattes(a: &[f32], b: &[f32], num_bins: usize) -> Result<f64> {
    if a.len() != b.len() {
        bail!("channel lengths differ: {} vs {}", a.len(), b.len());
    }
    if a.is_empty() {
        bail!("inputs must not be empty");
    }
    if num_bins < 2 {
        bail!("num_bins must be >= 2, got {}", num_bins);
    }
    let n = a.len();
    let (a_min, a_max) = min_max(a);
    let (b_min, b_max) = min_max(b);
    let a_range = if (a_max - a_min).abs() < f32::EPSILON {
        1.0_f64
    } else {
        (a_max - a_min) as f64
    };
    let b_range = if (b_max - b_min).abs() < f32::EPSILON {
        1.0_f64
    } else {
        (b_max - b_min) as f64
    };
    let scale_a = (num_bins - 1) as f64 / a_range;
    let scale_b = (num_bins - 1) as f64 / b_range;
    let total = n as f64;

    let mut joint = vec![0.0_f64; num_bins * num_bins];
    let mut hist_a = vec![0.0_f64; num_bins];
    let mut hist_b = vec![0.0_f64; num_bins];

    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let fa = ((ai as f64 - a_min as f64) * scale_a).clamp(0.0, (num_bins - 1) as f64);
        let fb = ((bi as f64 - b_min as f64) * scale_b).clamp(0.0, (num_bins - 1) as f64);
        let ia = fa.floor() as usize;
        let ib = fb.floor() as usize;
        let wa1 = fa - ia as f64;
        let wb1 = fb - ib as f64;
        let wa0 = 1.0 - wa1;
        let wb0 = 1.0 - wb1;
        let ia1 = (ia + 1).min(num_bins - 1);
        let ib1 = (ib + 1).min(num_bins - 1);
        joint[ia * num_bins + ib] += wa0 * wb0;
        joint[ia * num_bins + ib1] += wa0 * wb1;
        joint[ia1 * num_bins + ib] += wa1 * wb0;
        joint[ia1 * num_bins + ib1] += wa1 * wb1;
        hist_a[ia] += wa0;
        hist_a[ia1] += wa1;
        hist_b[ib] += wb0;
        hist_b[ib1] += wb1;
    }

    for v in joint.iter_mut() {
        *v /= total;
    }
    for v in hist_a.iter_mut() {
        *v /= total;
    }
    for v in hist_b.iter_mut() {
        *v /= total;
    }

    let h_a: f64 = hist_a
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum();
    let h_b: f64 = hist_b
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum();
    let h_ab: f64 = joint
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum();
    Ok((h_a + h_b - h_ab).max(0.0))
}

/// Conditional mutual information I(X;Y|Z) = H(X,Z) + H(Y,Z) − H(X,Y,Z) − H(Z).
///
/// Measures how much X and Y share that is NOT explained by Z.
/// I(X;Y|Z) ≥ 0 always (data processing inequality).
///
/// # Derivation
///
/// I(X;Y|Z) = H(X,Z) + H(Y,Z) − H(X,Y,Z) − H(Z)
///
/// # Errors
///
/// Returns an error when lengths differ, inputs are empty, `num_bins < 2`,
/// or the joint histogram size exceeds 4 194 304 entries.
pub fn conditional_mutual_information(
    x: &[f32],
    y: &[f32],
    z: &[f32],
    num_bins: usize,
) -> Result<f64> {
    if x.len() != y.len() || x.len() != z.len() {
        bail!(
            "channel lengths differ: x={} y={} z={}",
            x.len(),
            y.len(),
            z.len()
        );
    }
    if x.is_empty() {
        bail!("inputs must not be empty");
    }
    let h_xz = joint_entropy_n(&[x, z], num_bins)?;
    let h_yz = joint_entropy_n(&[y, z], num_bins)?;
    let h_xyz = joint_entropy_n(&[x, y, z], num_bins)?;
    let h_z = marginal_entropy(z, num_bins)?;
    Ok((h_xz + h_yz - h_xyz - h_z).max(0.0))
}

/// Interaction information (co-information) II(X;Y;Z) = I(X;Y) − I(X;Y|Z).
///
/// Measures the net effect of Z on the shared information between X and Y.
/// - II > 0: Z introduces synergy (knowing Z increases I(X;Y)).
/// - II < 0: Z carries redundant information (knowing Z reduces apparent I(X;Y)).
/// - II = 0: Z has no net effect on I(X;Y).
///
/// Unlike I(X;Y) ≥ 0, interaction information can be negative.
///
/// # Definition (McGill 1954)
///
/// II(X;Y;Z) = I(X;Y) − I(X;Y|Z)
///           = H(X) + H(Y) + H(Z) − H(X,Y) − H(X,Z) − H(Y,Z) + H(X,Y,Z)
///
/// # Errors
///
/// Returns an error when lengths differ, inputs are empty, `num_bins < 2`,
/// or the joint histogram size exceeds 4 194 304 entries.
pub fn interaction_information(x: &[f32], y: &[f32], z: &[f32], num_bins: usize) -> Result<f64> {
    if x.len() != y.len() || x.len() != z.len() {
        bail!(
            "channel lengths differ: x={} y={} z={}",
            x.len(),
            y.len(),
            z.len()
        );
    }
    if x.is_empty() {
        bail!("inputs must not be empty");
    }
    let mi_xy = mutual_information(x, y, num_bins)?;
    let cmi = conditional_mutual_information(x, y, z, num_bins)?;
    Ok(mi_xy - cmi)
}
