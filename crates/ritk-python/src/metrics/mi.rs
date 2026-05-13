//! Histogram-based Mutual Information slice-level implementation.
//!
//! # Formula
//! MI = H(A) + H(B) − H(A,B)
//!
//! Three variants:
//! - "mattes":     bilinear soft-binning (Mattes 2003, eq. 4)
//! - "standard":   nearest-bin hard assignment
//! - "normalized": 2·MI / (H(A)+H(B))  (Studholme 1999)

use anyhow::{bail, Result};

/// Returns (min, max) of a non-empty f32 slice.
pub(super) fn min_max(data: &[f32]) -> (f32, f32) {
    debug_assert!(!data.is_empty());
    data.iter()
        .fold((data[0], data[0]), |(mn, mx), &v| (mn.min(v), mx.max(v)))
}

/// Histogram-based MI with configurable binning strategy.
///
/// `variant` must be one of `"mattes"`, `"standard"`, `"normalized"`.
pub(super) fn mi_slices(a: &[f32], b: &[f32], num_bins: usize, variant: &str) -> Result<f64> {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    if n == 0 {
        bail!("cannot compute MI of empty images");
    }
    let bins = num_bins;

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

    let mut joint = vec![0.0_f64; bins * bins];
    let mut hist_a = vec![0.0_f64; bins];
    let mut hist_b = vec![0.0_f64; bins];

    match variant {
        "mattes" => {
            let scale_a = (bins - 1) as f64 / a_range;
            let scale_b = (bins - 1) as f64 / b_range;
            for (&ai, &bi) in a.iter().zip(b.iter()) {
                let fa =
                    ((ai as f64 - a_min as f64) * scale_a).clamp(0.0, (bins - 1) as f64);
                let fb =
                    ((bi as f64 - b_min as f64) * scale_b).clamp(0.0, (bins - 1) as f64);
                let ia = fa.floor() as usize;
                let ib = fb.floor() as usize;
                let wa1 = fa - ia as f64;
                let wb1 = fb - ib as f64;
                let wa0 = 1.0 - wa1;
                let wb0 = 1.0 - wb1;
                let ia1 = (ia + 1).min(bins - 1);
                let ib1 = (ib + 1).min(bins - 1);
                joint[ia * bins + ib] += wa0 * wb0;
                joint[ia * bins + ib1] += wa0 * wb1;
                joint[ia1 * bins + ib] += wa1 * wb0;
                joint[ia1 * bins + ib1] += wa1 * wb1;
                hist_a[ia] += wa0;
                hist_a[ia1] += wa1;
                hist_b[ib] += wb0;
                hist_b[ib1] += wb1;
            }
        }
        "standard" | "normalized" => {
            let scale_a = (bins - 1) as f64 / a_range;
            let scale_b = (bins - 1) as f64 / b_range;
            for (&ai, &bi) in a.iter().zip(b.iter()) {
                let ia =
                    (((ai as f64 - a_min as f64) * scale_a) as usize).min(bins - 1);
                let ib =
                    (((bi as f64 - b_min as f64) * scale_b) as usize).min(bins - 1);
                joint[ia * bins + ib] += 1.0;
                hist_a[ia] += 1.0;
                hist_b[ib] += 1.0;
            }
        }
        _ => unreachable!("variant validated before mi_slices"),
    }

    let total = n as f64;
    for v in joint.iter_mut() {
        *v /= total;
    }
    for v in hist_a.iter_mut() {
        *v /= total;
    }
    for v in hist_b.iter_mut() {
        *v /= total;
    }

    let h_a: f64 = hist_a.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum();
    let h_b: f64 = hist_b.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum();
    let h_ab: f64 = joint.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum();
    let mi = h_a + h_b - h_ab;

    if variant == "normalized" {
        let denom = h_a + h_b;
        if denom < 1e-15 {
            return Ok(0.0);
        }
        Ok(2.0 * mi / denom)
    } else {
        Ok(mi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mi_self_exceeds_constant() {
        // Analytical: MI(A,A) = H(A) > 0 for non-constant A.
        // MI(A, constant) = 0 since H(constant) = 0.
        let a: Vec<f32> = (0..32).map(|x| x as f32).collect();
        let b_const: Vec<f32> = vec![5.0_f32; 32];
        let mi_self = mi_slices(&a, &a, 16, "standard").unwrap();
        let mi_const = mi_slices(&a, &b_const, 16, "standard").unwrap();
        assert!(mi_self > 0.0, "MI(A,A) must be positive for non-constant A, got {mi_self}");
        assert!(
            mi_const.abs() < 1e-10,
            "MI(A,constant) must be 0, got {mi_const}"
        );
    }

    #[test]
    fn mi_normalized_variant_bounded() {
        let a: Vec<f32> = (0..64).map(|x| (x % 16) as f32).collect();
        let b: Vec<f32> = (0..64).map(|x| ((x + 4) % 16) as f32).collect();
        let nmi = mi_slices(&a, &b, 16, "normalized").unwrap();
        assert!(
            (0.0..=1.0).contains(&nmi),
            "normalized MI must be in [0,1], got {nmi}"
        );
    }

    #[test]
    fn min_max_single_element() {
        assert_eq!(min_max(&[42.0]), (42.0, 42.0));
    }
}
