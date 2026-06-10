//! Geometric utility functions for DICOM slice orientation and spacing analysis.

/// Compute the cross product of two 3-vectors.
#[inline]
pub(super) fn cross_3d(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Relative deviation threshold above which adjacent-pair spacing is non-uniform (1%).
pub(super) const NONUNIFORM_SPACING_THRESHOLD: f64 = 0.01;

/// Gap multiple above which an adjacent pair indicates missing slices (1.5×).
pub(super) const MISSING_SLICE_GAP_FACTOR: f64 = 1.5;

/// Normalize a 3-vector; returns `None` when the vector length is < 1e-10.
#[inline]
pub(in crate::format::dicom) fn normalize_3d(v: [f64; 3]) -> Option<[f64; 3]> {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-10 {
        None
    } else {
        Some([v[0] / len, v[1] / len, v[2] / len])
    }
}

/// Dot product of two 3-vectors.
#[inline]
pub(in crate::format::dicom) fn dot_3d(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Compute the normalized slice normal from ImageOrientationPatient.
///
/// Given IOP = [rx, ry, rz, cx, cy, cz], the slice normal is:
///   N̂ = normalize(cross([rx, ry, rz], [cx, cy, cz]))
///
/// Returns `None` for degenerate IOP.
pub(in crate::format::dicom) fn slice_normal_from_iop(iop: [f64; 6]) -> Option<[f64; 3]> {
    let r = [iop[0], iop[1], iop[2]];
    let c = [iop[3], iop[4], iop[5]];
    normalize_3d(cross_3d(r, c))
}

/// Result of analyzing per-slice spacing uniformity.
///
/// Derived from sorted projected positions `p[0] ≤ p[1] ≤ … ≤ p[N-1]`:
/// - gaps[i] = p[i+1] - p[i]
/// - nominal_spacing = median(gaps)
/// - max_relative_deviation = max_i |gaps[i] - nominal| / nominal
/// - missing_between: indices i where gaps[i] > 1.5 × nominal
#[derive(Debug, Clone)]
pub(in crate::format::dicom) struct SliceGeometryReport {
    pub nominal_spacing: f64,
    pub max_relative_deviation: f64,
    pub missing_between: Vec<usize>,
    pub is_nonuniform: bool,
    pub has_missing_slices: bool,
}

/// Analyze per-slice projected positions for spacing uniformity.
///
/// # Precondition
/// `positions` is sorted ascending; `positions.len() >= 2`.
pub(in crate::format::dicom) fn analyze_slice_spacing(positions: &[f64]) -> SliceGeometryReport {
    debug_assert!(positions.len() >= 2, "need >= 2 positions");
    let n = positions.len();
    let gaps: Vec<f64> = (0..n - 1)
        .map(|i| positions[i + 1] - positions[i])
        .collect();

    let mut sorted_gaps = gaps.clone();
    sorted_gaps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let nominal = if sorted_gaps.len().is_multiple_of(2) {
        let mid = sorted_gaps.len() / 2;
        (sorted_gaps[mid - 1] + sorted_gaps[mid]) / 2.0
    } else {
        sorted_gaps[sorted_gaps.len() / 2]
    };

    if nominal <= 0.0 || !nominal.is_finite() {
        return SliceGeometryReport {
            nominal_spacing: 1.0,
            max_relative_deviation: 0.0,
            missing_between: Vec::new(),
            is_nonuniform: false,
            has_missing_slices: false,
        };
    }

    let mut max_rel_dev = 0.0_f64;
    let mut missing_between = Vec::new();
    for (i, &g) in gaps.iter().enumerate() {
        let rel_dev = (g - nominal).abs() / nominal;
        if rel_dev > max_rel_dev {
            max_rel_dev = rel_dev;
        }
        if g > MISSING_SLICE_GAP_FACTOR * nominal {
            missing_between.push(i);
        }
    }

    let has_missing_slices = !missing_between.is_empty();
    let is_nonuniform = max_rel_dev > NONUNIFORM_SPACING_THRESHOLD;
    SliceGeometryReport {
        nominal_spacing: nominal,
        max_relative_deviation: max_rel_dev,
        missing_between,
        is_nonuniform,
        has_missing_slices,
    }
}

/// Resample decoded frames from nonuniform source positions to a uniform grid.
///
/// # Mathematical specification
///
/// Given sorted source positions p[0..N] and decoded frames src[0..N]:
/// - `N_target = round((p[N-1] - p[0]) / target_spacing) + 1`
/// - `target[k] = p[0] + k × target_spacing`, k ∈ [0, N_target)
///
/// For each target frame k, locate bracketing source pair (lo, hi) and interpolate:
///   `output[k][j] = (1 - t) × src[lo][j] + t × src[hi][j]`
///
/// Edge cases: clamp to first/last frame; degenerate gap uses frame lo.
pub(in crate::format::dicom) fn resample_frames_linear(
    decoded_frames: &[Vec<f32>],
    src_positions: &[f64],
    target_spacing: f64,
) -> Vec<Vec<f32>> {
    debug_assert_eq!(decoded_frames.len(), src_positions.len());
    if decoded_frames.is_empty() || target_spacing <= 0.0 {
        return decoded_frames.to_vec();
    }
    let first = src_positions[0];
    let last = *src_positions
        .last()
        .expect("source positions must not be empty");
    let span = last - first;
    if span <= 0.0 || !span.is_finite() {
        return decoded_frames.to_vec();
    }
    let n_target = (span / target_spacing).round() as usize + 1;
    let mut output = Vec::with_capacity(n_target);

    for k in 0..n_target {
        let target_pos = first + k as f64 * target_spacing;
        let idx = src_positions.partition_point(|&p| p <= target_pos);
        let frame = if idx == 0 {
            decoded_frames[0].clone()
        } else if idx >= src_positions.len() {
            decoded_frames[src_positions.len() - 1].clone()
        } else {
            let lo = idx - 1;
            let hi = idx;
            let gap = src_positions[hi] - src_positions[lo];
            if gap < 1e-10 {
                decoded_frames[lo].clone()
            } else {
                let t = ((target_pos - src_positions[lo]) / gap) as f32;
                let one_minus_t = 1.0_f32 - t;
                decoded_frames[lo]
                    .iter()
                    .zip(decoded_frames[hi].iter())
                    .map(|(&a, &b)| one_minus_t * a + t * b)
                    .collect()
            }
        };
        output.push(frame);
    }
    output
}
