//! Mutual Information similarity metric for classical registration.

use ndarray::{Array2, Array3, Axis};

/// Mutual Information similarity metric using histogram density estimation.
///
/// Given joint histogram H(a,b) with N bins each:
///   p(a,b) = H(a,b) / sum(H)
///   p(a)   = sum_b p(a,b)
///   p(b)   = sum_a p(a,b)
///   MI(A,B) = sum_a sum_b p(a,b) * log(p(a,b) / (p(a) * p(b)))
///
/// Normalized MI: NMI = 2*MI / (H(A) + H(B))
#[derive(Debug, Clone)]
pub struct MutualInformationMetric {
    /// Number of histogram bins per intensity dimension.
    pub num_bins: usize,
    /// Minimum intensity value for binning.
    min_intensity: f64,
    /// Width of each histogram bin.
    bin_width: f64,
}

impl MutualInformationMetric {
    /// Create a new Mutual Information metric with explicit parameters.
    pub fn new(num_bins: usize, min_intensity: f64, max_intensity: f64) -> Self {
        let bin_width = (max_intensity - min_intensity) / num_bins as f64;
        Self {
            num_bins,
            min_intensity,
            bin_width,
        }
    }

    /// Compute joint histogram between two volumes.
    fn compute_joint_histogram(&self, fixed: &Array3<f64>, moving: &Array3<f64>) -> Array2<f64> {
        let mut joint_hist = Array2::<f64>::zeros((self.num_bins, self.num_bins));

        let step = std::cmp::max(1, fixed.len() / 10000);

        // Iterate using direct iteration over the arrays
        let fixed_iter = fixed.iter().copied();
        let moving_iter = moving.iter().copied();

        for (f_val, m_val) in fixed_iter.zip(moving_iter).step_by(step) {
            let f_bin = ((f_val - self.min_intensity) / self.bin_width).floor() as usize;
            let m_bin = ((m_val - self.min_intensity) / self.bin_width).floor() as usize;

            if f_bin < self.num_bins && m_bin < self.num_bins {
                joint_hist[[f_bin, m_bin]] += 1.0;
            }
        }

        joint_hist
    }

    /// Compute normalized mutual information: NMI = 2 * MI / (H(X) + H(Y))
    /// where MI = sum p(x,y) * log(p(x,y) / (p(x) * p(y)))
    pub fn compute(&self, fixed: &Array3<f64>, moving: &Array3<f64>) -> f64 {
        let joint_hist = self.compute_joint_histogram(fixed, moving);
        let total = joint_hist.sum();
        if total == 0.0 {
            return 0.0;
        }

        // Compute joint and marginal probabilities
        let p_joint = &joint_hist / total;
        let p_x = p_joint.sum_axis(Axis(1));
        let p_y = p_joint.sum_axis(Axis(0));

        // Compute entropies - pass slices for 1D arrays
        let h_x = self.compute_entropy(p_x.as_slice().unwrap_or(&[]));
        let h_y = self.compute_entropy(p_y.as_slice().unwrap_or(&[]));
        let h_xy = self.compute_joint_entropy(&p_joint);

        // Normalized MI
        if h_x + h_y == 0.0 {
            // Both images are constant (entropy = 0).
            // NMI = 1.0 iff both concentrate at the same histogram bin (identical values).
            let peak_x = p_x
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            let peak_y = p_y
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(usize::MAX);
            return if peak_x == peak_y { 1.0 } else { 0.0 };
        }

        let mi = h_x + h_y - h_xy;
        (2.0 * mi) / (h_x + h_y)
    }

    /// Compute entropy H(X) = -sum p(x) * log(p(x))
    fn compute_entropy(&self, p: &[f64]) -> f64 {
        let eps = 1e-10_f64;
        p.iter().filter(|&&x| x > eps).map(|&x| -x * x.ln()).sum()
    }

    /// Compute joint entropy H(X,Y) = -sum p(x,y) * log(p(x,y))
    fn compute_joint_entropy(&self, p_joint: &Array2<f64>) -> f64 {
        let eps = 1e-10_f64;
        p_joint
            .iter()
            .filter(|&&x| x > eps)
            .map(|&x| -x * x.ln())
            .sum()
    }
}

impl Default for MutualInformationMetric {
    fn default() -> Self {
        Self::new(32, 0.0, 255.0)
    }
}
