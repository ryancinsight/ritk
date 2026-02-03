//! Gradient clipping utilities for registration optimization.
//!
//! This module provides gradient clipping strategies to prevent exploding gradients
//! during registration optimization, improving stability and convergence.

/// Clip gradients by global L2 norm.
///
/// # Arguments
/// * `gradients` - The gradients to clip (as a slice of f64 values)
/// * `max_norm` - Maximum global L2 norm
///
/// # Returns
/// The clipped gradients as a Vec<f64>
pub fn clip_gradients_l2(gradients: &[f64], max_norm: f64) -> Vec<f64> {
    let total_norm_sq: f64 = gradients.iter().map(|x| x * x).sum();
    let total_norm = total_norm_sq.sqrt();
    
    if total_norm > max_norm && total_norm > 0.0 {
        let scale = max_norm / total_norm;
        gradients.iter().map(|x| x * scale).collect()
    } else {
        gradients.to_vec()
    }
}

/// Gradient clipping strategy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GradientClipping {
    /// No gradient clipping
    None,
    /// Clip gradients by L2 norm
    L2Norm(f64),
    /// Clip gradients by L1 norm
    L1Norm(f64),
    /// Clip gradients by maximum absolute value
    MaxValue(f64),
}

impl Default for GradientClipping {
    fn default() -> Self {
        Self::None
    }
}

impl GradientClipping {
    /// Create a new L2 norm gradient clipper.
    pub fn l2_norm(max_norm: f64) -> Self {
        Self::L2Norm(max_norm)
    }

    /// Create a new L1 norm gradient clipper.
    pub fn l1_norm(max_norm: f64) -> Self {
        Self::L1Norm(max_norm)
    }

    /// Create a new max value gradient clipper.
    pub fn max_value(max_value: f64) -> Self {
        Self::MaxValue(max_value)
    }

    /// Apply gradient clipping to a gradient vector.
    ///
    /// # Arguments
    ///
    /// * `gradient` - The gradient vector to clip
    ///
    /// # Returns
    ///
    /// The clipped gradient vector
    pub fn clip(&self, gradient: &[f64]) -> Vec<f64> {
        match self {
            Self::None => gradient.to_vec(),
            Self::L2Norm(max_norm) => self.clip_l2(gradient, *max_norm),
            Self::L1Norm(max_norm) => self.clip_l1(gradient, *max_norm),
            Self::MaxValue(max_value) => self.clip_max(gradient, *max_value),
        }
    }

    /// Clip gradients by L2 norm.
    fn clip_l2(&self, gradient: &[f64], max_norm: f64) -> Vec<f64> {
        let norm: f64 = gradient.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm <= max_norm || norm == 0.0 {
            gradient.to_vec()
        } else {
            let scale = max_norm / norm;
            gradient.iter().map(|x| x * scale).collect()
        }
    }

    /// Clip gradients by L1 norm.
    fn clip_l1(&self, gradient: &[f64], max_norm: f64) -> Vec<f64> {
        let norm: f64 = gradient.iter().map(|x| x.abs()).sum();
        if norm <= max_norm || norm == 0.0 {
            gradient.to_vec()
        } else {
            let scale = max_norm / norm;
            gradient.iter().map(|x| x * scale).collect()
        }
    }

    /// Clip gradients by maximum absolute value.
    fn clip_max(&self, gradient: &[f64], max_value: f64) -> Vec<f64> {
        gradient
            .iter()
            .map(|x| x.clamp(-max_value, max_value))
            .collect()
    }

    /// Check if gradient clipping is enabled.
    pub fn is_enabled(&self) -> bool {
        !matches!(self, Self::None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_norm_clipping() {
        let clipper = GradientClipping::l2_norm(1.0);
        let gradient = vec![2.0, 2.0, 2.0];
        let clipped = clipper.clip(&gradient);
        
        // L2 norm of [2, 2, 2] is sqrt(12) ≈ 3.464
        // After clipping to 1.0, scale is 1.0/3.464 ≈ 0.289
        assert!((clipped[0] - 0.577).abs() < 0.01);
    }

    #[test]
    fn test_l1_norm_clipping() {
        let clipper = GradientClipping::l1_norm(1.0);
        let gradient = vec![2.0, 2.0, 2.0];
        let clipped = clipper.clip(&gradient);
        
        // L1 norm of [2, 2, 2] is 6
        // After clipping to 1.0, scale is 1.0/6 ≈ 0.167
        assert!((clipped[0] - 0.333).abs() < 0.01);
    }

    #[test]
    fn test_max_value_clipping() {
        let clipper = GradientClipping::max_value(1.0);
        let gradient = vec![2.0, -2.0, 0.5];
        let clipped = clipper.clip(&gradient);
        
        assert_eq!(clipped[0], 1.0);
        assert_eq!(clipped[1], -1.0);
        assert_eq!(clipped[2], 0.5);
    }

    #[test]
    fn test_no_clipping() {
        let clipper = GradientClipping::None;
        let gradient = vec![2.0, 2.0, 2.0];
        let clipped = clipper.clip(&gradient);
        
        assert_eq!(clipped, gradient);
    }
}
