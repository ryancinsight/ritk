//! Validation configuration bounding properties.

/// Validation configuration limits enforcing runtime bounds over coordinate operations.
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    pub max_gradient_norm: Option<f64>,
    pub min_value: Option<f32>,
    pub max_value: Option<f32>,
    pub nan_inf_tolerance: f32,
    pub validate_shapes: bool,
    pub check_numerical_stability: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            max_gradient_norm: Some(1000.0),
            min_value: Some(-1e6),
            max_value: Some(1e6),
            nan_inf_tolerance: 1e-6,
            validate_shapes: true,
            check_numerical_stability: true,
        }
    }
}

impl ValidationConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_max_gradient_norm(mut self, norm: f64) -> Self {
        self.max_gradient_norm = Some(norm);
        self
    }

    pub fn without_gradient_clipping(mut self) -> Self {
        self.max_gradient_norm = None;
        self
    }

    pub fn with_value_bounds(mut self, min: f32, max: f32) -> Self {
        self.min_value = Some(min);
        self.max_value = Some(max);
        self
    }

    pub fn without_shape_validation(mut self) -> Self {
        self.validate_shapes = false;
        self
    }

    pub fn without_numerical_checks(mut self) -> Self {
        self.check_numerical_stability = false;
        self
    }
}
