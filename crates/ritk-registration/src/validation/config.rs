//! Validation configuration bounding properties.

/// Whether tensor shape validation is performed before operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ShapeValidation {
    /// Validate tensor shapes (default).
    #[default]
    Enabled,
    /// Skip shape validation.
    Disabled,
}

/// Whether numerical stability (NaN/Inf) checks are performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NumericalCheck {
    /// Check for NaN/Inf (default).
    #[default]
    Enabled,
    /// Skip numerical checks.
    Disabled,
}

/// Validation configuration limits enforcing runtime bounds over coordinate operations.
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    pub max_gradient_norm: Option<f64>,
    pub min_value: Option<f32>,
    pub max_value: Option<f32>,
    pub nan_inf_tolerance: f32,
    pub shape_validation: ShapeValidation,
    pub numerical_check: NumericalCheck,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            max_gradient_norm: Some(1000.0),
            min_value: Some(-1e6),
            max_value: Some(1e6),
            nan_inf_tolerance: 1e-6,
            shape_validation: ShapeValidation::Enabled,
            numerical_check: NumericalCheck::Enabled,
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
        self.shape_validation = ShapeValidation::Disabled;
        self
    }

    pub fn without_numerical_checks(mut self) -> Self {
        self.numerical_check = NumericalCheck::Disabled;
        self
    }
}
