use burn::prelude::*;

/// Configuration for selective state space parameters
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StateSpaceParameters {
    /// State dimension (N in paper)
    pub state_dim: usize,
    /// Expansion factor for hidden dimension
    pub expand_factor: usize,
    /// Rank for low-rank parameterization of B and C
    pub dt_rank: usize,
    /// Minimum value for discretization step
    pub dt_min: f64,
    /// Maximum value for discretization step
    pub dt_max: f64,
}

impl Default for StateSpaceParameters {
    fn default() -> Self {
        Self {
            state_dim: 16,
            expand_factor: 2,
            dt_rank: 16,
            dt_min: 0.001,
            dt_max: 0.1,
        }
    }
}

/// Configuration for Selective State Space module
#[derive(Config, Debug, PartialEq)]
pub struct SelectiveStateSpaceConfig {
    /// Input channel dimension
    pub input_dim: usize,
    /// Output channel dimension (typically input_dim * expand_factor)
    pub output_dim: usize,
    /// State dimension for SSM
    #[config(default = "16")]
    pub state_dim: usize,
    /// Expansion factor for hidden dimension
    #[config(default = "2")]
    pub expand_factor: usize,
    /// Rank for low-rank parameterization
    #[config(default = "16")]
    pub dt_rank: usize,
    /// Dropout probability
    #[config(default = "0.0")]
    pub dropout: f64,
}

impl SelectiveStateSpaceConfig {
    /// Initialize with input/output dimensions
    pub fn new_with_dims(input_dim: usize, output_dim: usize) -> Self {
        Self {
            input_dim,
            output_dim,
            state_dim: 16,
            expand_factor: 2,
            dt_rank: 16,
            dropout: 0.0,
        }
    }
}
