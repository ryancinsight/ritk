//! Selective State Space (S6) Module
//!
//! Implements the core State Space Model from Mamba (Gu & Dao, 2023) adapted for
//! 3D volumetric medical image registration. The key innovation is input-dependent
//! parameters (Δ, B, C) that enable content-based selective propagation of information.
//!
//! # Architecture
//!
//! The continuous state space model is:
//!   h'(t) = Ah(t) + Bx(t)
//!   y(t)  = Ch(t)
//!
//! Discretized with input-dependent step size Δ:
//!   h_k = Āh_{k-1} + B̄x_k
//!   y_k = Ch_k
//!
//! where Ā = exp(ΔA) and B̄ = (ΔA)^{-1}(exp(ΔA) - I)·ΔB

use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};
use burn::module::Param;
use burn::tensor::activation;

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

/// Selective State Space (S6) module
///
/// Implements input-dependent state space transformation where parameters
/// Δ (discretization step), B (input matrix), and C (output matrix) are
/// computed from the input, enabling selective information propagation.
#[derive(Module, Debug)]
pub struct SelectiveStateSpace<B: Backend> {
    /// Input projection (expands channels)
    pub in_proj: Linear<B>,
    /// Output projection (contracts channels)
    pub out_proj: Linear<B>,
    /// Project input to Δ (discretization step)
    pub dt_proj: Linear<B>,
    /// Project input to B (input matrix, rank-reduced)
    pub b_proj: Linear<B>,
    /// Project input to C (output matrix, rank-reduced)
    pub c_proj: Linear<B>,
    /// Learnable state matrix A (initialized as HiPPO or random)
    pub a_log: Param<Tensor<B, 1>>,
    /// Skip connection parameter
    pub d: Param<Tensor<B, 1>>,
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// State dimension (stored for reference)
    state_dim: usize,
    /// Expansion factor
    expand_factor: usize,
    /// dt rank for low-rank projection
    dt_rank: usize,
}

impl<B: Backend> SelectiveStateSpace<B> {
    /// Create new SelectiveStateSpace module
    pub fn new(config: &SelectiveStateSpaceConfig, device: &B::Device) -> Self {
        let inner_dim = config.input_dim * config.expand_factor;
        
        // Input projection expands channels
        let in_proj = LinearConfig::new(config.input_dim, inner_dim * 2).init(device);
        
        // Output projection contracts back
        let out_proj = LinearConfig::new(inner_dim, config.output_dim).init(device);
        
        // Discretization step projection (low-rank)
        let dt_proj = LinearConfig::new(config.dt_rank, inner_dim).init(device);
        
        // B and C projections (rank-reduced)
        let b_proj = LinearConfig::new(inner_dim, config.state_dim).init(device);
        let c_proj = LinearConfig::new(inner_dim, config.state_dim).init(device);
        
        // Initialize A as log of random values (learnable)
        // Using real part of HiPPO initialization: A[n] = -(n+1) for n=0,...,N-1
        let a_init: Vec<f64> = (0..inner_dim * config.state_dim)
            .map(|i| {
                let n = i % config.state_dim;
                -((n + 1) as f64).ln()
            })
            .collect();
        let a_log = Tensor::<B, 1>::from_data(a_init.as_slice(), device)
            .reshape([inner_dim * config.state_dim]);
        
        // Skip connection parameter (initialized to 1.0)
        let d = Tensor::ones([inner_dim], device);
        
        Self {
            in_proj,
            out_proj,
            dt_proj,
            b_proj,
            c_proj,
            a_log: Param::from_tensor(a_log),
            d: Param::from_tensor(d),
            input_dim: config.input_dim,
            output_dim: config.output_dim,
            state_dim: config.state_dim,
            expand_factor: config.expand_factor,
            dt_rank: config.dt_rank,
        }
    }
    
    /// Forward pass through selective state space
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [..., seq_len, input_dim]
    ///
    /// # Returns
    /// * Output tensor of shape [..., seq_len, output_dim]
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D>
    where
        B: Backend,
    {
        let batch_dims = input.dims();
        let seq_len = batch_dims[D - 2];
        let input_dim = batch_dims[D - 1];
        let batch_size: usize = batch_dims[..D - 2].iter().product();
        
        // Reshape to [batch, seq, input_dim] for separate batch processing
        let x_in = input.reshape([batch_size, seq_len, input_dim]);
        
        // Project and split into x and residual
        let proj = self.in_proj.forward(x_in);
        let (x, residual): (Tensor<B, 3>, Tensor<B, 3>) = {
            let inner_dim = self.input_dim * self.expand_factor;
            let x_part = proj.clone().slice([0..batch_size, 0..seq_len, 0..inner_dim]);
            let res_part = proj.slice([0..batch_size, 0..seq_len, inner_dim..inner_dim*2]);
            (x_part, res_part)
        };
        
        // Compute selective parameters from input
        // All have shape [batch, seq, ...]
        let dt = self.compute_dt(&x);
        let b = self.b_proj.forward(x.clone());
        let c = self.c_proj.forward(x.clone());
        
        // Apply selective SSM using parallel scan
        let y = self.selective_scan(x, dt, b, c);
        
        // Skip connection with gating
        let output = y * activation::sigmoid(residual) * self.d.val().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0);
        
        // Project back to output dimension
        let out = self.out_proj.forward(output);
        
        // Reshape back to original batch dimensions
        let mut out_dims = batch_dims;
        out_dims[D - 1] = self.output_dim;
        out.reshape(out_dims)
    }
    
    /// Compute discretization step Δ from input
    fn compute_dt(&self, x: &Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq, _] = x.dims();
        let dt_rank = self.dt_rank;
        
        // First project to low-rank space
        // x: [batch, seq, inner]
        let x_rank = x.clone().slice([0..batch, 0..seq, 0..dt_rank]);
        
        // Expand to full dimension
        let dt_unbounded = self.dt_proj.forward(x_rank);
        
        // Softplus activation to ensure positive Δ
        activation::softplus(dt_unbounded, 1.0)
    }
    
    /// Selective scan operation using Parallel Associative Scan
    fn selective_scan(
        &self,
        x: Tensor<B, 3>, // [batch, seq, inner]
        dt: Tensor<B, 3>, // [batch, seq, inner]
        b: Tensor<B, 3>, // [batch, seq, state]
        c: Tensor<B, 3>, // [batch, seq, state]
    ) -> Tensor<B, 3> {
        let [_batch, _seq_len, inner_dim] = x.dims();
        let state_dim = self.state_dim;
        
        // Get A from log parameter (A = -exp(parameter))
        let a = self.a_log.val().exp().neg().reshape([inner_dim, state_dim]);
        
        // Discretize: Ā = exp(ΔA)
        // dt: [batch, seq, inner] -> [batch, seq, inner, 1]
        let dt_expanded: Tensor<B, 4> = dt.unsqueeze_dim::<4>(3);
        // a: [inner, state] -> [1, 1, inner, state]
        let a_expanded: Tensor<B, 4> = a.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);
        
        // Ā = exp(Δ * A) -> [batch, seq, inner, state]
        let a_bar = (dt_expanded.clone() * a_expanded).exp();
        
        // B̄ = Δ * B
        // b: [batch, seq, state] -> [batch, seq, 1, state]
        let b_expanded: Tensor<B, 4> = b.unsqueeze_dim::<4>(2);
        let b_bar = dt_expanded * b_expanded; // [batch, seq, inner, state]
        
        // U = B̄ * x
        // x: [batch, seq, inner] -> [batch, seq, inner, 1]
        let x_expanded: Tensor<B, 4> = x.unsqueeze_dim::<4>(3);
        // u = b_bar * x_expanded -> [batch, seq, inner, state]
        // Broadcasting: [batch, seq, inner, state] * [batch, seq, inner, 1] works
        let u = b_bar * x_expanded;
        
        // Perform Parallel Associative Scan
        // h_t = Ā_t * h_{t-1} + U_t
        let h = self.parallel_scan(a_bar, u);
        
        // Output: y = (C * h).sum(-1)
        // c: [batch, seq, state] -> [batch, seq, 1, state]
        let c_expanded: Tensor<B, 4> = c.unsqueeze_dim(2);
        // h: [batch, seq, inner, state]
        // y: [batch, seq, inner]
        (h * c_expanded).sum_dim(3).squeeze(3)
    }
    
    /// Parallel Associative Scan (Hillis-Steele)
    /// Computes prefix scan for: h_t = a_t * h_{t-1} + u_t
    fn parallel_scan(
        &self,
        a_in: Tensor<B, 4>,
        u_in: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let [batch, seq_len, inner, state] = a_in.dims();
        
        let mut a = a_in;
        let mut u = u_in;
        
        // Iterate log2(N) times
        let steps = (seq_len as f64).log2().ceil() as usize;
        
        for i in 0..steps {
            let k = 1 << i;
            if k >= seq_len { break; }
            
            // Slice views
            let a_curr = a.clone().slice([0..batch, k..seq_len, 0..inner, 0..state]);
            let u_curr = u.clone().slice([0..batch, k..seq_len, 0..inner, 0..state]);
            
            let a_prev = a.clone().slice([0..batch, 0..seq_len-k, 0..inner, 0..state]);
            let u_prev = u.clone().slice([0..batch, 0..seq_len-k, 0..inner, 0..state]);
            
            // Compute updates
            let a_new = a_curr.clone() * a_prev.clone();
            let u_new = a_curr * u_prev + u_curr;
            
            // Concatenate with unchanged prefix
            let a_prefix = a.clone().slice([0..batch, 0..k, 0..inner, 0..state]);
            let u_prefix = u.clone().slice([0..batch, 0..k, 0..inner, 0..state]);
            
            a = Tensor::cat(vec![a_prefix, a_new], 1);
            u = Tensor::cat(vec![u_prefix, u_new], 1);
        }
        
        u
    }
    
    /// Forward pass for 3D volumetric data
    ///
    /// Processes data as flattened sequences along spatial dimensions
    pub fn forward_3d(&self, input: Tensor<B, 5>) -> Tensor<B, 5> {
        let [batch, channels, depth, height, width] = input.dims();
        let seq_len = depth * height * width;
        
        // Reshape to [batch, seq_len, channels]
        let flat = input
            .permute([0, 2, 3, 4, 1])
            .reshape([batch, seq_len, channels]);
        
        // Apply SSM
        let out = self.forward(flat);
        
        // Reshape back
        out.reshape([batch, seq_len, self.output_dim])
            .permute([0, 2, 1])
            .reshape([batch, self.output_dim, depth, height, width])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    
    #[test]
    fn test_state_space_creation() {
        let device = <NdArray as Backend>::Device::default();
        let config = SelectiveStateSpaceConfig::new_with_dims(64, 64);
        let ssm = SelectiveStateSpace::<NdArray>::new(&config, &device);
        
        assert_eq!(ssm.input_dim, 64);
        assert_eq!(ssm.output_dim, 64);
        assert_eq!(ssm.state_dim, 16);
    }
    
    #[test]
    fn test_forward_pass() {
        let device = <NdArray as Backend>::Device::default();
        let config = SelectiveStateSpaceConfig::new_with_dims(32, 32);
        let ssm = SelectiveStateSpace::<NdArray>::new(&config, &device);
        
        // Create test input: [batch=2, seq=8, dim=32]
        let input = Tensor::<NdArray, 3>::zeros([2, 8, 32], &device);
        let output = ssm.forward(input);
        
        assert_eq!(output.dims(), [2, 8, 32]);
    }
    
    #[test]
    fn test_forward_3d() {
        let device = <NdArray as Backend>::Device::default();
        let config = SelectiveStateSpaceConfig::new_with_dims(16, 16);
        let ssm = SelectiveStateSpace::<NdArray>::new(&config, &device);
        
        // Create test input: [batch=1, channels=16, depth=4, height=8, width=8]
        let input = Tensor::<NdArray, 5>::zeros([1, 16, 4, 8, 8], &device);
        let output = ssm.forward_3d(input);
        
        assert_eq!(output.dims(), [1, 16, 4, 8, 8]);
    }
}
