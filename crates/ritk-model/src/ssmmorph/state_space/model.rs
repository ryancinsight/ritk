use burn::module::Param;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation;

use super::config::SelectiveStateSpaceConfig;
use super::scan::selective_scan;

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
    /// Project input to low-rank space
    pub dt_in_proj: Linear<B>,
    /// Project from low-rank space to Δ (discretization step)
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
    pub state_dim: usize,
    /// Expansion factor
    pub expand_factor: usize,
    /// dt rank for low-rank projection
    pub dt_rank: usize,
}

impl<B: Backend> SelectiveStateSpace<B> {
    /// Create new SelectiveStateSpace module
    pub fn new(config: &SelectiveStateSpaceConfig, device: &B::Device) -> Self {
        let inner_dim = config.input_dim * config.expand_factor;

        // Input projection expands channels
        let in_proj = LinearConfig::new(config.input_dim, inner_dim * 2).init(device);

        // Output projection contracts back
        let out_proj = LinearConfig::new(inner_dim, config.output_dim).init(device);

        // Discretization step projection (low-rank contraction and expansion)
        let dt_in_proj = LinearConfig::new(inner_dim, config.dt_rank).init(device);
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
            dt_in_proj,
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
            let x_part = proj
                .clone()
                .slice([0..batch_size, 0..seq_len, 0..inner_dim]);
            let res_part = proj.slice([0..batch_size, 0..seq_len, inner_dim..inner_dim * 2]);
            (x_part, res_part)
        };

        // Compute selective parameters from input
        let dt = self.compute_dt(&x);
        let b = self.b_proj.forward(x.clone());
        let c = self.c_proj.forward(x.clone());

        // Apply selective SSM using parallel scan
        let y = selective_scan(self.state_dim, self.a_log.val(), x, dt, b, c);

        // Skip connection with gating
        let output = y
            * activation::sigmoid(residual)
            * self.d.val().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0);

        // Project back to output dimension
        let out = self.out_proj.forward(output);

        // Reshape back to original batch dimensions
        let mut out_dims = batch_dims;
        out_dims[D - 1] = self.output_dim;
        out.reshape(out_dims)
    }

    /// Compute discretization step Δ from input
    fn compute_dt(&self, x: &Tensor<B, 3>) -> Tensor<B, 3> {
        // First project to low-rank space analytically instead of slicing
        let x_rank = self.dt_in_proj.forward(x.clone());

        // Expand to full dimension
        let dt_unbounded = self.dt_proj.forward(x_rank);

        // Softplus activation to ensure positive Δ
        activation::softplus(dt_unbounded, 1.0)
    }

    /// Forward pass for 3D volumetric data
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
