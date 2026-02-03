//! VMamba Block - Hybrid CNN-SSM Architecture
//!
//! Combines the local feature extraction of CNNs with the global context modeling
//! of State Space Models through cross-scan mechanism. This is the building block
//! of VMambaMorph architecture.
//!
//! # Architecture
//!```text
//! Input
//!   │
//!   ├──► LayerNorm ──► Conv3x3 (depthwise) ──┐
//!   │                                          │
//!   └──► (skip connection) ◄───────────────────┤
//!                                              │
//!   ┌──────────────────────────────────────────┘
//!   │
//!   ▼
//! Cross-Scan ──► SSM (per direction) ──► Cross-Merge
//!   │
//!   ▼
//! LayerNorm ──► Conv1x1 ──► GELU ──► Conv1x1
//!   │
//!   └──► (skip connection from above)
//!   │
//!   ▼
//! Output
//!```

use burn::prelude::*;
use burn::nn::{Gelu, LayerNorm, Linear};
use burn::nn::conv::{Conv3d, Conv3dConfig};
use burn::nn::PaddingConfig3d;

use super::state_space::{SelectiveStateSpace, SelectiveStateSpaceConfig};
use super::cross_scan::{CrossScan, CrossScanConfig, ScanDirection};

/// Configuration for VMamba Block
#[derive(Config, Debug, PartialEq)]
pub struct VMambaBlockConfig {
    /// Input/output channel dimension
    pub dim: usize,
    /// Expansion factor for SSM hidden dimension
    #[config(default = "2")]
    pub expand_factor: usize,
    /// State dimension for SSM
    #[config(default = "16")]
    pub state_dim: usize,
    /// Dropout probability
    #[config(default = "0.0")]
    pub dropout: f64,
    /// Use 3D cross-scan for volumetric data
    #[config(default = "true")]
    pub use_3d: bool,
    /// Drop path rate for stochastic depth
    #[config(default = "0.0")]
pub drop_path_rate: f64,
}

impl VMambaBlockConfig {
    /// Create new VMambaBlock config
    pub fn new_with_dim(dim: usize) -> Self {
        Self {
            dim,
            expand_factor: 2,
            state_dim: 16,
            dropout: 0.0,
            use_3d: true,
            drop_path_rate: 0.0,
        }
    }
}

use burn::module::Ignored;

/// VMamba Block combining CNN and SSM
///
/// Implements the core VMamba block from VMambaMorph paper:
/// 1. Depthwise convolution for local feature extraction
/// 2. Cross-scan SSM for global context
/// 3. FFN for channel mixing
#[derive(Module, Debug)]
pub struct VMambaBlock<B: Backend> {
    /// Layer normalization before SSM branch
    pub norm1: LayerNorm<B>,
    /// Layer normalization before FFN
    pub norm2: LayerNorm<B>,
    /// Depthwise convolution for local features
    pub dwconv: Conv3d<B>,
    /// Cross-scan module
    pub cross_scan: Ignored<CrossScan>,
    /// Selective state space module
    pub ssm: SelectiveStateSpace<B>,
    /// First linear layer in FFN
    pub ffn_expand: Linear<B>,
    /// Second linear layer in FFN
    pub ffn_project: Linear<B>,
    /// Activation function
    act: Gelu,
    /// Configuration
    pub config: Ignored<VMambaBlockConfig>,
}

impl<B: Backend> VMambaBlock<B> {
    /// Create new VMambaBlock
    pub fn new(config: &VMambaBlockConfig, device: &B::Device) -> Self {
        let ssm_config = SelectiveStateSpaceConfig::new_with_dims(config.dim, config.dim)
            .with_state_dim(config.state_dim)
            .with_expand_factor(config.expand_factor)
            .with_dropout(config.dropout);
        
        let cross_scan_config = if config.use_3d {
            CrossScanConfig::new_3d()
        } else {
            CrossScanConfig::new_2d()
        };
        
        // Depthwise convolution (groups = input channels)
        let dwconv_config = Conv3dConfig::new(
            [config.dim, config.dim],
            [3, 3, 3],
        )
        .with_stride([1, 1, 1])
        .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
        .with_groups(config.dim)
        .with_bias(true);
        
        let inner_dim = config.dim * 4; // Standard FFN expansion
        
        Self {
            norm1: burn::nn::LayerNormConfig::new(config.dim).init(device),
            norm2: burn::nn::LayerNormConfig::new(config.dim).init(device),
            dwconv: dwconv_config.init(device),
            cross_scan: Ignored(CrossScan::new(&cross_scan_config)),
            ssm: SelectiveStateSpace::new(&ssm_config, device),
            ffn_expand: burn::nn::LinearConfig::new(config.dim, inner_dim).with_bias(true).init(device),
            ffn_project: burn::nn::LinearConfig::new(inner_dim, config.dim).with_bias(true).init(device),
            act: Gelu::new(),
            config: Ignored(VMambaBlockConfig::new(config.dim)),
        }
    }
    
    /// Forward pass for 5D volumetric input [batch, channels, depth, height, width]
    pub fn forward(&self, input: Tensor<B, 5>) -> Tensor<B, 5> {
        let [batch, channels, depth, height, width] = input.dims();
        let _device = input.device();
        
        // First branch: SSM with cross-scan
        // LayerNorm expects channels last: [B, C, D, H, W] -> [B, D, H, W, C]
        let x_norm1 = self.norm1.forward(input.clone().permute([0, 2, 3, 4, 1]));
        let x_norm1 = x_norm1.permute([0, 4, 1, 2, 3]);
        
        // Depthwise convolution for local features
        let x_conv = self.dwconv.forward(x_norm1);
        
        // Cross-scan: get sequences from all directions
        let directions = self.cross_scan.directions();
        let scanned_sequences = self.cross_scan.apply(x_conv);
        
        // Apply SSM to each scanned direction
        let processed_sequences: Vec<Tensor<B, 3>> = scanned_sequences
            .into_iter()
            .map(|seq| {
                // seq shape: [batch, channels, seq_len]
                // Reshape to [batch*seq_len, channels] for SSM
                let [b, c, s] = seq.dims();
                let seq_flat = seq.permute([0, 2, 1]).reshape([b * s, c]);
                
                // Apply SSM
                let processed = self.ssm.forward(seq_flat);
                
                // Reshape back
                processed.reshape([b, s, c]).permute([0, 2, 1])
            })
            .collect();
        
        // Cross-merge: combine all directions
        let ssm_out = self.merge_directions(processed_sequences, [batch, channels, depth, height, width], directions);
        
        // First residual connection
        let x = input + ssm_out;
        
        // Second branch: FFN
        // LayerNorm expects channels last: [B, C, D, H, W] -> [B, D, H, W, C]
        let x_norm2 = self.norm2.forward(x.clone().permute([0, 2, 3, 4, 1]));
        let x_norm2 = x_norm2.permute([0, 4, 1, 2, 3]);
        
        // Flatten spatial for FFN
        let [b, c, d, h, w] = x_norm2.dims();
        let x_flat = x_norm2.permute([0, 2, 3, 4, 1]).reshape([b * d * h * w, c]);
        
        // FFN: expand -> activate -> project
        let x_ffn = self.ffn_expand.forward(x_flat);
        let x_ffn = self.act.forward(x_ffn);
        let x_ffn = self.ffn_project.forward(x_ffn);
        
        // Reshape back
        let x_ffn = x_ffn.reshape([b, d, h, w, c]).permute([0, 4, 1, 2, 3]);
        
        // Second residual connection with optional drop path
        // Drop path (stochastic depth) is applied during training
        // For inference, it's a no-op
        if false {  // Simplified: always false for now
            // Stochastic depth (training only, simplified here)
            x + x_ffn
        } else {
            x + x_ffn
        }
    }
    
    /// Merge processed sequences from all directions back to spatial tensor
    fn merge_directions(
        &self,
        sequences: Vec<Tensor<B, 3>>,
        dims: [usize; 5],
        directions: &[ScanDirection],
    ) -> Tensor<B, 5> {
        let [_batch, _channels, depth, height, width] = dims;
        let device = sequences[0].device();
        
        // Merge each sequence back to spatial
        let mut merged = Vec::new();
        for (seq, &dir) in sequences.into_iter().zip(directions.iter()) {
            let spatial = if self.cross_scan.use_3d() {
                super::cross_scan::Scan3D::merge(seq, depth, height, width, dir)
            } else {
                // For 2D case, we'd need different handling
                // This is a simplified version assuming 3D
                super::cross_scan::Scan3D::merge(seq, depth, height, width, dir)
            };
            merged.push(spatial);
        }
        
        // Average all directions
        let sum = merged.into_iter().fold(
            Tensor::zeros(dims, &device),
            |acc, t| acc + t,
        );
        
        sum / (directions.len() as f64)
    }
    
    /// Forward for 4D input (treat as single depth slice)
    pub fn forward_4d(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, channels, height, width] = input.dims();
        // Expand to 5D with depth=1
        let expanded = input.unsqueeze_dim(2);
        let output = self.forward(expanded);
        // Remove depth dimension
        output.reshape([batch, channels, height, width])
    }
}

/// Stack of VMamba Blocks for hierarchical feature processing
#[derive(Module, Debug)]
pub struct VMambaStage<B: Backend> {
    /// Sequence of VMamba blocks
    pub blocks: Vec<VMambaBlock<B>>,
    /// Downsampling layer (optional)
    pub downsample: Option<Conv3d<B>>,
}

impl<B: Backend> VMambaStage<B> {
    /// Create new VMamba stage
    ///
    /// # Arguments
    /// * `dim` - Channel dimension
    /// * `depth` - Number of blocks in this stage
    /// * `downsample` - Whether to downsample at the end
    /// * `device` - Device
    pub fn new(
        dim: usize,
        depth: usize,
        downsample: bool,
        device: &B::Device,
    ) -> Self {
        let config = VMambaBlockConfig::new_with_dim(dim);
        
        let blocks: Vec<_> = (0..depth)
            .map(|_| VMambaBlock::new(&config, device))
            .collect();
        
        let downsample_layer = if downsample {
            let ds_config = Conv3dConfig::new([dim, dim * 2], [3, 3, 3])
                .with_stride([2, 2, 2])
                .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
                .with_bias(false);
            Some(ds_config.init(device))
        } else {
            None
        };
        
        Self {
            blocks,
            downsample: downsample_layer,
        }
    }
    
    /// Forward pass through stage
    pub fn forward(&self, input: Tensor<B, 5>) -> Tensor<B, 5> {
        let mut x = input;
        
        // Pass through all blocks
        for block in &self.blocks {
            x = block.forward(x);
        }
        
        // Downsample if configured
        match &self.downsample {
            Some(ds) => ds.forward(x),
            None => x,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    
    #[test]
    fn test_vmamba_block_creation() {
        let device = <NdArray as Backend>::Device::default();
        let config = VMambaBlockConfig::new_with_dim(64);
        let block = VMambaBlock::<NdArray>::new(&config, &device);
        
        // Check that SSM was created with correct dimensions
        assert_eq!(block.ssm.input_dim, 64);
        assert_eq!(block.ssm.output_dim, 64);
    }
    
    #[test]
    fn test_vmamba_block_forward() {
        let device = <NdArray as Backend>::Device::default();
        let config = VMambaBlockConfig::new_with_dim(32);
        let block = VMambaBlock::<NdArray>::new(&config, &device);
        
        // Test input: [batch=1, channels=32, depth=4, height=8, width=8]
        let input = Tensor::<NdArray, 5>::zeros([1, 32, 4, 8, 8], &device);
        let output = block.forward(input);
        
        assert_eq!(output.dims(), [1, 32, 4, 8, 8]);
    }
    
    #[test]
    fn test_vmamba_stage() {
        let device = <NdArray as Backend>::Device::default();
        let stage = VMambaStage::<NdArray>::new(32, 2, true, &device);
        
        // Test input
        let input = Tensor::<NdArray, 5>::zeros([1, 32, 8, 16, 16], &device);
        let output = stage.forward(input);
        
        // With downsampling, spatial dims should be halved and channels doubled
        assert_eq!(output.dims(), [1, 64, 4, 8, 8]);
    }
}
