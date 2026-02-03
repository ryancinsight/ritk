//! SSMMorph Decoder - Hierarchical Feature Upsampling
//!
//! Decoder that progressively upsamples bottleneck features while incorporating
//! skip connections from the encoder. Uses VMamba blocks for feature refinement
//! at each scale.

use burn::prelude::*;
use burn::nn::conv::{Conv3d, Conv3dConfig, ConvTranspose3d, ConvTranspose3dConfig};
use burn::nn::{LayerNorm, PaddingConfig3d};

use super::vmamba_block::{VMambaBlock, VMambaBlockConfig};

/// Configuration for decoder stage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DecoderStageConfig {
    /// Number of input channels (from upsample + skip)
    pub in_channels: usize,
    /// Number of output channels
    pub out_channels: usize,
    /// Number of VMamba blocks
    pub depth: usize,
}

/// Configuration for SSMMorph decoder
#[derive(Config, Debug, PartialEq)]
pub struct SSMMorphDecoderConfig {
    /// Bottleneck channel dimension
    pub bottleneck_channels: usize,
    /// Output channel dimension (typically 3 for 3D displacement field)
    pub out_channels: usize,
    /// Number of decoder stages
    #[config(default = "4")]
    pub num_stages: usize,
    /// Number of blocks per stage
    #[config(default = "2")]
    pub blocks_per_stage: usize,
    /// Use skip connections
    #[config(default = "true")]
    pub use_skip_connections: bool,
}

impl SSMMorphDecoderConfig {
    /// Create decoder config matching encoder
    pub fn from_encoder(encoder_channels: &[usize], out_channels: usize) -> Self {
        let num_stages = encoder_channels.len();
        let bottleneck_channels = encoder_channels.last().copied().unwrap_or(256);
        
        Self {
            bottleneck_channels,
            out_channels,
            num_stages,
            blocks_per_stage: 2,
            use_skip_connections: true,
        }
    }
    
    /// Get stage configurations given encoder stage channels
    pub fn stage_configs(&self, encoder_channels: &[usize]) -> Vec<DecoderStageConfig> {
        let mut configs = Vec::new();
        let mut in_ch = self.bottleneck_channels;
        
        // Iterate reversed, skipping the bottleneck itself (last channel)
        // [C1, C2, C3, C4] -> [C3, C2, C1]
        for &skip_ch in encoder_channels.iter().rev().skip(1) {
            let out_ch = skip_ch;
            
            // Input is upsampled + skip connection
            let stage_in_ch = in_ch;
            
            configs.push(DecoderStageConfig {
                in_channels: stage_in_ch,
                out_channels: out_ch,
                depth: self.blocks_per_stage,
            });
            
            in_ch = out_ch;
        }
        
        // Add final stage (upsample to original resolution, no skip)
        // Typically output channels = C1 / 2
        let final_ch = encoder_channels.first().map(|c| c / 2).unwrap_or(16);
        configs.push(DecoderStageConfig {
            in_channels: in_ch,
            out_channels: final_ch,
            depth: self.blocks_per_stage,
        });
        
        configs
    }
}

/// Single decoder stage with upsampling and VMamba blocks
#[derive(Module, Debug)]
pub struct DecoderStage<B: Backend> {
    /// Upsampling layer (transposed convolution)
    pub upsample: ConvTranspose3d<B>,
    /// Projection for concatenated features
    pub fusion: Conv3d<B>,
    /// VMamba blocks
    pub blocks: Vec<VMambaBlock<B>>,
    /// Layer normalization after fusion
    pub norm: LayerNorm<B>,
}

impl<B: Backend> DecoderStage<B> {
    /// Create new decoder stage
    fn new(
        in_channels: usize,
        skip_channels: usize,
        out_channels: usize,
        depth: usize,
        device: &B::Device,
    ) -> Self {
        // Upsampling (doubles spatial resolution, halves channels)
        let upsample_config = ConvTranspose3dConfig::new(
            [in_channels, out_channels],
            [4, 4, 4],
        )
        .with_stride([2, 2, 2])
        .with_padding([1, 1, 1])
        .with_bias(false);
        let upsample = upsample_config.init(device);
        
        // Fusion convolution for skip connection
        let fusion_in_ch = out_channels + skip_channels;
        let fusion_config = Conv3dConfig::new(
            [fusion_in_ch, out_channels],
            [3, 3, 3],
        )
        .with_stride([1, 1, 1])
        .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
        .with_bias(false);
        let fusion = fusion_config.init(device);
        
        // VMamba blocks for feature refinement
        let block_config = VMambaBlockConfig::new_with_dim(out_channels);
        let blocks: Vec<_> = (0..depth)
            .map(|_| VMambaBlock::new(&block_config, device))
            .collect();
        
        // Layer normalization
        let norm = burn::nn::LayerNormConfig::new(out_channels).init(device);
        
        Self {
            upsample,
            fusion,
            blocks,
            norm,
        }
    }
    
    /// Forward pass through decoder stage
    ///
    /// # Arguments
    /// * `input` - Input from previous stage [batch, in_ch, d, h, w]
    /// * `skip` - Skip connection from encoder [batch, skip_ch, d*2, h*2, w*2]
    ///
    /// # Returns
    /// * Output features [batch, out_ch, d*2, h*2, w*2]
    pub fn forward(&self, input: Tensor<B, 5>, skip: Option<Tensor<B, 5>>) -> Tensor<B, 5> {
        // Upsample input
        let x_up = self.upsample.forward(input);
        
        // Fuse with skip connection if available
        let x_fused = match skip {
            Some(skip_feat) => {
                // Concatenate along channel dimension
                let x_cat = Tensor::cat(vec![x_up, skip_feat], 1);
                self.fusion.forward(x_cat)
            }
            None => x_up,
        };
        
        // Normalize after fusion
        // Permute for LayerNorm: [B, C, D, H, W] -> [B, D, H, W, C]
        let x_norm = self.norm.forward(x_fused.permute([0, 2, 3, 4, 1]));
        let x_norm = x_norm.permute([0, 4, 1, 2, 3]);
        
        // Pass through VMamba blocks
        let mut x = x_norm;
        for block in &self.blocks {
            x = block.forward(x);
        }
        
        x
    }
}

use burn::module::Ignored;

/// SSMMorph Decoder - Hierarchical upsampling with skip connections
#[derive(Module, Debug)]
pub struct SSMMorphDecoder<B: Backend> {
    /// Decoder stages
    pub stages: Vec<DecoderStage<B>>,
    /// Final output projection
    pub output_proj: Conv3d<B>,
    /// Configuration
    pub config: Ignored<SSMMorphDecoderConfig>,
    /// Skip channels for each stage
    pub skip_channels: Ignored<Vec<usize>>,
}

impl<B: Backend> SSMMorphDecoder<B> {
    /// Create new SSMMorph decoder
    ///
    /// # Arguments
    /// * `config` - Decoder configuration
    /// * `encoder_channels` - Channel dimensions from encoder stages (for skip connections)
    /// * `device` - Device
    pub fn new(
        config: &SSMMorphDecoderConfig,
        encoder_channels: &[usize],
        device: &B::Device,
    ) -> Self {
        let stage_configs = config.stage_configs(encoder_channels);
        
        let stages: Vec<_> = stage_configs
            .iter()
            .enumerate()
            .map(|(i, stage_config)| {
                let skip_ch = if i < encoder_channels.len() - 1 {
                    encoder_channels[encoder_channels.len() - 2 - i]
                } else {
                    0
                };
                
                DecoderStage::new(
                    stage_config.in_channels,
                    skip_ch,
                    stage_config.out_channels,
                    stage_config.depth,
                    device,
                )
            })
            .collect();
            
        // Final projection
        let proj_config = Conv3dConfig::new(
            [config.out_channels, config.out_channels],
            [3, 3, 3],
        )
        .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
        .with_bias(true);
        
        let output_proj = proj_config.init(device);
        
        Self {
            stages,
            output_proj,
            config: Ignored(config.clone()),
            skip_channels: Ignored(encoder_channels.to_vec()),
        }
    }
    
    /// Forward pass through decoder
    ///
    /// # Arguments
    /// * `bottleneck` - Bottleneck features from encoder
    /// * `skip_features` - Skip connections from encoder (in encoder order)
    ///
    /// # Returns
    /// * Output tensor [batch, out_channels, D, H, W]
    pub fn forward(
        &self,
        bottleneck: Tensor<B, 5>,
        skip_features: &[Tensor<B, 5>],
    ) -> Tensor<B, 5> {
        let mut x = bottleneck;
        
        // Reverse skip features to match decoder order
        let skip_iter: Vec<_> = skip_features.iter().rev().collect();
        
        // Pass through decoder stages
        for (i, stage) in self.stages.iter().enumerate() {
            let skip = skip_iter.get(i).copied().cloned();
            x = stage.forward(x, skip);
        }
        
        // Final output projection
        self.output_proj.forward(x)
    }
    
    /// Get number of stages
    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    
    #[test]
    fn test_decoder_creation() {
        let device = <NdArray as Backend>::Device::default();
        let encoder_channels = vec![32, 64, 128, 256];
        let config = SSMMorphDecoderConfig::from_encoder(&encoder_channels, 3);
        let decoder = SSMMorphDecoder::<NdArray>::new(&config, &encoder_channels, &device);
        
        assert_eq!(decoder.num_stages(), 4);
    }
    
    #[test]
    fn test_decoder_forward() {
        let device = <NdArray as Backend>::Device::default();
        let encoder_channels = vec![32, 64, 128, 256];
        let config = SSMMorphDecoderConfig::from_encoder(&encoder_channels, 3);
        let decoder = SSMMorphDecoder::<NdArray>::new(&config, &encoder_channels, &device);
        
        // Reduced dimensions for testing performance
        // Bottleneck: [batch=1, ch=256, d=1, h=2, w=2]
        let bottleneck = Tensor::<NdArray, 5>::zeros([1, 256, 1, 2, 2], &device);
        
        // Skip features (in encoder order: low to high resolution)
        // Skip1: [1, 32, 8, 16, 16]
        let skip1 = Tensor::<NdArray, 5>::zeros([1, 32, 8, 16, 16], &device);
        // Skip2: [1, 64, 4, 8, 8]
        let skip2 = Tensor::<NdArray, 5>::zeros([1, 64, 4, 8, 8], &device);
        // Skip3: [1, 128, 2, 4, 4]
        let skip3 = Tensor::<NdArray, 5>::zeros([1, 128, 2, 4, 4], &device);
        let skip_features = vec![skip1, skip2, skip3];
        
        let output = decoder.forward(bottleneck, &skip_features);
        
        // Output should be full resolution with 3 channels (displacement field)
        // Upsampling happens 4 times: 1x2x2 -> 2x4x4 -> 4x8x8 -> 8x16x16 -> 16x32x32
        assert_eq!(output.dims(), [1, 3, 16, 32, 32]);
    }
    
    #[test]
    fn test_decoder_stage() {
        let device = <NdArray as Backend>::Device::default();
        let stage = DecoderStage::<NdArray>::new(256, 128, 128, 2, &device);
        
        // Input: [batch=1, ch=256, d=2, h=8, w=8]
        let input = Tensor::<NdArray, 5>::zeros([1, 256, 2, 8, 8], &device);
        // Skip: [batch=1, ch=128, d=4, h=16, w=16]
        let skip = Tensor::<NdArray, 5>::zeros([1, 128, 4, 16, 16], &device);
        
        let output = stage.forward(input, Some(skip));
        
        // Output should be upsampled 2x
        assert_eq!(output.dims(), [1, 128, 4, 16, 16]);
    }
}
