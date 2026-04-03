use super::*;
use burn_ndarray::NdArray;

type TestBackend = NdArray<f32>;

#[test]
fn test_encoder_config() {
    let config = SSMMorphEncoderConfig::for_registration();
    assert_eq!(config.num_stages, 4);
    assert_eq!(config.base_channels, 32);

    let channels = config.stage_channels();
    assert_eq!(channels, vec![32, 64, 128, 256]);
}

#[test]
fn test_encoder_stage_creation() {
    let device = Default::default();
    let config = EncoderStageConfig {
        in_channels: 2,
        out_channels: 32,
        depth: 2,
        downsample: true,
    };

    let stage = EncoderStage::<TestBackend>::new(&config, &device);
    assert_eq!(stage.blocks.len(), 2);
    assert!(stage.downsample.is_some());
    assert!(stage.proj.is_some());
}

#[test]
fn test_encoder_stage_forward() {
    let device = Default::default();
    let config = EncoderStageConfig {
        in_channels: 2,
        out_channels: 32,
        depth: 1,
        downsample: true,
    };

    let stage = EncoderStage::<TestBackend>::new(&config, &device);

    // Input: [batch=1, channels=2, depth=16, height=64, width=64]
    let input = Tensor::<TestBackend, 5>::zeros([1, 2, 16, 64, 64], &device);
    let (features, output) = stage.forward(input);

    // Features should have output channels and same spatial size
    assert_eq!(features.dims(), [1, 32, 16, 64, 64]);

    // Output should be downsampled
    assert!(output.is_some());
    let out = output.unwrap();
    assert_eq!(out.dims(), [1, 32, 8, 32, 32]);
}

#[test]
fn test_encoder_creation() {
    let device = Default::default();
    let config = SSMMorphEncoderConfig::for_registration();
    let encoder = SSMMorphEncoder::<TestBackend>::new(&config, &device);

    assert_eq!(encoder.num_stages(), 4);
    assert_eq!(encoder.stage_channels(), &[32, 64, 128, 256]);
}

#[test]
fn test_encoder_forward() {
    let device = Default::default();
    let config = SSMMorphEncoderConfig::lightweight();
    let encoder = SSMMorphEncoder::<TestBackend>::new(&config, &device);

    // Input: [batch=1, channels=2, depth=16, height=64, width=64]
    let input = Tensor::<TestBackend, 5>::zeros([1, 2, 16, 64, 64], &device);
    let (features, bottleneck) = encoder.forward(input);

    // Should have features from each stage
    assert_eq!(features.len(), 3);

    // Check feature shapes (lightweight config: 3 stages)
    assert_eq!(features[0].dims(), [1, 16, 16, 64, 64]);
    assert_eq!(features[1].dims(), [1, 32, 8, 32, 32]);
    assert_eq!(features[2].dims(), [1, 64, 4, 16, 16]);

    // Bottleneck should be downsampled from last stage
    assert_eq!(bottleneck.dims(), [1, 64, 2, 8, 8]);
}

#[test]
fn test_encoder_presets() {
    let reg_config = SSMMorphEncoderConfig::for_registration();
    assert_eq!(reg_config.num_stages, 4);
    assert_eq!(reg_config.base_channels, 32);

    let lightweight_config = SSMMorphEncoderConfig::lightweight();
    assert_eq!(lightweight_config.num_stages, 3);
    assert_eq!(lightweight_config.base_channels, 16);

    let hq_config = SSMMorphEncoderConfig::high_quality();
    assert_eq!(hq_config.num_stages, 4);
    assert_eq!(hq_config.base_channels, 48);
    assert!(hq_config.use_drop_path);
}
