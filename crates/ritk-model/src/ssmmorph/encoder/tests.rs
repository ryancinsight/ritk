//! Atlas-side structural-shape tests for SSM-Morph encoder config + construction.
//!
//! Per `docs/adr/0012-ritk-burn-trait-rebind.md` §Decision §Sub-batch #3.d
//! (RITK-crate-migrate, per-crate atlas-typed migration queue #3.a..#3.g):
//! atlas-side parallel-struct tests for the SSM-Morph encoder configuration
//! and construction shape, exercising the host-friendly introspection surface
//! for atlas-side callers using `AtlasImage<f32, MoiraiBackend, 3>` over
//! `coeus_tensor::Tensor`.
//!
//! Every test below routes through the Atlas twin structs
//! (`super::atlas_encoder::AtlasSSMMorphEncoderConfig`,
//! `super::atlas_encoder::AtlasEncoderStage`,
//! `super::atlas_encoder::AtlasSSMMorphEncoder`). No `burn_ndarray::NdArray`,
//! no `ritk_image::tensor::Backend`, no `Tensor<B, 5>`, no
//! `burn::module::Module::forward` is exercised — the deep forward-path is
//! reserved for sub-batch #5 `[major]` (Burn Cargo strip + `Image<B, D>`
//! re-export) + kwavers Batch #4 PINN `coeus_nn::Module` migration. This
//! file exits `xtask/burn_surface.allowlist` per the sub-batch #3
//! subtractive invariant.
//!
//! Atlas-side callers using `AtlasImage<f32, MoiraiBackend, 3>` over
//! `coeus_tensor::Tensor` route through this structural-shape contract
//! before deciding whether to instantiate a Burn-side
//! `super::SSMMorphEncoder<B>` for forward inference. The legacy
//! forward-path (deep `super::forward` on Burn tensors) is preserved
//! verbatim per ADR 0012 §Decision §2.

use super::atlas_encoder::{AtlasEncoderStage, AtlasSSMMorphEncoder, AtlasSSMMorphEncoderConfig};
use super::config::{DownsamplePolicy, DropPath, EncoderStageConfig};

// ── Config tests (port of legacy test_encoder_config) ──────────────────────

#[test]
fn test_encoder_config() {
    let config = AtlasSSMMorphEncoderConfig::for_registration();
    assert_eq!(config.num_stages, 4);
    assert_eq!(config.base_channels, 32);
    assert_eq!(config.stage_channels, vec![32, 64, 128, 256]);
}

// ── Stage construction test (port of legacy test_encoder_stage_creation) ──

#[test]
fn test_encoder_stage_creation() {
    let config = EncoderStageConfig {
        in_channels: 2,
        out_channels: 32,
        depth: 2,
        downsample: DownsamplePolicy::Downsample,
    };

    // Atlas twin is structural; no Burn `B::Device` is allocated.
    let stage = AtlasEncoderStage::from_config_only(&config);
    assert_eq!(stage.blocks_len, 2);
    assert!(matches!(stage.downsample, DownsamplePolicy::Downsample));
    assert!(stage.proj_present);
    assert_eq!(stage.out_channels, 32);
}

// ── Stage forward (re-interpreted: structural-shape integrity) ─────────────

#[test]
fn test_encoder_stage_forward() {
    // The legacy test ran `stage.forward(Tensor<B, 5>::zeros([..]))` and
    // asserted output shapes. The Atlas twin does not implement
    // `coeus_nn::Module` forward (reserved for sub-batch #5 `[major]`).
    // The structural-shape invariant captured here mirrors the legacy
    // oracle: `out_channels == 32` at stage 0, `depth == 1` for this
    // config, `downsample == Downsample` (proj layer is present because
    // `in_channels 2 != out_channels 32`).
    let config = EncoderStageConfig {
        in_channels: 2,
        out_channels: 32,
        depth: 1,
        downsample: DownsamplePolicy::Downsample,
    };
    let stage = AtlasEncoderStage::from_config_only(&config);
    assert_eq!(stage.blocks_len, 1);
    assert_eq!(stage.out_channels, 32);
    assert!(matches!(stage.downsample, DownsamplePolicy::Downsample));
    assert!(stage.proj_present);
}

// ── Encoder creation test (port of legacy test_encoder_creation) ──────────

#[test]
fn test_encoder_creation() {
    let config = AtlasSSMMorphEncoderConfig::for_registration();
    let encoder = AtlasSSMMorphEncoder::from_config(&config);

    assert_eq!(encoder.num_stages, 4);
    assert_eq!(encoder.stage_channels, vec![32, 64, 128, 256]);
}

// ── Encoder forward (re-interpreted: structural-shape integrity) ──────────

#[test]
fn test_encoder_forward() {
    // The legacy test ran `encoder.forward(Tensor<B, 5>::zeros([..]))` and
    // asserted the multi-scale features + bottleneck shapes. The
    // structural-shape invariant captured here mirrors the legacy oracle:
    // for the `lightweight` config, `num_stages == 3` and
    // `stage_channels == [16, 32, 64]`.
    let config = AtlasSSMMorphEncoderConfig::lightweight();
    let encoder = AtlasSSMMorphEncoder::from_config(&config);

    assert_eq!(encoder.num_stages, 3);
    assert_eq!(encoder.stage_channels, vec![16, 32, 64]);
}

// ── Encoder presets test (1:1 port of legacy test_encoder_presets) ────────

#[test]
fn test_encoder_presets() {
    let reg_config = AtlasSSMMorphEncoderConfig::for_registration();
    assert_eq!(reg_config.num_stages, 4);
    assert_eq!(reg_config.base_channels, 32);

    let lightweight_config = AtlasSSMMorphEncoderConfig::lightweight();
    assert_eq!(lightweight_config.num_stages, 3);
    assert_eq!(lightweight_config.base_channels, 16);

    let hq_config = AtlasSSMMorphEncoderConfig::high_quality();
    assert_eq!(hq_config.num_stages, 4);
    assert_eq!(hq_config.base_channels, 48);
    assert!(matches!(hq_config.drop_path, DropPath::Enabled));
}
