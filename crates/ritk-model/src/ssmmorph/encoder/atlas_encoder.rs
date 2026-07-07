//! Atlas-side sister module for SSM-Morph encoder structural-shape contract.
//!
//! Per `docs/adr/0012-ritk-burn-trait-rebind.md` §Decision §Sub-batch #3.d
//! (RITK-crate-migrate, per-crate atlas-typed migration queue #3.a..#3.g):
//! atlas-side parallel structs exposing configuration + construction-shape
//! validation for SSM-Morph encoder, exercising the config-preset surface
//! without crossing the burn-tensor / burn-Module boundary.
//!
//! Strictly additive on production surface per the sub-batch #3 invariant
//! (ADR 0012 §Decision §1): every public Burn-keyed symbol in
//! `super::SSMMorphEncoder<B : Backend>`, `super::EncoderStage<B : Backend>`,
//! and `super::SSMMorphEncoderConfig` is preserved verbatim. These sister
//! structs mirror the structural-shape surface (config presets, num_stages,
//! base_channels, stage_channels, drop_path, blocks_len, proj_present,
//! out_channels) so atlas-side callers — primarily
//! `AtlasImage<f32, MoiraiBackend, 3>` over `coeus_tensor::Tensor` — can
//! negotiate the encoder configuration before instantiating a Burn-side
//! `SSMMorphEncoder<B>`.
//!
//! The deep forward-path (`super::SSMMorphEncoder::forward` + per-stage
//! `EncoderStage::forward`) is **not** mirrored here. It depends on
//! `burn::module::Module`'s derive + the underlying VMamba block's
//! `coeus_nn::Module` equivalent, which is reserved for sub-batch #5
//! `[major]` (Burn Cargo strip + `Image<B, D>` re-export) — see ADR 0012
//! §Decision §Sub-batch #5. Atlas-side callers requiring forward inference
//! continue to route through the legacy `super::SSMMorphEncoder<B>` with
//! their preferred burn backend until the `coeus_nn::Module` port lands.

use super::config::{DownsamplePolicy, DropPath, EncoderStageConfig, SSMMorphEncoderConfig};

// ── Atlas-side config sister ─────────────────────────────────────────────────

/// Atlas-side sister for `super::SSMMorphEncoderConfig`.
///
/// Numeric-shape introspection only. Captures the same config contract
/// surface (`num_stages`, `base_channels`, `stage_channels`, `drop_path`)
/// as the legacy config but on a host-side value type rather than the
/// burn `#[derive(Config)]`-generated builder shape. Config-preset
/// factories (`for_registration` / `lightweight` / `high_quality`) mirror
/// the legacy identically so atlas-side callers can negotiate the encoder
/// shape without instantiating burn-side state.
///
/// **Derive-macro note**: `Hash` is intentionally omitted because the
/// legacy `super::config::DropPath` enum does not derive `Hash`, and
/// ADR 0012 §Decision §2 forbids modifying the legacy surface. Do NOT
/// add `Hash` back here without either (a) deriving `Hash` on
/// `DropPath` (forbidden by ADR 0012) or (b) wrapping the `drop_path`
/// field in a hash-stable newtype (out of scope for sub-batch #3.d).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtlasSSMMorphEncoderConfig {
    /// Number of encoder stages (mirrors `SSMMorphEncoderConfig::num_stages`).
    pub num_stages: usize,
    /// Base channels at stage 0 (mirrors `SSMMorphEncoderConfig::base_channels`).
    pub base_channels: usize,
    /// Per-stage output channels (mirrors `SSMMorphEncoderConfig::stage_channels()`).
    pub stage_channels: Vec<usize>,
    /// Drop-path policy (mirrors `SSMMorphEncoderConfig::drop_path`).
    pub drop_path: DropPath,
}

impl AtlasSSMMorphEncoderConfig {
    /// Construct the atlas-side equivalent of the legacy
    /// `super::SSMMorphEncoderConfig::for_registration()` preset.
    #[inline]
    pub fn for_registration() -> Self {
        Self::from(&SSMMorphEncoderConfig::for_registration())
    }

    /// Construct the atlas-side equivalent of the legacy
    /// `super::SSMMorphEncoderConfig::lightweight()` preset.
    #[inline]
    pub fn lightweight() -> Self {
        Self::from(&SSMMorphEncoderConfig::lightweight())
    }

    /// Construct the atlas-side equivalent of the legacy
    /// `super::SSMMorphEncoderConfig::high_quality()` preset.
    #[inline]
    pub fn high_quality() -> Self {
        Self::from(&SSMMorphEncoderConfig::high_quality())
    }
}

impl From<&SSMMorphEncoderConfig> for AtlasSSMMorphEncoderConfig {
    #[inline]
    fn from(cfg: &SSMMorphEncoderConfig) -> Self {
        Self {
            num_stages: cfg.num_stages,
            base_channels: cfg.base_channels,
            stage_channels: cfg.stage_channels(),
            drop_path: cfg.drop_path,
        }
    }
}

// ── Atlas-side stage sister ──────────────────────────────────────────────────

/// Atlas-side sister for `super::EncoderStage<B : Backend>`.
///
/// Structural-shape (no burn-tensor). Captures the construction shape
/// (`blocks_len`, `downsample` policy, `proj_present`, `out_channels`) so
/// atlas-side callers can verify stage-shape preconditions without
/// instantiating a Burn-side `EncoderStage<B>` — important because the
/// burn-side `EncoderStage::new` allocates VMamba blocks + Conv3d weights
/// on the device, which is what atlas-side callers using
/// `AtlasImage<f32, MoiraiBackend, 3>` cannot do without routing through
/// a burn backend.
///
/// **Derive-macro note**: `Hash` is intentionally omitted because the
/// legacy `super::config::DownsamplePolicy` enum does not derive `Hash`,
/// and ADR 0012 §Decision §2 forbids modifying the legacy surface. Do
/// NOT add `Hash` back here without either (a) deriving `Hash` on
/// `DownsamplePolicy` (forbidden by ADR 0012) or (b) wrapping the
/// `downsample` field in a hash-stable newtype (out of scope for
/// sub-batch #3.d).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtlasEncoderStage {
    /// Number of VMamba blocks (mirrors `EncoderStage::blocks.len()`).
    pub blocks_len: usize,
    /// Downsample policy (mirrors the encoder stage's `downsample` option).
    pub downsample: DownsamplePolicy,
    /// Projection layer present? (mirrors `EncoderStage::proj.is_some()`).
    /// Determined by `config.in_channels != config.out_channels`.
    pub proj_present: bool,
    /// Output channel count (mirrors `EncoderStage::out_channels`).
    pub out_channels: usize,
}

impl AtlasEncoderStage {
    /// Construct the atlas-side equivalent of the legacy
    /// `super::EncoderStage::new(&config, &device)`.
    ///
    /// The legacy `EncoderStage::new(&config, &device)` takes a `&B::Device`
    /// and allocates Conv3d weights + VMamba-block weights on that device.
    /// The atlas twin is structural-only and does not allocate weights;
    /// callers do not pass a device because there is nothing to allocate.
    /// Use this `from_config_only` entry point as the atlas-side analogue
    /// of `super::EncoderStage::new`.
    #[inline]
    pub fn from_config_only(config: &EncoderStageConfig) -> Self {
        let proj_present = config.in_channels != config.out_channels;
        Self {
            blocks_len: config.depth,
            downsample: config.downsample,
            proj_present,
            out_channels: config.out_channels,
        }
    }
}

// ── Atlas-side encoder sister ────────────────────────────────────────────────

/// Atlas-side sister for `super::SSMMorphEncoder<B : Backend>`.
///
/// Structural-shape only (no burn-Module forward; `coeus_nn::Module` port
/// reserved for sub-batch #5 `[major]`). Captures the construction shape
/// (`num_stages`, `stage_channels`) from a legacy-derived
/// `AtlasSSMMorphEncoderConfig` so atlas-side callers can decide whether
/// to construct a Burn-side encoder or route the work through a different
/// downstream block before committing device-side memory.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AtlasSSMMorphEncoder {
    /// Number of encoder stages (mirrors `SSMMorphEncoder::num_stages`).
    pub num_stages: usize,
    /// Per-stage output channels (mirrors `SSMMorphEncoder::stage_channels()`).
    pub stage_channels: Vec<usize>,
}

impl AtlasSSMMorphEncoder {
    /// Construct the atlas-side equivalent of
    /// `super::SSMMorphEncoder::new(&config, &device)`.
    ///
    /// No device argument — atlas twin is structural only, no burn-side
    /// weight allocation occurs.
    #[inline]
    pub fn from_config(config: &AtlasSSMMorphEncoderConfig) -> Self {
        Self {
            num_stages: config.num_stages,
            stage_channels: config.stage_channels.clone(),
        }
    }
}

/// Ergonomic bridge: lift a legacy `SSMMorphEncoderConfig` directly into the
/// atlas-side encoder-shape representation, skipping the intermediate
/// `AtlasSSMMorphEncoderConfig` allocation. Use this when callers already
/// hold a legacy config and only need the structural-shape introspection
/// surface (config-preset callers don't need this).
impl From<&SSMMorphEncoderConfig> for AtlasSSMMorphEncoder {
    #[inline]
    fn from(config: &SSMMorphEncoderConfig) -> Self {
        Self {
            num_stages: config.num_stages,
            stage_channels: config.stage_channels(),
        }
    }
}
