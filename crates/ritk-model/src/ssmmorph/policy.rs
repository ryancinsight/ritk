//! Shared policy enums for the SSMMorph module.
//!
//! Enums used across multiple sub-modules are defined here and re-exported
//! from `ssmmorph::mod` to provide a single canonical import path.

/// Whether cross-scan operates on 2D (planar) or 3D (volumetric) input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum ScanDimensionality {
    /// 2D scanning over \[B, C, H, W\] tensors using 4 directions.
    Scan2d,
    /// 3D scanning over \[B, C, D, H, W\] tensors using 6 directions.
    #[default]
    Scan3d,
}
