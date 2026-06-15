//! Shared constants for the JPEG codec modules.

/// DCT block side length (8 pixels per JPEG T.81 §A.3.1).
pub(crate) const DCT_BLOCK_DIM: usize = 8;
/// DCT block total cells (DCT_BLOCK_DIM²).
pub(crate) const DCT_BLOCK_CELLS: usize = 64;

/// YCbCr chroma bias (center of the [0, 255] chroma range per JFIF §6).
pub(crate) const YCBCR_BIAS: i32 = 128;
/// Fixed-point coefficient for Cr → R channel (1.402 × 256 ≈ 359).
pub(crate) const CR_R_COEFF: i32 = 359;
/// Fixed-point coefficient for Cb → G channel (0.344136 × 256 ≈ 88).
pub(crate) const CB_G_COEFF: i32 = 88;
/// Fixed-point coefficient for Cr → G channel (0.714136 × 256 ≈ 183).
pub(crate) const CR_G_COEFF: i32 = 183;
/// Fixed-point coefficient for Cb → B channel (1.772 × 256 ≈ 454).
pub(crate) const CB_B_COEFF: i32 = 454;
/// Shift for 8-bit fixed-point scaling (divide by 256).
pub(crate) const FIXED_SHIFT: i32 = 8;
