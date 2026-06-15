//! YCbCr ↔ RGB color space conversion for JPEG baseline decode.
//!
//! JFIF §6 specifies ITU-R BT.601 YCbCr:
//!   R = Y                    + 1.402   · (Cr − 128)
//!   G = Y − 0.344136 · (Cb − 128) − 0.714136 · (Cr − 128)
//!   B = Y + 1.772   · (Cb − 128)
//!
//! Integer arithmetic uses signed 16.8 fixed-point (multiply by 256, round,
//! shift) to avoid floating-point on each pixel.

use crate::jpeg::constants::{
    CB_B_COEFF, CB_G_COEFF, CR_G_COEFF, CR_R_COEFF, FIXED_SHIFT, YCBCR_BIAS,
};

/// Convert a YCbCr triple to RGB using JFIF BT.601 fixed-point coefficients.
///
/// All inputs and outputs are in [0, 255].
#[inline]
pub(crate) fn ycbcr_to_rgb(y: i32, cb: i32, cr: i32) -> (u8, u8, u8) {
    let cb_bias = cb - YCBCR_BIAS;
    let cr_bias = cr - YCBCR_BIAS;
    // Fixed-point coefficients × 256 (8 fractional bits):
    //   1.402   × 256 = 358.912 → CR_R_COEFF
    //   0.344136× 256 =  88.099 →  CB_G_COEFF
    //   0.714136× 256 = 182.819 → CR_G_COEFF
    //   1.772   × 256 = 453.632 → CB_B_COEFF
    let r = y + ((CR_R_COEFF * cr_bias + YCBCR_BIAS) >> FIXED_SHIFT);
    let g = y - ((CB_G_COEFF * cb_bias + CR_G_COEFF * cr_bias + YCBCR_BIAS) >> FIXED_SHIFT);
    let b = y + ((CB_B_COEFF * cb_bias + YCBCR_BIAS) >> FIXED_SHIFT);
    (
        r.clamp(0, 255) as u8,
        g.clamp(0, 255) as u8,
        b.clamp(0, 255) as u8,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg::constants::YCBCR_BIAS;

    /// White (Y=255, Cb=128, Cr=128) → RGB (255, 255, 255).
    #[test]
    fn ycbcr_white_maps_to_white() {
        let (r, g, b) = ycbcr_to_rgb(255, YCBCR_BIAS, YCBCR_BIAS);
        assert_eq!(r, 255);
        assert_eq!(g, 255);
        assert_eq!(b, 255);
    }

    /// Black (Y=0, Cb=128, Cr=128) → RGB (0, 0, 0).
    #[test]
    fn ycbcr_black_maps_to_black() {
        let (r, g, b) = ycbcr_to_rgb(0, YCBCR_BIAS, YCBCR_BIAS);
        assert_eq!(r, 0);
        assert_eq!(g, 0);
        assert_eq!(b, 0);
    }

    /// Mid-gray (Y=128, Cb=128, Cr=128) → RGB all near 128.
    #[test]
    fn ycbcr_midgray_maps_to_midgray() {
        let (r, g, b) = ycbcr_to_rgb(YCBCR_BIAS, YCBCR_BIAS, YCBCR_BIAS);
        assert!(r.abs_diff(YCBCR_BIAS as u8) <= 1);
        assert!(g.abs_diff(YCBCR_BIAS as u8) <= 1);
        assert!(b.abs_diff(YCBCR_BIAS as u8) <= 1);
    }

    /// Pure red in YCbCr: Y=76, Cb=85, Cr=255.
    /// Expected: R≈255, G≈0, B≈0 (within ±4 for integer rounding).
    #[test]
    fn ycbcr_red_maps_to_red() {
        let (r, g, b) = ycbcr_to_rgb(76, 85, 255);
        assert!(r > 240, "R should be near 255, got {r}");
        assert!(g < 20, "G should be near 0, got {g}");
        assert!(b < 20, "B should be near 0, got {b}");
    }
}
