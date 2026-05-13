//! YCbCr ↔ RGB color space conversion for JPEG baseline decode.
//!
//! JFIF §6 specifies ITU-R BT.601 YCbCr:
//!   R = Y                    + 1.402   · (Cr − 128)
//!   G = Y − 0.344136 · (Cb − 128) − 0.714136 · (Cr − 128)
//!   B = Y + 1.772   · (Cb − 128)
//!
//! Integer arithmetic uses signed 16.8 fixed-point (multiply by 256, round,
//! shift) to avoid floating-point on each pixel.

/// Convert a YCbCr triple to RGB using JFIF BT.601 fixed-point coefficients.
///
/// All inputs and outputs are in [0, 255].
#[inline]
pub(crate) fn ycbcr_to_rgb(y: i32, cb: i32, cr: i32) -> (u8, u8, u8) {
    let cb_bias = cb - 128;
    let cr_bias = cr - 128;
    // Fixed-point coefficients × 256 (8 fractional bits):
    //   1.402   × 256 = 358.912 → 359
    //   0.344136× 256 =  88.099 →  88
    //   0.714136× 256 = 182.819 → 183
    //   1.772   × 256 = 453.632 → 454
    let r = y + ((359 * cr_bias + 128) >> 8);
    let g = y - ((88 * cb_bias + 183 * cr_bias + 128) >> 8);
    let b = y + ((454 * cb_bias + 128) >> 8);
    (
        r.clamp(0, 255) as u8,
        g.clamp(0, 255) as u8,
        b.clamp(0, 255) as u8,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// White (Y=255, Cb=128, Cr=128) → RGB (255, 255, 255).
    #[test]
    fn ycbcr_white_maps_to_white() {
        let (r, g, b) = ycbcr_to_rgb(255, 128, 128);
        assert_eq!(r, 255);
        assert_eq!(g, 255);
        assert_eq!(b, 255);
    }

    /// Black (Y=0, Cb=128, Cr=128) → RGB (0, 0, 0).
    #[test]
    fn ycbcr_black_maps_to_black() {
        let (r, g, b) = ycbcr_to_rgb(0, 128, 128);
        assert_eq!(r, 0);
        assert_eq!(g, 0);
        assert_eq!(b, 0);
    }

    /// Mid-gray (Y=128, Cb=128, Cr=128) → RGB all near 128.
    #[test]
    fn ycbcr_midgray_maps_to_midgray() {
        let (r, g, b) = ycbcr_to_rgb(128, 128, 128);
        assert!(r.abs_diff(128) <= 1);
        assert!(g.abs_diff(128) <= 1);
        assert!(b.abs_diff(128) <= 1);
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
