//! Shared decode-dimension bounds for untrusted codec input.
//!
//! Image codecs read `width`/`height` from header markers (JPEG SOF, JPEG-LS
//! SOF55, JPEG 2000 SIZ) — typically 16-bit fields, so a hostile or corrupt
//! header can declare up to 65535×65535 ≈ 4.3 billion pixels. Decoders size
//! per-pixel working and output buffers from those dimensions, so an unbounded
//! product turns a tiny file into a multi-gigabyte allocation (and, where the
//! decode loop runs once per declared pixel, billions of iterations).
//!
//! This is the single source of truth for decode-size caps; every decoder
//! validates declared dimensions through [`checked_pixel_count`] before
//! allocating. Multi-component decoders additionally validate their total
//! sample count through [`checked_sample_count`].

use anyhow::{anyhow, Result};

/// Upper bound on a decoded image's pixel count (`width × height`).
///
/// 256 Mi pixels ≈ a 1 GiB `i32`/`u8`-per-pixel working buffer — generous for
/// realistic medical frames while rejecting absurd headers. The bound is on
/// declared dimensions rather than encoded byte length because entropy-coded
/// run/repeat modes can expand a handful of bits into an arbitrarily long pixel
/// run, so the decoded size cannot be inferred from the input length.
pub(crate) const MAX_DECODED_PIXELS: usize = 1 << 28;

/// Validate codec-declared image dimensions, returning the checked pixel count.
///
/// # Errors
/// Returns an error when `width * height` overflows `usize` or exceeds
/// [`MAX_DECODED_PIXELS`].
pub(crate) fn checked_pixel_count(width: usize, height: usize) -> Result<usize> {
    width
        .checked_mul(height)
        .filter(|&n| n <= MAX_DECODED_PIXELS)
        .ok_or_else(|| {
            anyhow!("image {width}x{height} exceeds the {MAX_DECODED_PIXELS}-pixel decode limit")
        })
}

/// Validate the total sample count for a decoded multi-component frame.
///
/// The sample cap equals the pixel cap so increasing the component count
/// cannot amplify a valid pixel grid into an unbounded output allocation.
///
/// # Errors
/// Returns an error when `pixels * samples_per_pixel` overflows `usize`, is
/// zero, or exceeds [`MAX_DECODED_PIXELS`].
pub(crate) fn checked_sample_count(pixels: usize, samples_per_pixel: usize) -> Result<usize> {
    pixels
        .checked_mul(samples_per_pixel)
        .filter(|&n| n > 0 && n <= MAX_DECODED_PIXELS)
        .ok_or_else(|| {
            anyhow!(
                "frame with {pixels} pixels and {samples_per_pixel} samples per pixel exceeds \
                 the {MAX_DECODED_PIXELS}-sample decode limit"
            )
        })
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]
    use super::*;

    #[test]
    fn accepts_realistic_dimensions() {
        assert_eq!(checked_pixel_count(512, 512).unwrap(), 262_144);
        assert_eq!(
            checked_pixel_count(16_384, 16_384).unwrap(),
            MAX_DECODED_PIXELS
        );
    }

    #[test]
    fn rejects_oversized_dimensions() {
        let err = checked_pixel_count(65_535, 65_535).expect_err("must reject");
        assert!(err.to_string().contains("decode limit"), "got: {err}");
    }

    #[test]
    fn rejects_overflowing_product() {
        let err = checked_pixel_count(usize::MAX, 2).expect_err("must reject overflow");
        assert!(err.to_string().contains("decode limit"), "got: {err}");
    }

    #[test]
    fn rejects_component_amplified_sample_count() {
        let err = checked_sample_count(MAX_DECODED_PIXELS, 3).expect_err("must reject");
        assert!(
            err.to_string().contains("sample decode limit"),
            "got: {err}"
        );
    }

    #[test]
    fn accepts_bounded_multicomponent_sample_count() {
        assert_eq!(checked_sample_count(512 * 512, 3).unwrap(), 786_432);
    }
}
