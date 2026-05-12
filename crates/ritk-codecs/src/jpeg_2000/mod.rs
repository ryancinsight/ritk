//! Native JPEG 2000 (ISO 15444-1) decoder for DICOM encapsulated frames.
//!
//! # Architecture
//! - [`stream`]: OpenJPEG in-memory read stream via custom I/O callbacks.
//! - [`image`]:  Safe pixel extraction from a decoded `opj_image_t`.
//! - This module: public API, marker constants, codec/stream lifecycle.
//!
//! # Specification (ISO 15444-1 / DICOM PS3.5)
//! DICOM JPEG 2000 encapsulates a raw J2K codestream (not a JP2 file wrapper):
//! - Transfer Syntax 1.2.840.10008.1.2.4.90: JPEG 2000 Lossless Only (J2K, NEAR=0).
//! - Transfer Syntax 1.2.840.10008.1.2.4.91: JPEG 2000 (J2K, lossy or lossless).
//!
//! A valid J2K codestream begins with SOC (0xFF4F) and ends with EOC (0xFFD9).
//! OpenJPEG 2.5 (`openjpeg-sys 1.0.12`) is used as the conformant ISO 15444-1
//! decode engine.  All unsafe code is isolated in [`stream`] and [`image`].
//!
//! # Decode path
//! 1. Validate SOC marker at byte 0.
//! 2. Create in-memory stream (`J2kMemStream`).
//! 3. `opj_create_decompress(OPJ_CODEC_J2K)`.
//! 4. `opj_set_default_decoder_parameters`, `opj_setup_decoder`.
//! 5. `opj_read_header` → `opj_decode` → `opj_end_decompress`.
//! 6. Extract pixels via `image::extract_pixels`.
//! 7. Destroy codec, stream, image in reverse order.

mod image;
mod stream;

use anyhow::{bail, Context, Result};
use openjpeg_sys as opj;

use crate::PixelLayout;
use stream::J2kMemStream;

// ─── JPEG 2000 / J2K Markers (ISO 15444-1 §A.3) ─────────────────────────────

/// J2K Start of Codestream marker (ISO 15444-1 §A.3): bytes `0xFF 0x4F`.
pub(crate) const SOC: u16 = 0xFF4F;

/// JPEG / JFIF Start of Image marker, shared with JPEG-LS (0xFFD8).
/// Presence of SOI instead of SOC indicates JP2 file or JPEG variant, not a
/// bare J2K codestream.
#[allow(dead_code)]
pub(crate) const SOI: u16 = 0xFFD8;

// ─── Public API ───────────────────────────────────────────────────────────────

/// Decode a DICOM-encapsulated JPEG 2000 J2K codestream fragment.
///
/// # Arguments
/// - `fragment`: raw bytes of the encapsulated pixel data item.
/// - `layout`:   pixel geometry and DICOM rescale parameters.
///
/// # Errors
/// Returns an error if:
/// - `fragment` does not begin with the SOC marker (0xFF4F).
/// - OpenJPEG fails to parse the codestream header or decode the image.
/// - Decoded dimensions do not match `layout`.
pub fn decode_jpeg2000_fragment(fragment: &[u8], layout: PixelLayout) -> Result<Vec<f32>> {
    if !is_jpeg2000_codestream(fragment) {
        bail!(
            "JPEG 2000 fragment does not begin with SOC marker 0xFF4F \
             (first 2 bytes: {:02X?})",
            &fragment[..fragment.len().min(2)]
        );
    }

    // ── Codec and stream setup ──────────────────────────────────────────────
    let mut mem_stream = J2kMemStream::new(fragment.to_vec());

    // SAFETY: All unsafe operations follow the documented lifecycle:
    //   1. codec is created before stream creation.
    //   2. stream is created and live for the full decode call.
    //   3. decode sequence: read_header → decode → end_decompress.
    //   4. resources destroyed in reverse order: image, stream, codec.
    unsafe {
        // ── Create codec ───────────────────────────────────────────────────
        let codec = opj::opj_create_decompress(opj::CODEC_FORMAT::OPJ_CODEC_J2K);
        if codec.is_null() {
            bail!("opj_create_decompress returned null");
        }

        // Silence OpenJPEG's stderr output; errors are captured via Rust return.
        // SAFETY: handler function pointers are None — OpenJPEG accepts null handlers
        // as "suppress output" in 2.5.x.
        opj::opj_set_info_handler(codec, None, std::ptr::null_mut());
        opj::opj_set_warning_handler(codec, None, std::ptr::null_mut());
        opj::opj_set_error_handler(codec, None, std::ptr::null_mut());

        // ── Decoder parameters ─────────────────────────────────────────────
        let mut params: opj::opj_dparameters_t = std::mem::zeroed();
        opj::opj_set_default_decoder_parameters(&mut params);
        // cp_reduce = 0: decode at full resolution.
        // cp_layer  = 0: decode all quality layers.
        params.cp_reduce = 0;
        params.cp_layer = 0;

        if opj::opj_setup_decoder(codec, &mut params) != opj::OPJ_TRUE as opj::OPJ_BOOL {
            opj::opj_destroy_codec(codec);
            bail!("opj_setup_decoder failed");
        }

        // ── In-memory stream ───────────────────────────────────────────────
        let stream = mem_stream.create_opj_stream();

        // ── Read header ────────────────────────────────────────────────────
        let mut image: *mut opj::opj_image_t = std::ptr::null_mut();
        let header_ok = opj::opj_read_header(stream, codec, &mut image);

        if header_ok != opj::OPJ_TRUE as opj::OPJ_BOOL {
            opj::opj_stream_destroy(stream);
            opj::opj_destroy_codec(codec);
            bail!("opj_read_header failed for JPEG 2000 fragment");
        }

        // ── Decode ─────────────────────────────────────────────────────────
        let decode_ok = opj::opj_decode(codec, stream, image);

        if decode_ok != opj::OPJ_TRUE as opj::OPJ_BOOL {
            if !image.is_null() {
                opj::opj_image_destroy(image);
            }
            opj::opj_stream_destroy(stream);
            opj::opj_destroy_codec(codec);
            bail!("opj_decode failed for JPEG 2000 fragment");
        }

        let end_ok = opj::opj_end_decompress(codec, stream);

        // ── Resource cleanup ───────────────────────────────────────────────
        opj::opj_stream_destroy(stream);
        opj::opj_destroy_codec(codec);

        if end_ok != opj::OPJ_TRUE as opj::OPJ_BOOL {
            if !image.is_null() {
                opj::opj_image_destroy(image);
            }
            bail!("opj_end_decompress failed for JPEG 2000 fragment");
        }

        // ── Pixel extraction ───────────────────────────────────────────────
        let pixels = image::extract_pixels(image, &layout)
            .with_context(|| "JPEG 2000 pixel extraction failed")?;
        opj::opj_image_destroy(image);
        Ok(pixels)
    }
}

/// Returns `true` if `fragment` begins with the J2K SOC marker (`0xFF 0x4F`).
///
/// A bare DICOM JPEG 2000 codestream always starts with SOC (ISO 15444-1 §A.3).
/// JP2 file wrappers begin with the 12-byte JP2 Signature Box and do not appear
/// in DICOM transfer syntaxes 1.2.840.10008.1.2.4.90/91.
#[inline]
pub(crate) fn is_jpeg2000_codestream(fragment: &[u8]) -> bool {
    fragment.len() >= 2 && fragment[0] == (SOC >> 8) as u8 && fragment[1] == (SOC & 0xFF) as u8
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use openjpeg_sys as opj;

    // ─── Marker detection ─────────────────────────────────────────────────

    #[test]
    fn soc_marker_constant_matches_iso_15444_1() {
        // ISO 15444-1 §A.3.1: SOC is the 2-byte sequence 0xFF 0x4F.
        assert_eq!(SOC, 0xFF4F, "SOC must equal 0xFF4F per ISO 15444-1 §A.3.1");
        assert_eq!(SOC >> 8, 0xFF, "SOC high byte must be 0xFF (marker prefix)");
        assert_eq!(SOC & 0xFF, 0x4F, "SOC low byte must be 0x4F");
    }

    #[test]
    fn soi_constant_matches_jpeg_start_of_image() {
        // JPEG SOI = 0xFFD8 (also JPEG-LS codestream prefix).
        assert_eq!(SOI, 0xFFD8, "SOI must equal 0xFFD8");
        assert_ne!(SOI, SOC, "SOI and SOC must be distinct markers");
    }

    #[test]
    fn is_jpeg2000_codestream_detects_soc_at_byte_0() {
        let codestream = [0xFF_u8, 0x4F, 0x00, 0x00];
        assert!(
            is_jpeg2000_codestream(&codestream),
            "0xFF 0x4F prefix must be recognized as J2K SOC"
        );
    }

    #[test]
    fn is_jpeg2000_codestream_rejects_jpeg_ls_prefix() {
        // JPEG-LS starts with SOI = 0xFFD8, not SOC = 0xFF4F.
        let jpeg_ls = [0xFF_u8, 0xD8, 0xFF, 0xF7];
        assert!(
            !is_jpeg2000_codestream(&jpeg_ls),
            "JPEG-LS SOI prefix 0xFFD8 must not be recognized as J2K SOC"
        );
    }

    #[test]
    fn is_jpeg2000_codestream_rejects_rle_prefix() {
        let rle = [0x00_u8, 0x00, 0x00, 0x01];
        assert!(
            !is_jpeg2000_codestream(&rle),
            "RLE prefix must not be recognized as J2K SOC"
        );
    }

    #[test]
    fn is_jpeg2000_codestream_rejects_empty_and_single_byte() {
        assert!(
            !is_jpeg2000_codestream(&[]),
            "empty slice must return false"
        );
        assert!(
            !is_jpeg2000_codestream(&[0xFF]),
            "single byte must return false (SOC requires 2 bytes)"
        );
    }

    // ─── Error path: malformed input ──────────────────────────────────────

    #[test]
    fn decode_returns_error_for_non_soc_prefix() {
        use crate::PixelLayout;
        let layout = PixelLayout {
            rows: 2,
            cols: 2,
            samples_per_pixel: 1,
            bits_allocated: 8,
            pixel_representation: 0,
            rescale_slope: 1.0,
            rescale_intercept: 0.0,
        };
        // Feed a JPEG-LS SOI prefix — decode must reject it, not panic.
        let fragment = [0xFF_u8, 0xD8, 0xFF, 0xF7, 0x00, 0x0B];
        let result = decode_jpeg2000_fragment(&fragment, layout);
        assert!(result.is_err(), "non-SOC prefix must return Err");
        let msg = format!("{:#}", result.unwrap_err());
        assert!(
            msg.contains("SOC") || msg.contains("0xFF4F") || msg.contains("FF4F"),
            "error must mention SOC marker; got: {}",
            msg
        );
    }

    #[test]
    fn decode_returns_error_for_truncated_codestream() {
        use crate::PixelLayout;
        let layout = PixelLayout {
            rows: 4,
            cols: 4,
            samples_per_pixel: 1,
            bits_allocated: 8,
            pixel_representation: 0,
            rescale_slope: 1.0,
            rescale_intercept: 0.0,
        };
        // SOC prefix present but codestream is truncated — OpenJPEG must reject it.
        let truncated = [0xFF_u8, 0x4F, 0x00];
        let result = decode_jpeg2000_fragment(&truncated, layout);
        assert!(result.is_err(), "truncated J2K codestream must return Err");
    }

    // ─── Positive round-trip: encode + decode via OpenJPEG ───────────────

    /// Build a minimal lossless JPEG 2000 J2K codestream for a given
    /// `rows × cols` 8-bit unsigned grayscale image using OpenJPEG 2.5.
    ///
    /// The produced codestream uses:
    /// - 1 component, 8-bit unsigned, 1×1 subsampling.
    /// - 0 DWT decomposition levels (no spatial transform).
    /// - Lossless reversible coding (5/3 wavelet, NEAR=0 via lossy=0).
    /// - Single quality layer.
    /// - L-R-C-P progression order.
    ///
    /// # Safety
    /// All OpenJPEG resources are destroyed before return in both success and
    /// error paths.  The output buffer is grown by the stream write callback.
    #[cfg(test)]
    unsafe fn encode_to_j2k(pixels: &[i32], rows: u32, cols: u32) -> Vec<u8> {
        use std::ffi::c_void;

        // ── Output buffer + write-stream state ────────────────────────────
        let mut out: Vec<u8> = Vec::new();

        // Callback: append `nb_bytes` from `buffer` to `user_data: *mut Vec<u8>`.
        unsafe extern "C" fn write_fn(
            buffer: *mut c_void,
            nb_bytes: opj::OPJ_SIZE_T,
            user_data: *mut c_void,
        ) -> opj::OPJ_SIZE_T {
            let out = &mut *(user_data as *mut Vec<u8>);
            let slice = std::slice::from_raw_parts(buffer as *const u8, nb_bytes);
            out.extend_from_slice(slice);
            nb_bytes
        }

        // seek_fn for output: seek within already-written data or append zeros.
        unsafe extern "C" fn seek_out_fn(
            nb_bytes: opj::OPJ_OFF_T,
            _user_data: *mut c_void,
        ) -> opj::OPJ_BOOL {
            // For a growing output stream, seek is called at the very end to
            // record the total written size.  We accept any non-negative position.
            if nb_bytes >= 0 {
                opj::OPJ_TRUE as opj::OPJ_BOOL
            } else {
                opj::OPJ_FALSE as opj::OPJ_BOOL
            }
        }

        // skip_fn for output: advance write position (not used in practice).
        unsafe extern "C" fn skip_out_fn(
            nb_bytes: opj::OPJ_OFF_T,
            _user_data: *mut c_void,
        ) -> opj::OPJ_OFF_T {
            nb_bytes.max(0)
        }

        let write_stream = opj::opj_stream_default_create(opj::OPJ_FALSE as opj::OPJ_BOOL);
        assert!(
            !write_stream.is_null(),
            "opj_stream_default_create (write) returned null"
        );
        opj::opj_stream_set_write_function(write_stream, Some(write_fn));
        opj::opj_stream_set_skip_function(write_stream, Some(skip_out_fn));
        opj::opj_stream_set_seek_function(write_stream, Some(seek_out_fn));
        opj::opj_stream_set_user_data(write_stream, &mut out as *mut Vec<u8> as *mut c_void, None);

        // ── Create image ──────────────────────────────────────────────────
        let mut cmptparm: opj::opj_image_cmptparm_t = std::mem::zeroed();
        cmptparm.dx = 1;
        cmptparm.dy = 1;
        cmptparm.w = cols;
        cmptparm.h = rows;
        cmptparm.prec = 8;
        cmptparm.sgnd = 0;

        let image = opj::opj_image_create(1, &mut cmptparm, opj::COLOR_SPACE::OPJ_CLRSPC_GRAY);
        assert!(!image.is_null(), "opj_image_create returned null");

        (*image).x0 = 0;
        (*image).y0 = 0;
        (*image).x1 = cols;
        (*image).y1 = rows;

        // Copy pixel data into component 0.
        let comp = &mut *(*image).comps;
        let n = (rows * cols) as usize;
        for i in 0..n {
            *comp.data.add(i) = pixels[i];
        }

        // ── Encoder parameters ────────────────────────────────────────────
        let mut cparams: opj::opj_cparameters_t = std::mem::zeroed();
        opj::opj_set_default_encoder_parameters(&mut cparams);
        cparams.numresolution = 1; // 1 resolution = 0 DWT levels (no spatial transform)
        cparams.irreversible = 0; // 0 = reversible (5/3 wavelet, lossless)

        let codec = opj::opj_create_compress(opj::CODEC_FORMAT::OPJ_CODEC_J2K);
        assert!(!codec.is_null(), "opj_create_compress returned null");

        opj::opj_set_info_handler(codec, None, std::ptr::null_mut());
        opj::opj_set_warning_handler(codec, None, std::ptr::null_mut());
        opj::opj_set_error_handler(codec, None, std::ptr::null_mut());

        let setup_ok = opj::opj_setup_encoder(codec, &mut cparams, image);
        assert_eq!(
            setup_ok,
            opj::OPJ_TRUE as opj::OPJ_BOOL,
            "opj_setup_encoder failed"
        );

        let start_ok = opj::opj_start_compress(codec, image, write_stream);
        assert_eq!(
            start_ok,
            opj::OPJ_TRUE as opj::OPJ_BOOL,
            "opj_start_compress failed"
        );

        let encode_ok = opj::opj_encode(codec, write_stream);
        assert_eq!(
            encode_ok,
            opj::OPJ_TRUE as opj::OPJ_BOOL,
            "opj_encode failed"
        );

        let end_ok = opj::opj_end_compress(codec, write_stream);
        assert_eq!(
            end_ok,
            opj::OPJ_TRUE as opj::OPJ_BOOL,
            "opj_end_compress failed"
        );

        opj::opj_stream_destroy(write_stream);
        opj::opj_destroy_codec(codec);
        opj::opj_image_destroy(image);

        out
    }

    #[test]
    fn decode_jpeg2000_lossless_round_trip_4x4_uniform() {
        // Analytical ground truth: 4×4 image with uniform pixel value 128.
        // Lossless + identity rescale: output = stored × 1.0 + 0.0 = 128.0 (PS3.3 §C.7.6.3.1).
        use crate::PixelLayout;

        let rows = 4u32;
        let cols = 4u32;
        let pixel_value = 128i32;
        let pixels: Vec<i32> = vec![pixel_value; (rows * cols) as usize];

        let j2k = unsafe { encode_to_j2k(&pixels, rows, cols) };

        // Encoded codestream must begin with SOC.
        assert!(
            is_jpeg2000_codestream(&j2k),
            "openjpeg encode output must start with SOC 0xFF4F; got: {:02X?}",
            &j2k[..j2k.len().min(4)]
        );

        let layout = PixelLayout {
            rows: rows as usize,
            cols: cols as usize,
            samples_per_pixel: 1,
            bits_allocated: 8,
            pixel_representation: 0,
            rescale_slope: 1.0,
            rescale_intercept: 0.0,
        };

        let decoded = decode_jpeg2000_fragment(&j2k, layout)
            .expect("lossless JPEG 2000 round-trip must succeed");

        assert_eq!(
            decoded.len(),
            (rows * cols) as usize,
            "decoded pixel count must equal rows × cols"
        );

        // Lossless + identity rescale: every decoded value must equal the raw integer
        // (PS3.3 §C.7.6.3.1: output = stored × 1.0 + 0.0 = stored).
        let expected = pixel_value as f32;
        for (i, &v) in decoded.iter().enumerate() {
            assert_eq!(
                v, expected,
                "pixel[{i}] = {v}, expected {expected} (lossless round-trip invariant)"
            );
        }
    }

    #[test]
    fn decode_jpeg2000_lossless_round_trip_gradient_2x4() {
        // Analytical ground truth: 2×4 image with distinct values [0..7].
        // Row 0: [0, 1, 2, 3], Row 1: [4, 5, 6, 7].
        // Lossless + identity rescale: output = stored × 1.0 + 0.0 = stored.
        // (PS3.3 §C.7.6.3.1)
        use crate::PixelLayout;

        let rows = 2u32;
        let cols = 4u32;
        let pixels: Vec<i32> = (0..8).collect();

        let j2k = unsafe { encode_to_j2k(&pixels, rows, cols) };

        let layout = PixelLayout {
            rows: rows as usize,
            cols: cols as usize,
            samples_per_pixel: 1,
            bits_allocated: 8,
            pixel_representation: 0,
            rescale_slope: 1.0,
            rescale_intercept: 0.0,
        };

        let decoded =
            decode_jpeg2000_fragment(&j2k, layout).expect("gradient round-trip must succeed");

        assert_eq!(decoded.len(), 8);

        // Verify each pixel independently: decoded = raw × 1.0 + 0.0 = raw.
        for (i, (&raw, &decoded_val)) in pixels.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(
                decoded_val, raw as f32,
                "gradient pixel[{i}]: raw={raw}, decoded={decoded_val}"
            );
        }
    }

    #[test]
    fn decode_jpeg2000_lossless_rescale_applied_correctly() {
        // Verify DICOM rescale (slope, intercept) per PS3.3 §C.7.6.3.1:
        //   output = stored × slope + intercept
        // Image: 1×1 pixel, value=100 (unsigned 8-bit).
        // Rescale: slope=2.0, intercept=-1024.0.
        // Expected: 100 × 2.0 + (−1024.0) = −824.0.
        use crate::PixelLayout;

        let pixels = [100i32];
        let j2k = unsafe { encode_to_j2k(&pixels, 1, 1) };

        let layout = PixelLayout {
            rows: 1,
            cols: 1,
            samples_per_pixel: 1,
            bits_allocated: 8,
            pixel_representation: 0,
            rescale_slope: 2.0,
            rescale_intercept: -1024.0,
        };

        let decoded =
            decode_jpeg2000_fragment(&j2k, layout).expect("1×1 rescale test must succeed");

        assert_eq!(
            decoded.len(),
            1,
            "single-pixel image must yield 1 decoded value"
        );

        // 100 × 2.0 + (−1024.0) = −824.0 exactly (f32 representable).
        assert_eq!(
            decoded[0], -824.0f32,
            "rescale: decoded[0]={}, expected=-824.0 (slope=2, intercept=-1024)",
            decoded[0],
        );
    }

    #[test]
    fn openjpeg_version_is_2_5_x() {
        // Verify the embedded OpenJPEG library is the expected 2.5.x build.
        // This is an invariant: codec correctness depends on ISO 15444-1 compliance
        // in OpenJPEG 2.5, which fixed several entropy-coding correctness issues
        // present in 2.3/2.4.
        let version_str = unsafe {
            let ptr = opj::opj_version();
            assert!(!ptr.is_null(), "opj_version() returned null");
            std::ffi::CStr::from_ptr(ptr).to_str().unwrap_or("unknown")
        };
        assert!(
            version_str.starts_with("2.5"),
            "openjpeg-sys must embed OpenJPEG 2.5.x; got version: {}",
            version_str
        );
    }
}
