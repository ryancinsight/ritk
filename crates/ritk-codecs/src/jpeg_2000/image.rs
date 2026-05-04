//! Safe pixel extraction from a decoded OpenJPEG `opj_image_t`.
//!
//! # Specification
//! A decoded JPEG 2000 frame is stored in `opj_image_t` as an array of
//! `numcomps` component planes.  Each component plane `comps[i]` has:
//! - `data: *mut OPJ_INT32` — row-major sample array of `w × h` signed integers.
//! - `prec: OPJ_UINT32` — bit depth (e.g. 8, 12, 16).
//! - `sgnd: OPJ_UINT32` — 1 if samples are signed, 0 if unsigned.
//!
//! Rescaling to DICOM-linear floating-point values follows DICOM PS3.3 §C.7.6.3.1:
//! ```text
//!   output = stored_integer × rescale_slope + rescale_intercept
//! ```
//! OpenJPEG stores decoded samples as signed integers (`OPJ_INT32`).  For
//! unsigned components (`sgnd=0`) values lie in [0, 2^prec − 1]; for signed
//! components (`sgnd=1`) they lie in [−2^(prec−1), 2^(prec−1) − 1].  In both
//! cases the raw integer is passed directly to the rescale transform, identical
//! to the semantics of `decode_native_pixel_bytes_unchecked` for uncompressed
//! frames.
//! For multi-component images (RGB photometric) each component is mapped
//! independently — the caller is responsible for further compositing.
//!
//! # Safety
//! All pointer dereferences are gated on explicit non-null checks and
//! component-count and dimension validation performed before any raw access.

use anyhow::{bail, Result};
use openjpeg_sys as opj;

use crate::PixelLayout;

// ─── Public extraction entry point ────────────────────────────────────────────

/// Extract decoded pixel samples from a non-null `opj_image_t` into a `Vec<f32>`.
///
/// Validates that:
/// - `image` is non-null.
/// - The number of decoded components matches `layout.samples_per_pixel`.
/// - Each component has non-zero dimensions and a non-null data pointer.
/// - The total sample count matches `layout.rows × layout.cols × layout.samples_per_pixel`.
///
/// # Safety
/// `image` must be a fully decoded, non-null pointer returned by
/// `opj_read_header` + `opj_decode` with `opj_end_decompress` called.
/// The caller retains ownership of `image` and must call `opj_image_destroy`.
pub(super) unsafe fn extract_pixels(
    image:  *mut opj::opj_image_t,
    layout: &PixelLayout,
) -> Result<Vec<f32>> {
    if image.is_null() {
        bail!("opj_image_t is null after decode");
    }

    // SAFETY: image is confirmed non-null above.
    let img = &*image;

    let expected_comps = layout.samples_per_pixel as u32;
    if img.numcomps == 0 || img.numcomps != expected_comps {
        bail!(
            "JPEG 2000 numcomps={} does not match PixelLayout samples_per_pixel={}",
            img.numcomps,
            expected_comps
        );
    }

    let expected_pixels = layout.rows * layout.cols;
    let n_comps = img.numcomps as usize;
    let mut out = Vec::with_capacity(expected_pixels * n_comps);

    for ci in 0..n_comps {
        // SAFETY: numcomps bounds ci; comps is a valid array of numcomps entries
        // allocated by OpenJPEG.
        let comp = &*img.comps.add(ci);

        if comp.data.is_null() {
            bail!("JPEG 2000 component {} has null data pointer", ci);
        }
        let w = comp.w as usize;
        let h = comp.h as usize;
        if w == 0 || h == 0 {
            bail!("JPEG 2000 component {} has zero dimensions {}×{}", ci, w, h);
        }
        if w * h != expected_pixels {
            bail!(
                "JPEG 2000 component {} decoded size {}×{}={} does not match layout {}×{}={}",
                ci, w, h, w * h, layout.cols, layout.rows, expected_pixels
            );
        }

        let n_samples = w * h;
        // SAFETY: `comp.data` points to `w * h` valid OPJ_INT32 values after decode.
        let raw_slice = std::slice::from_raw_parts(comp.data, n_samples);

        for &raw in raw_slice {
            // Apply DICOM modality LUT: output = stored_integer × slope + intercept
            // (PS3.3 §C.7.6.3.1).  No [0,1] normalisation — slope and intercept
            // operate on the raw integer sample value, identical to the semantics
            // of decode_native_pixel_bytes_unchecked for uncompressed frames.
            let hu = (raw as f64 * layout.rescale_slope as f64
                     + layout.rescale_intercept as f64) as f32;
            out.push(hu);
        }
    }

    Ok(out)
}
