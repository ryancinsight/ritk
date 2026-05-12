//! Safe pixel extraction from a decoded JPEG 2000 image.
//!
//! # Specification
//! A decoded JPEG 2000 frame is represented as component planes.  Each component
//! exposes row-major `i32` decoded samples, component dimensions, precision, and
//! signedness.  The JPEG 2000 backend may internally use a port of OpenJPEG, but
//! this module depends only on the safe `jpeg2k` component API.
//!
//! Rescaling to DICOM-linear floating-point values follows DICOM PS3.3 §C.7.6.3.1:
//! ```text
//!   output = stored_integer × rescale_slope + rescale_intercept
//! ```
//! For unsigned components values lie in [0, 2^prec − 1]; for signed components
//! values lie in [−2^(prec−1), 2^(prec−1) − 1].  In both cases the raw integer is
//! passed directly to the rescale transform, identical to the semantics of
//! `decode_native_pixel_bytes_unchecked` for uncompressed frames.
//! For multi-component images (RGB photometric) each component is mapped
//! independently — the caller is responsible for further compositing.

use anyhow::{bail, Result};
use jpeg2k::Image;

use crate::PixelLayout;

// ─── Public extraction entry point ────────────────────────────────────────────

/// Extract decoded pixel samples from a JPEG 2000 image into a `Vec<f32>`.
///
/// Validates that:
/// - The number of decoded components matches `layout.samples_per_pixel`.
/// - Each component has non-zero dimensions.
/// - Component signedness matches `layout.pixel_representation`.
/// - The total sample count matches `layout.rows × layout.cols × layout.samples_per_pixel`.
pub(super) fn extract_pixels(image: &Image, layout: &PixelLayout) -> Result<Vec<f32>> {
    let comps = image.components();
    let expected_comps = layout.samples_per_pixel;
    if comps.is_empty() || comps.len() != expected_comps {
        bail!(
            "JPEG 2000 numcomps={} does not match PixelLayout samples_per_pixel={}",
            comps.len(),
            expected_comps
        );
    }

    let Some(expected_pixels) = layout.rows.checked_mul(layout.cols) else {
        bail!(
            "JPEG 2000 layout dimensions overflow rows={} cols={}",
            layout.rows,
            layout.cols
        );
    };
    let expected_signed = layout.pixel_representation != 0;
    let Some(expected_samples) = expected_pixels.checked_mul(comps.len()) else {
        bail!(
            "JPEG 2000 sample count overflow pixels={} components={}",
            expected_pixels,
            comps.len()
        );
    };
    let mut out = Vec::with_capacity(expected_samples);

    for (ci, comp) in comps.iter().enumerate() {
        let w = comp.width() as usize;
        let h = comp.height() as usize;
        if w == 0 || h == 0 {
            bail!("JPEG 2000 component {} has zero dimensions {}×{}", ci, w, h);
        }
        if comp.precision() > layout.bits_allocated as u32 {
            bail!(
                "JPEG 2000 component {} precision {} exceeds BitsAllocated {}",
                ci,
                comp.precision(),
                layout.bits_allocated
            );
        }
        if comp.is_signed() != expected_signed {
            bail!(
                "JPEG 2000 component {} signedness {} does not match PixelRepresentation {}",
                ci,
                comp.is_signed(),
                layout.pixel_representation
            );
        }
        let Some(component_pixels) = w.checked_mul(h) else {
            bail!(
                "JPEG 2000 component {} dimensions overflow width={} height={}",
                ci,
                w,
                h
            );
        };
        if component_pixels != expected_pixels {
            bail!(
                "JPEG 2000 component {} decoded size {}×{}={} does not match layout {}×{}={}",
                ci,
                w,
                h,
                component_pixels,
                layout.cols,
                layout.rows,
                expected_pixels
            );
        }

        let raw_slice = comp.data();
        if raw_slice.len() != expected_pixels {
            bail!(
                "JPEG 2000 component {} data length {} does not match expected pixel count {}",
                ci,
                raw_slice.len(),
                expected_pixels
            );
        }
        for &raw in raw_slice {
            // Apply DICOM modality LUT: output = stored_integer × slope + intercept
            // (PS3.3 §C.7.6.3.1).  No [0,1] normalisation — slope and intercept
            // operate on the raw integer sample value, identical to the semantics
            // of decode_native_pixel_bytes_unchecked for uncompressed frames.
            let hu =
                (raw as f64 * layout.rescale_slope as f64 + layout.rescale_intercept as f64) as f32;
            out.push(hu);
        }
    }

    Ok(out)
}
