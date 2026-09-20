//! Pixel-space viewport transforms shared by browser presentation and input.

use crate::tools::interaction::ViewportOffset;
use anyhow::{bail, Result};

const RGBA_BYTES_PER_PIXEL: usize = 4;

/// Applies the viewer zoom and pan policy to a row-major RGBA frame.
///
/// The output keeps the source dimensions. Pixel centres are mapped around
/// the frame centre, so identity state is byte-preserving, zoom crops the
/// centre, and positive pan moves the image right and down. Samples outside
/// the source are opaque black. The destination buffer is caller-owned and
/// reused across frames.
pub(super) fn transform_rgba(
    source_size: [usize; 2],
    source: &[u8],
    zoom: f32,
    pan: ViewportOffset,
    destination: &mut Vec<u8>,
) -> Result<()> {
    let [width, height] = source_size;
    if width == 0 || height == 0 {
        bail!("viewport source dimensions must be nonzero");
    }
    let pixel_count = width
        .checked_mul(height)
        .ok_or_else(|| anyhow::anyhow!("viewport pixel count overflows usize"))?;
    let byte_count = pixel_count
        .checked_mul(RGBA_BYTES_PER_PIXEL)
        .ok_or_else(|| anyhow::anyhow!("viewport byte count overflows usize"))?;
    if source.len() != byte_count {
        bail!(
            "viewport source byte count {} does not match {}x{} RGBA storage",
            source.len(),
            width,
            height
        );
    }
    if !zoom.is_finite() || zoom <= 0.0 {
        bail!("viewport zoom must be finite and positive");
    }
    if !pan.x().is_finite() || !pan.y().is_finite() {
        bail!("viewport pan must be finite");
    }

    destination.resize(byte_count, 0);
    let zoom = f64::from(zoom);
    let pan_x = f64::from(pan.x());
    let pan_y = f64::from(pan.y());
    let center_x = width as f64 / 2.0;
    let center_y = height as f64 / 2.0;

    for output_y in 0..height {
        for output_x in 0..width {
            let output_index = (output_y * width + output_x) * RGBA_BYTES_PER_PIXEL;
            let source_x = ((output_x as f64 + 0.5 - center_x - pan_x) / zoom) + center_x - 0.5;
            let source_y = ((output_y as f64 + 0.5 - center_y - pan_y) / zoom) + center_y - 0.5;
            let Some(source_x) = source_x.is_finite().then_some(source_x) else {
                destination[output_index..output_index + RGBA_BYTES_PER_PIXEL]
                    .copy_from_slice(&[0, 0, 0, 255]);
                continue;
            };
            let Some(source_y) = source_y.is_finite().then_some(source_y) else {
                destination[output_index..output_index + RGBA_BYTES_PER_PIXEL]
                    .copy_from_slice(&[0, 0, 0, 255]);
                continue;
            };
            let source_x = source_x.floor();
            let source_y = source_y.floor();
            if source_x < 0.0
                || source_y < 0.0
                || source_x >= width as f64
                || source_y >= height as f64
            {
                destination[output_index..output_index + RGBA_BYTES_PER_PIXEL]
                    .copy_from_slice(&[0, 0, 0, 255]);
                continue;
            }
            let source_x = source_x as usize;
            let source_y = source_y as usize;
            let source_index = (source_y * width + source_x) * RGBA_BYTES_PER_PIXEL;
            destination[output_index..output_index + RGBA_BYTES_PER_PIXEL]
                .copy_from_slice(&source[source_index..source_index + RGBA_BYTES_PER_PIXEL]);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::transform_rgba;
    use crate::tools::interaction::ViewportOffset;

    fn pixels() -> Vec<u8> {
        (0_u8..8)
            .flat_map(|value| [value, value.wrapping_add(10), value.wrapping_add(20), 255])
            .collect()
    }

    #[test]
    fn identity_preserves_rgba_pixels() {
        let source = pixels();
        let mut destination = Vec::new();
        transform_rgba(
            [4, 2],
            &source,
            1.0,
            ViewportOffset::new(0.0, 0.0),
            &mut destination,
        )
        .expect("identity transform");
        assert_eq!(destination, source);
    }

    #[test]
    fn repeated_transforms_reuse_destination_capacity() {
        let source = pixels();
        let mut destination = Vec::with_capacity(source.len());
        transform_rgba(
            [4, 2],
            &source,
            2.0,
            ViewportOffset::new(0.0, 0.0),
            &mut destination,
        )
        .expect("first transform");
        let capacity = destination.capacity();
        transform_rgba(
            [4, 2],
            &source,
            1.5,
            ViewportOffset::new(0.5, -0.25),
            &mut destination,
        )
        .expect("second transform");
        assert_eq!(destination.capacity(), capacity);
    }

    #[test]
    fn zoom_crops_around_the_frame_centre() {
        let source = pixels();
        let mut destination = Vec::new();
        transform_rgba(
            [4, 2],
            &source,
            2.0,
            ViewportOffset::new(0.0, 0.0),
            &mut destination,
        )
        .expect("zoom transform");
        assert_eq!(
            destination,
            [
                0, 10, 20, 255, 1, 11, 21, 255, 1, 11, 21, 255, 2, 12, 22, 255, 0, 10, 20, 255, 1,
                11, 21, 255, 1, 11, 21, 255, 2, 12, 22, 255,
            ]
        );
    }

    #[test]
    fn pan_moves_pixels_and_fills_exposed_edges() {
        let source = pixels();
        let mut destination = Vec::new();
        transform_rgba(
            [4, 2],
            &source,
            1.0,
            ViewportOffset::new(1.0, 0.0),
            &mut destination,
        )
        .expect("pan transform");
        assert_eq!(
            destination,
            [
                0, 0, 0, 255, 0, 10, 20, 255, 1, 11, 21, 255, 2, 12, 22, 255, 0, 0, 0, 255, 4, 14,
                24, 255, 5, 15, 25, 255, 6, 16, 26, 255,
            ]
        );
    }

    #[test]
    fn invalid_transform_inputs_are_rejected() {
        let mut destination = Vec::new();
        for (zoom, pan) in [
            (0.0, ViewportOffset::new(0.0, 0.0)),
            (f32::NAN, ViewportOffset::new(0.0, 0.0)),
            (1.0, ViewportOffset::new(f32::INFINITY, 0.0)),
        ] {
            assert!(
                transform_rgba([1, 1], &[1, 2, 3, 4], zoom, pan, &mut destination).is_err(),
                "invalid transform must be rejected"
            );
        }
    }
}
