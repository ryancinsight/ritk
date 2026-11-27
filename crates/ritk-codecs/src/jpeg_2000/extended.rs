//! Decode general Part 1/15 packet organizations into RITK's stored-sample domain.
use super::codestream::SizMarker;
use crate::PixelLayout;
use anyhow::{ensure, Context, Result};
use j2k_native::{DecodeSettings, Image};

pub(super) fn decode<T: Copy + Default>(
    data: &[u8],
    layout: PixelLayout,
    siz: &SizMarker,
    convert: impl Fn(i64) -> T,
) -> Result<Vec<T>> {
    let image = Image::new(data, &DecodeSettings::default()).context("JPEG 2000 packet headers")?;
    let decoded = image
        .decode_native_components()
        .context("JPEG 2000 component reconstruction")?;
    let planes = decoded.planes();
    ensure!(
        planes.len() == layout.samples_per_pixel,
        "component count changed during reconstruction"
    );
    let count = layout.samples_per_frame()?;
    let mut out = Vec::new();
    out.try_reserve_exact(count)?;
    out.resize(count, T::default());
    for (channel, (plane, spec)) in planes.iter().zip(&siz.components).enumerate() {
        let width = usize::from(plane.bytes_per_sample());
        ensure!(
            u32::from(plane.bit_depth()) == spec.precision()
                && plane.signed() == spec.is_signed()
                && width == usize::from(plane.bit_depth()).div_ceil(8)
                && plane.dimensions() == (siz.width(), siz.height())
                && plane.data().len()
                    == (count / planes.len())
                        .checked_mul(width)
                        .context("component extent overflow")?,
            "invalid reconstructed component layout"
        );
        let shift = 64 - spec.precision();
        for (i, word) in plane.data().chunks_exact(width).enumerate() {
            let mut bytes = [0; 8];
            bytes[..width].copy_from_slice(word);
            let value = u64::from_le_bytes(bytes) << shift;
            let raw = if spec.is_signed() {
                (value as i64) >> shift
            } else {
                i64::try_from(value >> shift)?
            };
            out[i * planes.len() + channel] = convert(raw);
        }
    }
    Ok(out)
}
