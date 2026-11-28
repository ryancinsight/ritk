//! Encode the unsigned 16-bit scalar frames emitted by the DICOM multi-frame writer.
use super::HtJ2kProfile;
use anyhow::{ensure, Context, Result};
use htj2k::{
    codestream::Encoder,
    types::{Point, Size},
};

/// Encode an unsigned 16-bit monochrome DICOM frame. The unrestricted profile
/// writes irreversible 9/7 coefficients. Both lossless profiles use 5/3.
///
/// # Errors
/// Returns an error for an empty frame, inconsistent geometry or an encoding failure.
pub fn encode_grayscale_htj2k(
    samples: &[u16],
    rows: u32,
    columns: u32,
    profile: HtJ2kProfile,
) -> Result<Vec<u8>> {
    let width = usize::try_from(columns)?;
    ensure!(
        width > 0 && rows > 0 && usize::try_from(rows)?.checked_mul(width) == Some(samples.len()),
        "invalid HTJ2K frame dimensions"
    );
    let mut encoder = Encoder::new();
    let siz = encoder.access_siz_mut()?;
    siz.set_image_extent(Point::new(columns, rows));
    siz.set_tile_size(Size::new(columns, rows));
    siz.set_num_components(1);
    siz.set_component(0, Point::new(1, 1), 16, false)?;
    let thumbnail_axis = rows.min(columns);
    let levels = thumbnail_axis
        .div_ceil(64)
        .next_power_of_two()
        .ilog2()
        .max(thumbnail_axis.ilog2().min(3));
    let cod = encoder.access_cod_mut()?;
    cod.set_num_decomposition(levels)?;
    cod.set_reversible(profile != HtJ2kProfile::Unrestricted);
    if profile == HtJ2kProfile::LosslessRpcl {
        cod.set_progression_order("RPCL")?;
        encoder.set_tilepart_divisions(true, false)?;
        encoder.request_tlm_marker(true)?;
    }
    let mut bytes = Vec::new();
    encoder.write_headers(&mut bytes, &[])?;
    let mut lines = samples.chunks_exact(width);
    while let Some((_, buffer)) = encoder.exchange()? {
        let line = lines
            .next()
            .context("HTJ2K encoder requested an extra row")?;
        ensure!(
            buffer.len() == width,
            "HTJ2K encoder requested an invalid row width"
        );
        for (output, &sample) in buffer.iter_mut().zip(line) {
            *output = i32::from(sample);
        }
    }
    ensure!(lines.next().is_none(), "HTJ2K encoder omitted image rows");
    encoder.flush(&mut bytes)?;
    Ok(bytes)
}
