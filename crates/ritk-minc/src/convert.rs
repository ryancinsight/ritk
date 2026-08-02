//! Raw byte to `f32` conversion for MINC2 voxel data.
//!
//! Converts a byte slice from HDF5 contiguous storage into a `Vec<f32>`
//! based on the HDF5 `Datatype` metadata. All integer types are cast
//! to `f32`; floating-point 64-bit values are narrowed from `f64` to
//! `f32` (lossy, but within RITK tensor precision contract).

use crate::scaling::IntegerScaling;
use anyhow::{bail, Context, Result};
use consus_core::Datatype;

fn append_mapped<I, F>(
    values: I,
    start_index: usize,
    output: &mut Vec<f32>,
    mut map: F,
) -> Result<()>
where
    I: ExactSizeIterator<Item = f64>,
    F: FnMut(usize, f64) -> Result<f32>,
{
    output
        .try_reserve(values.len())
        .context("reserve decoded MINC2 voxel chunk")?;
    for (offset, value) in values.enumerate() {
        let index = start_index
            .checked_add(offset)
            .context("MINC2 voxel index overflows usize")?;
        output.push(map(index, value)?);
    }
    Ok(())
}

fn append_float<I>(values: I, start_index: usize, output: &mut Vec<f32>) -> Result<()>
where
    I: ExactSizeIterator<Item = f64>,
{
    append_mapped(values, start_index, output, |_, value| Ok(value as f32))
}

fn append_integer<I>(
    values: I,
    start_index: usize,
    output: &mut Vec<f32>,
    scaling: Option<&IntegerScaling>,
) -> Result<()>
where
    I: ExactSizeIterator<Item = f64>,
{
    match scaling {
        Some(scaling) => append_mapped(values, start_index, output, |index, value| {
            scaling.scale(value, index)
        }),
        None => append_mapped(values, start_index, output, |_, value| Ok(value as f32)),
    }
}

/// Decode one element-aligned raw chunk into an existing output buffer.
pub(crate) fn decode_raw_bytes_into(
    raw: &[u8],
    dtype: &Datatype,
    start_index: usize,
    output: &mut Vec<f32>,
    integer_scaling: Option<&IntegerScaling>,
) -> Result<()> {
    use consus_core::ByteOrder;

    let element_size = dtype
        .element_size()
        .context("Variable-length MINC2 voxel datatype is unsupported")?;
    if !raw.len().is_multiple_of(element_size) {
        bail!(
            "MINC2 raw chunk length {} is not divisible by element width {element_size}",
            raw.len()
        );
    }

    match dtype {
        Datatype::Float { bits, byte_order } => {
            let bw = bits.get();
            match (bw, byte_order) {
                (32, ByteOrder::LittleEndian) => append_float(
                    raw.chunks_exact(4).map(|chunk| {
                        f64::from(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    }),
                    start_index,
                    output,
                ),
                (32, ByteOrder::BigEndian) => append_float(
                    raw.chunks_exact(4).map(|chunk| {
                        f64::from(f32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    }),
                    start_index,
                    output,
                ),
                (64, ByteOrder::LittleEndian) => append_float(
                    raw.chunks_exact(8).map(|chunk| {
                        f64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ])
                    }),
                    start_index,
                    output,
                ),
                (64, ByteOrder::BigEndian) => append_float(
                    raw.chunks_exact(8).map(|chunk| {
                        f64::from_be_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ])
                    }),
                    start_index,
                    output,
                ),
                _ => bail!("Unsupported float bit width: {}", bw),
            }
        }

        Datatype::Integer {
            bits,
            byte_order,
            signed,
        } => {
            let bw = bits.get();
            match (bw, byte_order, signed) {
                // 8-bit
                (8, _, false) => append_integer(
                    raw.iter().map(|&value| f64::from(value)),
                    start_index,
                    output,
                    integer_scaling,
                ),
                (8, _, true) => append_integer(
                    raw.iter().map(|&value| f64::from(value as i8)),
                    start_index,
                    output,
                    integer_scaling,
                ),

                // 16-bit little-endian
                (16, ByteOrder::LittleEndian, true) => append_integer(
                    raw.chunks_exact(2)
                        .map(|chunk| f64::from(i16::from_le_bytes([chunk[0], chunk[1]]))),
                    start_index,
                    output,
                    integer_scaling,
                ),
                (16, ByteOrder::LittleEndian, false) => append_integer(
                    raw.chunks_exact(2)
                        .map(|chunk| f64::from(u16::from_le_bytes([chunk[0], chunk[1]]))),
                    start_index,
                    output,
                    integer_scaling,
                ),

                // 16-bit big-endian
                (16, ByteOrder::BigEndian, true) => append_integer(
                    raw.chunks_exact(2)
                        .map(|chunk| f64::from(i16::from_be_bytes([chunk[0], chunk[1]]))),
                    start_index,
                    output,
                    integer_scaling,
                ),
                (16, ByteOrder::BigEndian, false) => append_integer(
                    raw.chunks_exact(2)
                        .map(|chunk| f64::from(u16::from_be_bytes([chunk[0], chunk[1]]))),
                    start_index,
                    output,
                    integer_scaling,
                ),

                // 32-bit little-endian
                (32, ByteOrder::LittleEndian, true) => append_integer(
                    raw.chunks_exact(4).map(|chunk| {
                        f64::from(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    }),
                    start_index,
                    output,
                    integer_scaling,
                ),
                (32, ByteOrder::LittleEndian, false) => append_integer(
                    raw.chunks_exact(4).map(|chunk| {
                        f64::from(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    }),
                    start_index,
                    output,
                    integer_scaling,
                ),

                // 32-bit big-endian
                (32, ByteOrder::BigEndian, true) => append_integer(
                    raw.chunks_exact(4).map(|chunk| {
                        f64::from(i32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    }),
                    start_index,
                    output,
                    integer_scaling,
                ),
                (32, ByteOrder::BigEndian, false) => append_integer(
                    raw.chunks_exact(4).map(|chunk| {
                        f64::from(u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    }),
                    start_index,
                    output,
                    integer_scaling,
                ),

                // 64-bit (lossy cast to f32)
                (64, ByteOrder::LittleEndian, true) => append_integer(
                    raw.chunks_exact(8).map(|chunk| {
                        i64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ]) as f64
                    }),
                    start_index,
                    output,
                    integer_scaling,
                ),
                (64, ByteOrder::LittleEndian, false) => append_integer(
                    raw.chunks_exact(8).map(|chunk| {
                        u64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ]) as f64
                    }),
                    start_index,
                    output,
                    integer_scaling,
                ),
                (64, ByteOrder::BigEndian, true) => append_integer(
                    raw.chunks_exact(8).map(|chunk| {
                        i64::from_be_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ]) as f64
                    }),
                    start_index,
                    output,
                    integer_scaling,
                ),
                (64, ByteOrder::BigEndian, false) => append_integer(
                    raw.chunks_exact(8).map(|chunk| {
                        u64::from_be_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ]) as f64
                    }),
                    start_index,
                    output,
                    integer_scaling,
                ),

                _ => bail!("Unsupported integer type: {} bits, signed={}", bw, signed),
            }
        }

        Datatype::Boolean => append_integer(
            raw.iter().map(|&value| if value == 0 { 0.0 } else { 1.0 }),
            start_index,
            output,
            integer_scaling,
        ),

        other => bail!("Unsupported MINC2 voxel datatype: {:?}", other),
    }
}

/// Decode raw bytes to `Vec<f32>` based on the HDF5 datatype.
///
/// This low-level conversion preserves stored numeric values. The MINC2 reader
/// applies quantitative integer scaling separately while streaming the image
/// dataset.
///
/// # Errors
///
/// Returns `Err` for unsupported or variable-length data types and for an input
/// length that is not an exact multiple of the element width.
pub fn decode_raw_bytes(raw: &[u8], dtype: &Datatype) -> Result<Vec<f32>> {
    let mut output = Vec::new();
    decode_raw_bytes_into(raw, dtype, 0, &mut output, None)?;
    Ok(output)
}

/// Decode a floating-point metadata dataset without narrowing its `f64` values.
pub(crate) fn decode_float_bytes(raw: &[u8], dtype: &Datatype) -> Result<Vec<f64>> {
    use consus_core::ByteOrder;

    let (width, little_endian) = match dtype {
        Datatype::Float { bits, byte_order } => match (bits.get(), byte_order) {
            (32, ByteOrder::LittleEndian) => (4, true),
            (32, ByteOrder::BigEndian) => (4, false),
            (64, ByteOrder::LittleEndian) => (8, true),
            (64, ByteOrder::BigEndian) => (8, false),
            (bits, _) => bail!("Unsupported MINC2 image-range float width: {bits}"),
        },
        other => bail!("MINC2 image-min/image-max must be floating-point, got {other:?}"),
    };
    if !raw.len().is_multiple_of(width) {
        bail!(
            "MINC2 image-range byte length {} is not divisible by element width {width}",
            raw.len()
        );
    }

    let count = raw.len() / width;
    let mut output = Vec::new();
    output
        .try_reserve_exact(count)
        .context("reserve MINC2 image-range values")?;
    for chunk in raw.chunks_exact(width) {
        let value = match (width, little_endian) {
            (4, true) => f64::from(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])),
            (4, false) => f64::from(f32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])),
            (8, true) => f64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]),
            (8, false) => f64::from_be_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]),
            _ => unreachable!("invariant: width is validated as four or eight bytes"),
        };
        output.push(value);
    }
    Ok(output)
}

#[cfg(test)]
#[path = "tests_convert.rs"]
mod tests;
