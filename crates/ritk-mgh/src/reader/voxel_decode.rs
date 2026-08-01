//! Bounded streaming conversion of big-endian MGH voxels.

use crate::types::VoxelType;
use anyhow::{bail, Context, Result};
use std::io::{ErrorKind, Read};

/// Fixed input scratch for streaming voxel conversion.
///
/// The value is divisible by every supported voxel width. At 16 KiB it keeps
/// the decoder's temporary storage small while amortizing `Read` and output
/// reservation overhead over thousands of scalar values.
const INPUT_CHUNK_BYTES: usize = 16 * 1024;

pub(super) fn decode_voxels<R: Read>(
    reader: &mut R,
    voxel_type: VoxelType,
    voxel_count: usize,
) -> Result<Vec<f32>> {
    match voxel_type {
        VoxelType::UnsignedByte => {
            decode_stream::<1, _, _>(reader, voxel_count, decode_unsigned_byte)
        }
        VoxelType::SignedShort => {
            decode_stream::<2, _, _>(reader, voxel_count, decode_signed_short)
        }
        VoxelType::SignedInteger => {
            decode_stream::<4, _, _>(reader, voxel_count, decode_signed_integer)
        }
        VoxelType::Float => decode_stream::<4, _, _>(reader, voxel_count, decode_float),
    }
}

pub(super) fn decode_volumes<R: Read>(
    reader: &mut R,
    voxel_type: VoxelType,
    voxels_per_volume: usize,
    volume_count: usize,
) -> Result<Vec<Vec<f32>>> {
    let mut volumes = Vec::new();
    volumes
        .try_reserve_exact(volume_count)
        .context("cannot allocate MGH volume table")?;
    for volume in 0..volume_count {
        let values = decode_voxels(reader, voxel_type, voxels_per_volume)
            .with_context(|| format!("failed to decode MGH volume {volume} of {volume_count}"))?;
        volumes.push(values);
    }
    Ok(volumes)
}

fn decode_stream<const BYTES_PER_VOXEL: usize, R: Read, F>(
    reader: &mut R,
    voxel_count: usize,
    mut decode_chunk: F,
) -> Result<Vec<f32>>
where
    F: FnMut(&[u8], &mut Vec<f32>),
{
    let voxels_per_chunk = INPUT_CHUNK_BYTES / BYTES_PER_VOXEL;
    let mut input = [0u8; INPUT_CHUNK_BYTES];
    let mut output = Vec::new();

    while output.len() < voxel_count {
        let chunk_voxels = (voxel_count - output.len()).min(voxels_per_chunk);
        let chunk_bytes = chunk_voxels
            .checked_mul(BYTES_PER_VOXEL)
            .context("MGH decode chunk byte count overflows usize")?;
        let voxel_offset = output.len();

        read_encoded_chunk::<BYTES_PER_VOXEL, _>(
            reader,
            &mut input[..chunk_bytes],
            voxel_offset,
            voxel_count,
        )?;

        reserve_from_confirmed_input(&mut output, chunk_voxels, voxel_count)?;
        decode_chunk(&input[..chunk_bytes], &mut output);
        debug_assert_eq!(output.len(), voxel_offset + chunk_voxels);
    }

    Ok(output)
}

fn read_encoded_chunk<const BYTES_PER_VOXEL: usize, R: Read>(
    reader: &mut R,
    input: &mut [u8],
    voxel_offset: usize,
    voxel_count: usize,
) -> Result<()> {
    let mut bytes_read = 0usize;
    while bytes_read < input.len() {
        match reader.read(&mut input[bytes_read..]) {
            Ok(0) => {
                let incomplete_voxel = voxel_offset + bytes_read / BYTES_PER_VOXEL;
                bail!(
                    "MGH voxel payload is truncated at voxel {incomplete_voxel} \
                     of {voxel_count}"
                );
            }
            Ok(read) => {
                bytes_read = bytes_read
                    .checked_add(read)
                    .context("confirmed MGH payload byte count overflows usize")?;
            }
            Err(error) if error.kind() == ErrorKind::Interrupted => {}
            Err(error) => {
                let incomplete_voxel = voxel_offset + bytes_read / BYTES_PER_VOXEL;
                return Err(error).with_context(|| {
                    format!(
                        "failed to read MGH voxel {incomplete_voxel} \
                         of {voxel_count}"
                    )
                });
            }
        }
    }
    Ok(())
}

fn reserve_from_confirmed_input(
    output: &mut Vec<f32>,
    chunk_voxels: usize,
    voxel_count: usize,
) -> Result<()> {
    let confirmed = output
        .len()
        .checked_add(chunk_voxels)
        .context("confirmed MGH voxel count overflows usize")?;
    if confirmed > output.capacity() {
        let growth_target = output
            .capacity()
            .max(1)
            .saturating_mul(2)
            .max(confirmed)
            .min(voxel_count);
        output
            .try_reserve_exact(growth_target - output.len())
            .with_context(|| {
                format!(
                    "cannot allocate decoded MGH output for {growth_target} \
                     confirmed voxels"
                )
            })?;
    }
    Ok(())
}

fn decode_unsigned_byte(input: &[u8], output: &mut Vec<f32>) {
    output.extend(input.iter().copied().map(f32::from));
}

fn decode_signed_short(input: &[u8], output: &mut Vec<f32>) {
    output.extend(input.chunks_exact(2).map(|bytes| {
        let [high, low] = bytes else {
            unreachable!("two-byte chunks have exactly two elements")
        };
        f32::from(i16::from_be_bytes([*high, *low]))
    }));
}

fn decode_signed_integer(input: &[u8], output: &mut Vec<f32>) {
    output.extend(input.chunks_exact(4).map(|bytes| {
        let [a, b, c, d] = bytes else {
            unreachable!("four-byte chunks have exactly four elements")
        };
        // MGH MRI_INT is converted to the crate's documented f32 image output.
        // Values beyond f32's exact-integer range round according to Rust's
        // numeric-cast semantics.
        i32::from_be_bytes([*a, *b, *c, *d]) as f32
    }));
}

fn decode_float(input: &[u8], output: &mut Vec<f32>) {
    output.extend(input.chunks_exact(4).map(|bytes| {
        let [a, b, c, d] = bytes else {
            unreachable!("four-byte chunks have exactly four elements")
        };
        f32::from_be_bytes([*a, *b, *c, *d])
    }));
}

#[cfg(test)]
pub(super) const fn input_chunk_bytes() -> usize {
    INPUT_CHUNK_BYTES
}
