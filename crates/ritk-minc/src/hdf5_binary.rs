//! Low-level HDF5 binary construction for MINC2 files.
//!
//! Builds a v1-object-header HDF5 file with the MINC2 group hierarchy.
//! Uses `consus_io::WriteAt` for positioned writes.
//!
//! # Voxel type and real range
//!
//! The image dataset stores the samples of the image's own type, little-endian.
//! An integer image also carries scalar `image-min` and `image-max` datasets
//! equal to the stored type's range, which with the default `valid_range` is
//! the identity map from stored to real values; a floating-point image has
//! none, because MINC applies no scaling to floats.
//!
//! # direction_cosines Encoding
//!
//! The `direction_cosines` attribute is written as a 1-D HDF5 float array
//! of 3 `f64` values (dataspace rank=1, dim0=3). This matches what the
//! MINC2 reader's `parse_dimension_attrs` expects when it calls
//! `extract_float_array_3` on an `AttributeValue::FloatArray(3)`.

mod messages;

use crate::scaling::integer_storage_range;
use anyhow::{Context, Result};
use consus_core::{extend_encoded, ByteOrder};
use consus_io::WriteAt;
use messages::{
    build_attr_msg_float, build_attr_msg_float_array, build_attr_msg_int, build_link_msg,
    dataset_messages, float_datatype, sample_datatype, write_v1_oh,
};
use ritk_codecs::sample::Sample;
use ritk_spatial::Direction;

const VOXEL_STREAM_VALUES: usize = 2_048;
const OFFSET_SIZE: u8 = 8;
const LENGTH_SIZE: u8 = 8;

/// Generous object-header size budgets.
const OH_GROUP: u64 = 256;
/// Dimension groups carry 4 attributes each.
const OH_DIM: u64 = 512;
const OH_DATASET: u64 = 512;

/// Geometry parameters for a MINC2 volume.
#[derive(Debug, Clone, Copy)]
struct Minc2VolumeGeometry {
    shape: [usize; 3],
    origin: [f64; 3],
    spacing: [f64; 3],
    direction: Direction<3>,
}

/// Construct a MINC2-compliant HDF5 file at `path`.
///
/// # Arguments
///
/// - `path`: output file path.
/// - `voxels`: voxel values, encoded as little-endian `T` while writing.
/// - `shape`: `[nz, ny, nx]`.
/// - `origin`: physical start per dimorder axis.
/// - `spacing`: voxel spacing per dimorder axis.
/// - `direction`: 3×3 direction matrix (columns = axis direction cosines).
pub fn write_minc2_hdf5<T: Sample>(
    path: &std::path::Path,
    voxels: &[T],
    shape: [usize; 3],
    origin: [f64; 3],
    spacing: [f64; 3],
    direction: &Direction<3>,
) -> Result<()> {
    let mut file = std::fs::File::create(path)
        .map_err(|e| anyhow::anyhow!("Cannot create MINC2 file {:?}: {}", path, e))?;

    build_minc2_hdf5_binary(
        &mut file,
        voxels,
        Minc2VolumeGeometry {
            shape,
            origin,
            spacing,
            direction: *direction,
        },
        ["zspace", "yspace", "xspace"],
    )?;

    std::io::Write::flush(&mut file)
        .map_err(|e| anyhow::anyhow!("Failed to flush MINC2 file: {}", e))?;

    Ok(())
}

/// Build the HDF5 binary of a MINC2 file using positioned writes.
///
/// # Binary Layout
///
/// ```text
/// Offset 0:    Superblock v2 (44 bytes)
/// Offset 44:   Root group OH  → link "minc-2.0"
/// ...          minc-2.0 OH    → links "dimensions", "image"
/// ...          dimensions OH  → links xspace, yspace, zspace
/// ...          xspace OH      → attrs: start, step, length, direction_cosines
/// ...          yspace OH      → (same)
/// ...          zspace OH      → (same)
/// ...          image grp OH   → link "0"
/// ...          0 grp OH       → links "image", and "image-min", "image-max"
/// ...                           for an integer image
/// ...          image ds OH    → datatype, dataspace, layout
/// ...          image-min OH, image-max OH → scalar f64 datasets
/// offset N:    raw voxel data (contiguous, little-endian)
/// offset M:    image-min and image-max values (integer images)
/// ```
fn build_minc2_hdf5_binary<T: Sample>(
    file: &mut std::fs::File,
    voxels: &[T],
    geom: Minc2VolumeGeometry,
    dim_names: [&str; 3],
) -> Result<()> {
    let Minc2VolumeGeometry {
        shape,
        origin,
        spacing,
        direction,
    } = geom;

    let root_addr: u64 = 44;
    let minc20_addr = root_addr + OH_GROUP;
    let dims_addr = minc20_addr + OH_GROUP;
    let xspace_addr = dims_addr + OH_GROUP;
    let yspace_addr = xspace_addr + OH_DIM;
    let zspace_addr = yspace_addr + OH_DIM;
    let image_grp_addr = zspace_addr + OH_DIM;
    let zero_grp_addr = image_grp_addr + OH_GROUP;
    let image_ds_addr = zero_grp_addr + OH_GROUP;
    let image_min_addr = image_ds_addr + OH_DATASET;
    let image_max_addr = image_min_addr + OH_DATASET;

    let min_data_offset = image_max_addr + OH_DATASET;
    let data_offset = (min_data_offset + 511) & !511; // 512-byte aligned

    let voxel_bytes = voxels
        .len()
        .checked_mul(T::TYPE.byte_width())
        .context("MINC2 voxel byte count overflows usize")?;
    let voxel_bytes_u64 = u64::try_from(voxel_bytes).context("MINC2 voxel payload exceeds u64")?;
    let voxel_end = data_offset
        .checked_add(voxel_bytes_u64)
        .context("MINC2 file length overflows u64")?;
    let image_range = integer_storage_range(T::TYPE);
    let ranges_offset = (voxel_end + 7) & !7;
    let eof = match image_range {
        Some(_) => ranges_offset + 16,
        None => voxel_end,
    };

    let mut sb = [0u8; 44];
    sb[0..8].copy_from_slice(b"\x89HDF\r\n\x1a\n");
    sb[8] = 2; // version
    sb[9] = OFFSET_SIZE;
    sb[10] = LENGTH_SIZE;
    sb[11] = 0; // consistency flags
    sb[12..20].copy_from_slice(&0u64.to_le_bytes()); // base address
    sb[20..28].copy_from_slice(&u64::MAX.to_le_bytes()); // extension = UNDEF
    sb[28..36].copy_from_slice(&eof.to_le_bytes());
    sb[36..44].copy_from_slice(&root_addr.to_le_bytes());
    file.write_at(0, &sb)
        .map_err(|e| anyhow::anyhow!("Failed to write superblock: {}", e))?;

    let link_minc20 = build_link_msg("minc-2.0", minc20_addr);
    write_v1_oh(file, root_addr, &[link_minc20])?;

    let link_dims = build_link_msg("dimensions", dims_addr);
    let link_image = build_link_msg("image", image_grp_addr);
    write_v1_oh(file, minc20_addr, &[link_dims, link_image])?;

    let dim_addrs = [xspace_addr, yspace_addr, zspace_addr];
    let dim_links: Vec<Vec<u8>> = dim_names
        .iter()
        .zip(dim_addrs.iter())
        .map(|(name, addr)| build_link_msg(name, *addr))
        .collect();
    write_v1_oh(file, dims_addr, &dim_links)?;

    for (i, &addr) in dim_addrs.iter().enumerate() {
        let start_attr = build_attr_msg_float("start", origin[i]);
        let step_attr = build_attr_msg_float("step", spacing[i]);
        let length = i32::try_from(shape[i]).context("MINC2 dimension exceeds i32")?;
        let length_attr = build_attr_msg_int("length", length);
        // direction_cosines as a single FloatArray(3) attribute.
        let dc = [direction[(0, i)], direction[(1, i)], direction[(2, i)]];
        let dc_attr = build_attr_msg_float_array("direction_cosines", &dc);
        write_v1_oh(file, addr, &[start_attr, step_attr, length_attr, dc_attr])?;
    }

    let link_zero = build_link_msg("0", zero_grp_addr);
    write_v1_oh(file, image_grp_addr, &[link_zero])?;

    let mut zero_links = vec![build_link_msg("image", image_ds_addr)];
    if image_range.is_some() {
        zero_links.push(build_link_msg("image-min", image_min_addr));
        zero_links.push(build_link_msg("image-max", image_max_addr));
    }
    write_v1_oh(file, zero_grp_addr, &zero_links)?;

    let image_messages = dataset_messages(
        &sample_datatype(T::TYPE),
        &shape,
        data_offset,
        voxel_bytes_u64,
    );
    write_v1_oh(file, image_ds_addr, &image_messages)?;
    write_voxel_stream(file, data_offset, voxels)?;

    if let Some([minimum, maximum]) = image_range {
        for (address, offset, value) in [
            (image_min_addr, ranges_offset, minimum),
            (image_max_addr, ranges_offset + 8, maximum),
        ] {
            let messages = dataset_messages(&float_datatype(8), &[], offset, 8);
            write_v1_oh(file, address, &messages)?;
            file.write_at(offset, &value.to_le_bytes())
                .map_err(|error| anyhow::anyhow!("Failed to write image range: {error}"))?;
        }
    }

    Ok(())
}

/// Write `voxels` as little-endian samples at `data_offset`, one bounded block
/// at a time.
fn write_voxel_stream<T: Sample>(
    file: &mut std::fs::File,
    data_offset: u64,
    voxels: &[T],
) -> Result<()> {
    let mut encoded =
        Vec::with_capacity(VOXEL_STREAM_VALUES.min(voxels.len()) * T::TYPE.byte_width());
    let mut written = 0_u64;
    for chunk in voxels.chunks(VOXEL_STREAM_VALUES) {
        encoded.clear();
        extend_encoded(&mut encoded, chunk.iter().copied(), ByteOrder::LittleEndian);
        let offset = data_offset
            .checked_add(written)
            .context("MINC2 voxel write offset overflows u64")?;
        file.write_at(offset, &encoded)
            .map_err(|error| anyhow::anyhow!("Failed to write voxel data: {error}"))?;
        let chunk_bytes = u64::try_from(encoded.len()).context("MINC2 chunk exceeds u64")?;
        written = written
            .checked_add(chunk_bytes)
            .context("MINC2 written byte count overflows u64")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
