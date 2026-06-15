//! Low-level HDF5 binary construction for MINC2 files.
//!
//! Builds a v1-object-header HDF5 file with the MINC2 group hierarchy.
//! Uses `consus_io::WriteAt` for positioned writes.
//!
//! # direction_cosines Encoding
//!
//! The `direction_cosines` attribute is written as a 1-D HDF5 float array
//! of 3 `f64` values (dataspace rank=1, dim0=3). This matches what the
//! MINC2 reader's `parse_dimension_attrs` expects when it calls
//! `extract_float_array_3` on an `AttributeValue::FloatArray(3)`.

use anyhow::Result;
use consus_io::WriteAt;

/// Geometry parameters for a MINC2 volume.
#[derive(Debug, Clone, Copy)]
struct Minc2VolumeGeometry {
    shape: [usize; 3],
    origin: [f64; 3],
    spacing: [f64; 3],
    direction: nalgebra::SMatrix<f64, 3, 3>,
}

/// Construct a MINC2-compliant HDF5 file at `path`.
///
/// # Arguments
///
/// - `path`: output file path.
/// - `raw_data`: voxel bytes (little-endian f32).
/// - `shape`: `[nz, ny, nx]`.
/// - `origin`: physical start per dimorder axis.
/// - `spacing`: voxel spacing per dimorder axis.
/// - `direction`: 3×3 direction matrix (columns = axis direction cosines).
pub fn write_minc2_hdf5(
    path: &std::path::Path,
    raw_data: &[u8],
    shape: [usize; 3],
    origin: [f64; 3],
    spacing: [f64; 3],
    direction: &nalgebra::SMatrix<f64, 3, 3>,
) -> Result<()> {
    let mut file = std::fs::File::create(path)
        .map_err(|e| anyhow::anyhow!("Cannot create MINC2 file {:?}: {}", path, e))?;

    let dim_names = ["zspace", "yspace", "xspace"];
    let offset_size: u8 = 8;
    let length_size: u8 = 8;

    build_minc2_hdf5_binary(
        &mut file,
        raw_data,
        Minc2VolumeGeometry {
            shape,
            origin,
            spacing,
            direction: *direction,
        },
        dim_names,
        offset_size,
        length_size,
    )?;

    std::io::Write::flush(&mut file)
        .map_err(|e| anyhow::anyhow!("Failed to flush MINC2 file: {}", e))?;

    Ok(())
}

// ── Object header helpers ─────────────────────────────────────────────────────

/// Write a v1 object header with the given messages at `offset`.
///
/// v1 OH: version(1) + reserved(1) + num_messages(2) + ref_count(4) +
///        header_data_size(4) + messages
fn write_v1_oh(file: &mut std::fs::File, offset: u64, messages: &[Vec<u8>]) -> Result<u64> {
    let msg_total: usize = messages.iter().map(|m| m.len()).sum();
    let mut header = Vec::with_capacity(12 + msg_total);
    header.push(1); // version
    header.push(0); // reserved
    header.extend_from_slice(&(messages.len() as u16).to_le_bytes());
    header.extend_from_slice(&1u32.to_le_bytes()); // ref_count
    header.extend_from_slice(&(msg_total as u32).to_le_bytes());
    for msg in messages {
        header.extend_from_slice(msg);
    }
    file.write_at(offset, &header)
        .map_err(|e| anyhow::anyhow!("Failed to write OH at {}: {}", offset, e))?;
    Ok(offset + header.len() as u64)
}

/// Build a v1 hard-link message (type 0x0006).
fn build_link_msg(name: &str, target_addr: u64) -> Vec<u8> {
    let name_bytes = name.as_bytes();
    let flags: u8 = 0x00; // hard link, 1-byte name length, no extras
    let mut msg_data = Vec::new();
    msg_data.push(1); // link message version
    msg_data.push(flags);
    msg_data.push(name_bytes.len() as u8);
    msg_data.extend_from_slice(name_bytes);
    msg_data.extend_from_slice(&target_addr.to_le_bytes());

    let mut envelope = Vec::new();
    envelope.extend_from_slice(&0x0006u16.to_le_bytes());
    envelope.extend_from_slice(&(msg_data.len() as u16).to_le_bytes());
    envelope.push(0);
    envelope.extend_from_slice(&[0u8; 3]);
    envelope.extend_from_slice(&msg_data);
    envelope
}

// ── Attribute message helpers ─────────────────────────────────────────────────

#[inline]
fn pad8(n: usize) -> usize {
    (n + 7) & !7
}

/// Build the shared attribute message header and body for a scalar attribute.
///
/// Encodes the name (null-terminated, padded to 8 bytes), the given datatype
/// bytes, a scalar dataspace (rank=0, 8 bytes), and the value bytes, then
/// wraps the whole in an attribute envelope.
fn build_scalar_attr_raw(
    name: &str,
    datatype_bytes: impl AsRef<[u8]>,
    value_bytes: impl AsRef<[u8]>,
) -> Vec<u8> {
    let name_bytes = name.as_bytes();
    let name_size = name_bytes.len() + 1; // null-terminated
    let dt_bytes = datatype_bytes.as_ref();
    let dt_size = dt_bytes.len() as u16;
    let ds_size: u16 = 8; // scalar dataspace

    let mut msg_data = Vec::new();
    msg_data.push(1); // attribute version
    msg_data.push(0); // reserved
    msg_data.extend_from_slice(&(name_size as u16).to_le_bytes());
    msg_data.extend_from_slice(&dt_size.to_le_bytes());
    msg_data.extend_from_slice(&ds_size.to_le_bytes());

    // Name: null-terminated, padded to 8 bytes.
    msg_data.extend_from_slice(name_bytes);
    msg_data.push(0);
    msg_data.resize(msg_data.len() + pad8(name_size) - name_size, 0);

    // Datatype.
    msg_data.extend_from_slice(dt_bytes);

    // Dataspace: scalar (rank=0).
    msg_data.extend_from_slice(&[1u8, 0, 0, 0, 0, 0, 0, 0]);

    // Data.
    msg_data.extend_from_slice(value_bytes.as_ref());

    wrap_attr_envelope(msg_data)
}

/// Build an attribute message (type 0x000C, v1) for a scalar `f64`.
pub(crate) fn build_attr_msg_float(name: &str, value: f64) -> Vec<u8> {
    let mut dt = [0u8; 8];
    dt[0] = 0x11; // class=1 (float), version=1
    dt[1] = 0x20;
    dt[4..8].copy_from_slice(&8u32.to_le_bytes());
    build_scalar_attr_raw(name, dt, value.to_le_bytes())
}

/// Build an attribute message (type 0x000C, v1) for a scalar `i32`.
pub(crate) fn build_attr_msg_int(name: &str, value: i32) -> Vec<u8> {
    let mut dt = [0u8; 8];
    dt[0] = 0x00; // class=0 (integer), version=0
    dt[1] = 0x08; // byte order LE, signed
    dt[4..8].copy_from_slice(&4u32.to_le_bytes());
    build_scalar_attr_raw(name, dt, value.to_le_bytes())
}

/// Build an attribute message for a 3-element `f64` array.
///
/// Encodes `direction_cosines` as a 1-D HDF5 float array of 3 `f64` values.
/// The reader's `extract_float_array_3` expects `AttributeValue::FloatArray(3)`.
pub(crate) fn build_attr_msg_float_array(name: &str, values: &[f64; 3]) -> Vec<u8> {
    let name_bytes = name.as_bytes();
    let name_size = name_bytes.len() + 1;
    let dt_size: u16 = 8; // f64 datatype descriptor
    let ds_size: u16 = 16; // 1-D dataspace: version(1)+rank(1)+flags(1)+rsvd(1)+rsvd(4)+dim0(8)

    let mut msg_data = Vec::new();
    msg_data.push(1);
    msg_data.push(0);
    msg_data.extend_from_slice(&(name_size as u16).to_le_bytes());
    msg_data.extend_from_slice(&dt_size.to_le_bytes());
    msg_data.extend_from_slice(&ds_size.to_le_bytes());

    msg_data.extend_from_slice(name_bytes);
    msg_data.push(0);
    msg_data.resize(msg_data.len() + pad8(name_size) - name_size, 0);

    // Datatype: 64-bit LE float.
    let mut dt = [0u8; 8];
    dt[0] = 0x11;
    dt[1] = 0x20;
    dt[4..8].copy_from_slice(&8u32.to_le_bytes());
    msg_data.extend_from_slice(&dt);

    // Dataspace: 1-D, dim0 = 3.
    let mut ds = [0u8; 16];
    ds[0] = 1; // version
    ds[1] = 1; // rank = 1
               // ds[2] = 0 (no max dims), ds[3..8] reserved
    ds[8..16].copy_from_slice(&3u64.to_le_bytes());
    msg_data.extend_from_slice(&ds);

    // Data: 3 × f64 LE.
    for &v in values {
        msg_data.extend_from_slice(&v.to_le_bytes());
    }

    wrap_attr_envelope(msg_data)
}

fn wrap_attr_envelope(msg_data: Vec<u8>) -> Vec<u8> {
    let mut envelope = Vec::new();
    envelope.extend_from_slice(&0x000Cu16.to_le_bytes()); // ATTRIBUTE
    envelope.extend_from_slice(&(msg_data.len() as u16).to_le_bytes());
    envelope.push(0);
    envelope.extend_from_slice(&[0u8; 3]);
    envelope.extend_from_slice(&msg_data);
    envelope
}

// ── Main builder ──────────────────────────────────────────────────────────────

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
/// ...          0 grp OH       → link "image"
/// ...          image ds OH    → datatype, dataspace, layout
/// offset N:    raw voxel data (contiguous f32 LE)
/// ```
fn build_minc2_hdf5_binary(
    file: &mut std::fs::File,
    raw_data: &[u8],
    geom: Minc2VolumeGeometry,
    dim_names: [&str; 3],
    offset_size: u8,
    length_size: u8,
) -> Result<()> {
    let Minc2VolumeGeometry {
        shape,
        origin,
        spacing,
        direction,
    } = geom;
    let _s = offset_size as usize;
    let _l = length_size as usize;

    // Generous object-header size budgets.
    let oh_group: u64 = 256;
    let oh_dim: u64 = 512; // dimension groups carry 4 attributes each
    let oh_dataset: u64 = 512;

    let root_addr: u64 = 44;
    let minc20_addr = root_addr + oh_group;
    let dims_addr = minc20_addr + oh_group;
    let xspace_addr = dims_addr + oh_group;
    let yspace_addr = xspace_addr + oh_dim;
    let zspace_addr = yspace_addr + oh_dim;
    let image_grp_addr = zspace_addr + oh_dim;
    let zero_grp_addr = image_grp_addr + oh_group;
    let image_ds_addr = zero_grp_addr + oh_group;

    let min_data_offset = image_ds_addr + oh_dataset;
    let data_offset = (min_data_offset + 511) & !511; // 512-byte aligned

    let eof = data_offset + raw_data.len() as u64;

    // ── Superblock v2 ─────────────────────────────────────────────────────
    let mut sb = [0u8; 44];
    sb[0..8].copy_from_slice(b"\x89HDF\r\n\x1a\n");
    sb[8] = 2; // version
    sb[9] = offset_size;
    sb[10] = length_size;
    sb[11] = 0; // consistency flags
    sb[12..20].copy_from_slice(&0u64.to_le_bytes()); // base address
    sb[20..28].copy_from_slice(&u64::MAX.to_le_bytes()); // extension = UNDEF
    sb[28..36].copy_from_slice(&eof.to_le_bytes());
    sb[36..44].copy_from_slice(&root_addr.to_le_bytes());
    file.write_at(0, &sb)
        .map_err(|e| anyhow::anyhow!("Failed to write superblock: {}", e))?;

    // ── Root group OH ────────────────────────────────────────────────────
    let link_minc20 = build_link_msg("minc-2.0", minc20_addr);
    write_v1_oh(file, root_addr, &[link_minc20])?;

    // ── minc-2.0 group OH ────────────────────────────────────────────────
    let link_dims = build_link_msg("dimensions", dims_addr);
    let link_image = build_link_msg("image", image_grp_addr);
    write_v1_oh(file, minc20_addr, &[link_dims, link_image])?;

    // ── dimensions group OH ──────────────────────────────────────────────
    let dim_addrs = [xspace_addr, yspace_addr, zspace_addr];
    let dim_links: Vec<Vec<u8>> = dim_names
        .iter()
        .zip(dim_addrs.iter())
        .map(|(name, addr)| build_link_msg(name, *addr))
        .collect();
    write_v1_oh(file, dims_addr, &dim_links)?;

    // ── Dimension group OHs ───────────────────────────────────────────────
    for (i, &addr) in dim_addrs.iter().enumerate() {
        let start_attr = build_attr_msg_float("start", origin[i]);
        let step_attr = build_attr_msg_float("step", spacing[i]);
        let length_attr = build_attr_msg_int("length", shape[i] as i32);
        // direction_cosines as a single FloatArray(3) attribute.
        let dc = [direction[(0, i)], direction[(1, i)], direction[(2, i)]];
        let dc_attr = build_attr_msg_float_array("direction_cosines", &dc);
        write_v1_oh(file, addr, &[start_attr, step_attr, length_attr, dc_attr])?;
    }

    // ── image group OH ────────────────────────────────────────────────────
    let link_zero = build_link_msg("0", zero_grp_addr);
    write_v1_oh(file, image_grp_addr, &[link_zero])?;

    // ── 0 group OH ────────────────────────────────────────────────────────
    let link_image_ds = build_link_msg("image", image_ds_addr);
    write_v1_oh(file, zero_grp_addr, &[link_image_ds])?;

    // ── image dataset OH ──────────────────────────────────────────────────
    // DATATYPE (0x0003): 32-bit LE float.
    let mut dt_data = [0u8; 8];
    dt_data[0] = 0x11;
    dt_data[1] = 0x20;
    dt_data[2] = 0x1F;
    dt_data[4..8].copy_from_slice(&4u32.to_le_bytes());
    let dt_msg = wrap_msg(0x0003, &dt_data);

    // DATASPACE (0x0001): 3-D fixed.
    let mut ds_data = vec![1u8, 3u8, 0u8, 0u8]; // version, rank=3, flags, reserved
    ds_data.extend_from_slice(&0u32.to_le_bytes()); // reserved
    for &dim in &shape {
        ds_data.extend_from_slice(&(dim as u64).to_le_bytes());
    }
    let ds_msg = wrap_msg(0x0001, &ds_data);

    // DATA LAYOUT (0x0008): contiguous.
    let mut layout_data = Vec::new();
    layout_data.push(3u8); // version 3
    layout_data.push(1u8); // class 1 = contiguous
    layout_data.extend_from_slice(&data_offset.to_le_bytes());
    layout_data.extend_from_slice(&(raw_data.len() as u64).to_le_bytes());
    let layout_msg = wrap_msg(0x0008, &layout_data);

    write_v1_oh(file, image_ds_addr, &[dt_msg, ds_msg, layout_msg])?;

    // ── Raw voxel data ────────────────────────────────────────────────────
    file.write_at(data_offset, raw_data)
        .map_err(|e| anyhow::anyhow!("Failed to write voxel data: {}", e))?;

    Ok(())
}

fn wrap_msg(msg_type: u16, data: &[u8]) -> Vec<u8> {
    let mut envelope = Vec::new();
    envelope.extend_from_slice(&msg_type.to_le_bytes());
    envelope.extend_from_slice(&(data.len() as u16).to_le_bytes());
    envelope.push(0); // flags
    envelope.extend_from_slice(&[0u8; 3]); // reserved
    envelope.extend_from_slice(data);
    envelope
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_attr_msg_float_contains_value() {
        let msg = build_attr_msg_float("start", 3.5);
        // The f64 value 3.5 should appear as LE bytes somewhere in the message.
        let expected = 3.5f64.to_le_bytes();
        assert!(
            msg.windows(8).any(|w| w == expected),
            "f64 value not found in attribute message"
        );
        // Type field 0x000C (ATTRIBUTE).
        assert_eq!(&msg[0..2], &0x000Cu16.to_le_bytes());
    }

    #[test]
    fn build_attr_msg_int_contains_value() {
        let msg = build_attr_msg_int("length", 128);
        let expected = 128i32.to_le_bytes();
        assert!(
            msg.windows(4).any(|w| w == expected),
            "i32 value not found in attribute message"
        );
    }

    #[test]
    fn build_attr_msg_float_array_contains_all_values() {
        let values = [0.707f64, 0.0, -0.707];
        let msg = build_attr_msg_float_array("direction_cosines", &values);
        for &v in &values {
            let expected = v.to_le_bytes();
            assert!(
                msg.windows(8).any(|w| w == expected),
                "f64 value {} not found in array attribute message",
                v
            );
        }
        // Attribute type 0x000C.
        assert_eq!(&msg[0..2], &0x000Cu16.to_le_bytes());
    }

    #[test]
    fn build_attr_msg_float_array_ds_rank_is_one() {
        // The dataspace descriptor in the message must have rank = 1.
        // Verify the dataspace segment size field (ds_size) equals 16.
        let msg = build_attr_msg_float_array("direction_cosines", &[1.0, 0.0, 0.0]);
        // Envelope: type(2) + data_size(2) + flags(1) + reserved(3) = 8 bytes preamble.
        // Then msg_data starts. Offset 8: version(1), reserved(1), name_size(2), dt_size(2), ds_size(2).
        let ds_size_bytes: [u8; 2] = [msg[14], msg[15]];
        let ds_size = u16::from_le_bytes(ds_size_bytes);
        assert_eq!(ds_size, 16u16, "1-D dataspace should be 16 bytes");
    }

    #[test]
    fn write_v1_oh_length_matches_messages() {
        use std::io::Write;
        use tempfile::tempfile;

        let mut f = tempfile().unwrap();
        // Extend file to at least 256 bytes so the write at offset 0 lands inside.
        f.write_all(&[0u8; 256]).unwrap();
        let msg = build_attr_msg_float("start", 1.0);
        let end = write_v1_oh(&mut f, 0, std::slice::from_ref(&msg)).unwrap();
        // 12 (prefix) + msg.len()
        assert_eq!(end, (12 + msg.len()) as u64);
    }
}
