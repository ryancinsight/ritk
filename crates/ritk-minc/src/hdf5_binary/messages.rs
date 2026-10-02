//! HDF5 v1 object-header message encoders for the MINC2 writer.

use anyhow::Result;
use consus_io::WriteAt;
use ritk_codecs::sample::SampleType;

/// Write a v1 object header with the given messages at `offset`.
///
/// v1 OH layout (HDF5 spec §IV.A.1):
///   version(1) + reserved(1) + num_messages(2) + ref_count(4) +
///   header_data_size(4) + padding(4) + messages
///
/// The 4 padding bytes at offset 12–15 are mandatory: the reader advances
/// to byte 16 before parsing the first message (`V1_HEADER_PADDING = 4`).
pub(super) fn write_v1_oh(
    file: &mut std::fs::File,
    offset: u64,
    messages: &[Vec<u8>],
) -> Result<u64> {
    let msg_total: usize = messages.iter().map(|m| m.len()).sum();
    // 12-byte prefix + 4-byte mandatory padding + messages
    let mut header = Vec::with_capacity(16 + msg_total);
    header.push(1); // version
    header.push(0); // reserved
    header.extend_from_slice(&(messages.len() as u16).to_le_bytes());
    header.extend_from_slice(&1u32.to_le_bytes()); // ref_count
    header.extend_from_slice(&(msg_total as u32).to_le_bytes()); // header_data_size
    header.extend_from_slice(&[0u8; 4]); // 4-byte padding (bytes 12–15)
    for msg in messages {
        header.extend_from_slice(msg);
    }
    file.write_at(offset, &header)
        .map_err(|e| anyhow::anyhow!("Failed to write OH at {}: {}", offset, e))?;
    Ok(offset + header.len() as u64)
}

/// Build a v1 hard-link message (type 0x0006).
pub(super) fn build_link_msg(name: &str, target_addr: u64) -> Vec<u8> {
    let name_bytes = name.as_bytes();
    let flags: u8 = 0x00; // hard link, 1-byte name length, no extras
    let mut msg_data = Vec::new();
    msg_data.push(1); // link message version
    msg_data.push(flags);
    msg_data.push(name_bytes.len() as u8);
    msg_data.extend_from_slice(name_bytes);
    msg_data.extend_from_slice(&target_addr.to_le_bytes());

    wrap_message(0x0006, msg_data)
}

#[inline]
fn pad8(n: usize) -> usize {
    (n + 7) & !7
}

/// Wrap message data in a v1 object-header message envelope.
///
/// Envelope: type(2) + size(2) + flags(1) + reserved(3), then the data. The
/// HDF5 v1 object header format requires each message to occupy a multiple of
/// eight bytes and the size field to be rounded up to that boundary, so the data
/// is zero-padded and `size` reports the padded length. Without this, every
/// message after the first parses at a misaligned offset and the file is
/// unreadable.
pub(super) fn wrap_message(type_code: u16, mut msg_data: Vec<u8>) -> Vec<u8> {
    let padded = pad8(msg_data.len());
    msg_data.resize(padded, 0);
    let mut envelope = Vec::with_capacity(8 + padded);
    envelope.extend_from_slice(&type_code.to_le_bytes());
    envelope.extend_from_slice(&(padded as u16).to_le_bytes());
    envelope.push(0);
    envelope.extend_from_slice(&[0u8; 3]);
    envelope.extend_from_slice(&msg_data);
    envelope
}

/// HDF5 datatype descriptor for a little-endian IEEE-754 float (class 1, v1).
///
/// `size` is 4 (`f32`) or 8 (`f64`). The descriptor is the 8-byte header plus
/// the 12 mandatory floating-point property bytes (bit offset/precision,
/// exponent/mantissa location and size, exponent bias); omitting the properties
/// makes the type unreadable ("floating-point properties truncated").
pub(super) fn float_datatype(size: u32) -> Vec<u8> {
    let (exp_size, mant_size, bias): (u8, u8, u32) = match size {
        4 => (8, 23, 127),
        8 => (11, 52, 1023),
        other => unreachable!("unsupported float datatype size {other}"),
    };
    let precision = (size * 8) as u16;
    let mut dt = vec![0u8; 20];
    dt[0] = 0x11; // version 1, class 1 (floating-point)
    dt[1] = 0x20; // bit field: LE byte order, mantissa normalization = 2
    dt[2] = (size * 8 - 1) as u8; // sign bit location
    dt[4..8].copy_from_slice(&size.to_le_bytes());
    dt[10..12].copy_from_slice(&precision.to_le_bytes()); // bit precision (offset stays 0)
    dt[12] = mant_size; // exponent location
    dt[13] = exp_size; // exponent size
    dt[15] = mant_size; // mantissa size (location stays 0)
    dt[16..20].copy_from_slice(&bias.to_le_bytes());
    dt
}

/// HDF5 datatype descriptor for a little-endian fixed-point integer (class 0,
/// v1): 8-byte header plus the 4 mandatory bit-offset/precision property bytes.
fn int_datatype(size: u32, signed: bool) -> Vec<u8> {
    let mut dt = vec![0u8; 12];
    dt[0] = 0x10; // version 1, class 0 (fixed-point)
    dt[1] = if signed { 0x08 } else { 0x00 }; // LE byte order; bit 3 = signed
    dt[4..8].copy_from_slice(&size.to_le_bytes());
    dt[10..12].copy_from_slice(&((size * 8) as u16).to_le_bytes()); // bit precision
    dt
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

    // Datatype, padded to an 8-byte boundary (the reader advances by
    // `align_up(dt_size, 8)`; the size field stays the unpadded length).
    msg_data.extend_from_slice(dt_bytes);
    msg_data.resize(msg_data.len() + pad8(dt_bytes.len()) - dt_bytes.len(), 0);

    // Dataspace: scalar (rank=0).
    msg_data.extend_from_slice(&[1u8, 0, 0, 0, 0, 0, 0, 0]);

    // Data.
    msg_data.extend_from_slice(value_bytes.as_ref());

    wrap_attr_envelope(msg_data)
}

/// Build an attribute message (type 0x000C, v1) for a scalar `f64`.
pub(crate) fn build_attr_msg_float(name: &str, value: f64) -> Vec<u8> {
    build_scalar_attr_raw(name, float_datatype(8), value.to_le_bytes())
}

/// Build an attribute message (type 0x000C, v1) for a scalar `i32`.
pub(crate) fn build_attr_msg_int(name: &str, value: i32) -> Vec<u8> {
    build_scalar_attr_raw(name, int_datatype(4, true), value.to_le_bytes())
}

/// Build an attribute message for a 3-element `f64` array.
///
/// Encodes `direction_cosines` as a 1-D HDF5 float array of 3 `f64` values.
/// The reader's `extract_float_array_3` expects `AttributeValue::FloatArray(3)`.
pub(crate) fn build_attr_msg_float_array(name: &str, values: &[f64; 3]) -> Vec<u8> {
    let name_bytes = name.as_bytes();
    let name_size = name_bytes.len() + 1;
    let datatype = float_datatype(8);
    let dt_size = datatype.len() as u16; // f64 datatype descriptor
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

    // Datatype: 64-bit LE float, padded to an 8-byte boundary.
    msg_data.extend_from_slice(&datatype);
    msg_data.resize(msg_data.len() + pad8(datatype.len()) - datatype.len(), 0);

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
    wrap_message(0x000C, msg_data)
}

/// HDF5 datatype descriptor of `sample_type` stored little-endian.
pub(super) fn sample_datatype(sample_type: SampleType) -> Vec<u8> {
    let size =
        u32::try_from(sample_type.byte_width()).expect("invariant: sample widths are at most 8");
    if sample_type.is_float() {
        float_datatype(size)
    } else {
        let signed = matches!(
            sample_type,
            SampleType::I8 | SampleType::I16 | SampleType::I32 | SampleType::I64
        );
        int_datatype(size, signed)
    }
}

/// Object-header messages of a contiguous dataset: datatype, a fixed dataspace
/// of `dims` (scalar for an empty slice), and the layout of `data_bytes` at
/// `data_offset`.
pub(super) fn dataset_messages(
    datatype: &[u8],
    dims: &[usize],
    data_offset: u64,
    data_bytes: u64,
) -> Vec<Vec<u8>> {
    // DATATYPE (0x0003).
    let datatype_message = wrap_msg(0x0003, datatype);

    // DATASPACE (0x0001): version 1, rank, flags, reserved, then the extents.
    let rank = u8::try_from(dims.len()).expect("invariant: dataset rank is at most 3");
    let mut dataspace = vec![1u8, rank, 0u8, 0u8];
    dataspace.extend_from_slice(&0u32.to_le_bytes());
    for &dim in dims {
        dataspace.extend_from_slice(&(dim as u64).to_le_bytes());
    }
    let dataspace_message = wrap_msg(0x0001, &dataspace);

    // DATA LAYOUT (0x0008): version 3, class 1 = contiguous.
    let mut layout = vec![3u8, 1u8];
    layout.extend_from_slice(&data_offset.to_le_bytes());
    layout.extend_from_slice(&data_bytes.to_le_bytes());
    let layout_message = wrap_msg(0x0008, &layout);

    vec![datatype_message, dataspace_message, layout_message]
}

fn wrap_msg(msg_type: u16, data: &[u8]) -> Vec<u8> {
    wrap_message(msg_type, data.to_vec())
}
