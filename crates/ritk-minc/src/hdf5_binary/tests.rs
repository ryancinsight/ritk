#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]
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
    // 12 (prefix) + 4 (mandatory padding) + msg.len()
    assert_eq!(end, (16 + msg.len()) as u64);
}
