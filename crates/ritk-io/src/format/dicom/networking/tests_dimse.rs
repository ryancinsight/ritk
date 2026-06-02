//! Value-semantic tests for the DIMSE SCU module.
//!
//! Unit tests validate encoding/decoding deterministically against known byte
//! patterns. Integration tests (in `tests_dimse_association.rs`) spin a minimal
//! loopback SCP on a random port and verify real DIMSE protocol exchange.

use super::association::{AeTitle, NetworkingError};
use super::command::{
    build_command_pdu, build_dataset_ivr_le, encode_str, encode_ui, encode_us,
    parse_command_response, parse_dataset_ivr_le, CommandElementValue, C_ECHO_RSP, NO_DATASET,
    STATUS_SUCCESS, VERIFICATION_SOP_CLASS,
};
use super::find::{FindLevel, FindQuery};
use super::move_::MoveDestination;

// ── AeTitle validation ────────────────────────────────────────────────────────

#[test]
fn ae_title_accepts_single_char() {
    let t = AeTitle::new("A").unwrap();
    assert_eq!(t.as_str(), "A");
}

#[test]
fn ae_title_accepts_max_length() {
    let s = "ABCDEFGHIJKLMNOP"; // 16 chars
    let t = AeTitle::new(s).unwrap();
    assert_eq!(t.as_str(), s);
}

#[test]
fn ae_title_accepts_with_spaces() {
    let t = AeTitle::new("MY SCU AET").unwrap();
    assert_eq!(t.as_str(), "MY SCU AET");
}

#[test]
fn ae_title_rejects_empty() {
    assert!(matches!(
        AeTitle::new(""),
        Err(NetworkingError::InvalidAeTitle(_, _))
    ));
}

#[test]
fn ae_title_rejects_too_long() {
    let s = "ABCDEFGHIJKLMNOPQ"; // 17 chars
    assert!(matches!(
        AeTitle::new(s),
        Err(NetworkingError::InvalidAeTitle(_, _))
    ));
}

#[test]
fn ae_title_rejects_backslash() {
    assert!(matches!(
        AeTitle::new("A\\B"),
        Err(NetworkingError::InvalidAeTitle(_, _))
    ));
}

#[test]
fn ae_title_rejects_control_char() {
    assert!(matches!(
        AeTitle::new("AE\x01T"),
        Err(NetworkingError::InvalidAeTitle(_, _))
    ));
}

#[test]
fn ae_title_rejects_del() {
    assert!(matches!(
        AeTitle::new("AE\x7FT"),
        Err(NetworkingError::InvalidAeTitle(_, _))
    ));
}

// ── encode_ui ─────────────────────────────────────────────────────────────────

#[test]
fn encode_ui_odd_length_uid_padded_with_null() {
    // "1.2.3.4" has 7 chars → padded to 8 with null byte.
    let uid = "1.2.3.4";
    let enc = encode_ui(uid);
    assert_eq!(enc.len() % 2, 0, "UI must be even-length");
    assert_eq!(&enc[..7], uid.as_bytes());
    assert_eq!(enc[7], 0x00, "null pad on odd-length UID");
}

#[test]
fn encode_ui_verification_sop_class_odd_gets_null_pad() {
    // "1.2.840.10008.1.1" = 17 chars (odd) → null-padded to 18 bytes.
    let enc = encode_ui(VERIFICATION_SOP_CLASS);
    assert_eq!(enc.len() % 2, 0);
    assert_eq!(enc.len(), 18);
    assert_eq!(enc[17], 0x00, "null pad byte for odd-length UID");
}

// ── encode_us ─────────────────────────────────────────────────────────────────

#[test]
fn encode_us_little_endian() {
    assert_eq!(encode_us(0x0030u16), [0x30, 0x00]);
    assert_eq!(encode_us(0xFEFFu16), [0xFF, 0xFE]);
    assert_eq!(encode_us(0x0000u16), [0x00, 0x00]);
    assert_eq!(encode_us(0xFFFFu16), [0xFF, 0xFF]);
}

// ── encode_str ────────────────────────────────────────────────────────────────

#[test]
fn encode_str_odd_length_padded_with_space() {
    let enc = encode_str("ABC");
    assert_eq!(enc.len(), 4);
    assert_eq!(enc[3], b' ');
}

#[test]
fn encode_str_even_length_no_pad() {
    let enc = encode_str("ABCD");
    assert_eq!(enc.len(), 4);
    assert_eq!(&enc, b"ABCD");
}

// ── build_command_pdu ─────────────────────────────────────────────────────────

#[test]
fn build_command_pdu_group_length_correct() {
    let cmd = build_command_pdu(&[
        (0x0000_0002, CommandElementValue::Ui(VERIFICATION_SOP_CLASS)),
        (0x0000_0100, CommandElementValue::Us(0x0030)),
        (0x0000_0110, CommandElementValue::Us(1)),
        (0x0000_0800, CommandElementValue::Us(NO_DATASET)),
    ]);

    // First 12 bytes: (0000,0000) UL = group length.
    let group = u16::from_le_bytes([cmd[0], cmd[1]]);
    let element = u16::from_le_bytes([cmd[2], cmd[3]]);
    let field_len = u32::from_le_bytes([cmd[4], cmd[5], cmd[6], cmd[7]]);
    let body_len = u32::from_le_bytes([cmd[8], cmd[9], cmd[10], cmd[11]]);

    assert_eq!(group, 0x0000);
    assert_eq!(element, 0x0000);
    assert_eq!(field_len, 4, "UL attribute length must be 4");
    assert_eq!(
        body_len as usize + 12,
        cmd.len(),
        "group length must equal remaining bytes count"
    );
}

#[test]
fn build_command_pdu_round_trips_through_parse() {
    // Build a synthetic C-ECHO-RSP.
    let cmd = build_command_pdu(&[
        (0x0000_0002, CommandElementValue::Ui(VERIFICATION_SOP_CLASS)),
        (0x0000_0100, CommandElementValue::Us(C_ECHO_RSP)),
        (0x0000_0120, CommandElementValue::Us(1)),
        (0x0000_0800, CommandElementValue::Us(NO_DATASET)),
        (0x0000_0900, CommandElementValue::Us(STATUS_SUCCESS)),
    ]);

    let parsed = parse_command_response(&cmd).unwrap();
    assert_eq!(parsed.command_field, C_ECHO_RSP, "command_field");
    assert_eq!(parsed.message_id_responded, 1, "message_id_responded");
    assert_eq!(parsed.data_set_type, NO_DATASET, "data_set_type");
    assert_eq!(parsed.status, STATUS_SUCCESS, "status");
}

// ── build_dataset_ivr_le ──────────────────────────────────────────────────────

#[test]
fn build_dataset_ivr_le_single_element() {
    let level = encode_str("STUDY");
    let ds = build_dataset_ivr_le(&[(0x0008_0052, level.as_slice())]);

    // tag(4) + len(4) + value(6 = "STUDY" + space) = 14 bytes.
    assert_eq!(ds.len(), 14);

    let group = u16::from_le_bytes([ds[0], ds[1]]);
    let element = u16::from_le_bytes([ds[2], ds[3]]);
    let len = u32::from_le_bytes([ds[4], ds[5], ds[6], ds[7]]) as usize;

    assert_eq!(group, 0x0008);
    assert_eq!(element, 0x0052);
    assert_eq!(len, 6);
    assert_eq!(&ds[8..14], b"STUDY ");
}

// ── parse_command_response ────────────────────────────────────────────────────

#[test]
fn parse_command_response_missing_command_field_errors() {
    let err = parse_command_response(&[]);
    assert!(matches!(err, Err(NetworkingError::ParseError(_))));
}

#[test]
fn parse_command_response_c_echo_rsp_from_synthetic_bytes() {
    let cmd = build_command_pdu(&[
        (0x0000_0002, CommandElementValue::Ui(VERIFICATION_SOP_CLASS)),
        (0x0000_0100, CommandElementValue::Us(C_ECHO_RSP)),
        (0x0000_0120, CommandElementValue::Us(1)),
        (0x0000_0800, CommandElementValue::Us(NO_DATASET)),
        (0x0000_0900, CommandElementValue::Us(STATUS_SUCCESS)),
    ]);

    let resp = parse_command_response(&cmd).unwrap();
    assert_eq!(resp.command_field, C_ECHO_RSP, "command_field");
    assert_eq!(resp.message_id_responded, 1, "message_id_responded");
    assert_eq!(resp.data_set_type, NO_DATASET, "data_set_type");
    assert_eq!(resp.status, STATUS_SUCCESS, "status");
}

// ── parse_dataset_ivr_le ──────────────────────────────────────────────────────

#[test]
fn parse_dataset_ivr_le_round_trips_two_elements() {
    let uid = encode_ui("1.2.3.4.5");
    let ds = build_dataset_ivr_le(&[
        (0x0008_0052, encode_str("STUDY").as_slice()),
        (0x0020_000D, uid.as_slice()),
    ]);

    let elements = parse_dataset_ivr_le(&ds);
    assert_eq!(elements.len(), 2);
    assert_eq!(elements[0].0, (0x0008, 0x0052));
    assert_eq!(&elements[0].1, b"STUDY ");
    assert_eq!(elements[1].0, (0x0020, 0x000D));
}

// ── FindQuery ─────────────────────────────────────────────────────────────────

#[test]
fn find_query_builder_stores_keys_in_order() {
    let q = FindQuery::new(FindLevel::Study)
        .with_key(0x0010, 0x0020, "PT001")
        .with_key(0x0020, 0x000D, "");

    assert_eq!(q.level, FindLevel::Study);
    assert_eq!(q.keys.len(), 2);
    assert_eq!(q.keys[0], (0x0010, 0x0020, "PT001".to_owned()));
    assert_eq!(q.keys[1], (0x0020, 0x000D, "".to_owned()));
}

// ── MoveDestination ───────────────────────────────────────────────────────────

#[test]
fn move_destination_holds_ae_title() {
    let ae = AeTitle::new("DEST_SCU").unwrap();
    let dest = MoveDestination::new(ae);
    assert_eq!(dest.as_str(), "DEST_SCU");
}
