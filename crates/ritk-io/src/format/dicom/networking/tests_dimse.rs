//! Value-semantic tests for the DIMSE SCU module.
//!
//! Unit tests validate encoding/decoding deterministically against known byte
//! patterns. Integration tests spin a minimal loopback SCP on a random port
//! using `dicom_ul::ServerAssociationOptions` and verify real DIMSE protocol exchange.

use super::association::{AeTitle, DicomAddress, NetworkingError};
use super::context::AssociationConfig;
use super::command::{
    build_command_pdu, build_dataset_ivr_le, encode_str, encode_ui, encode_us,
    parse_command_response, parse_dataset_ivr_le, C_ECHO_RSP, C_FIND_RSP, C_MOVE_RSP,
    HAS_DATASET, NO_DATASET, STATUS_SUCCESS, STUDY_ROOT_FIND_SOP_CLASS,
    STUDY_ROOT_MOVE_SOP_CLASS, VERIFICATION_SOP_CLASS,
};
use super::echo::echo;
use super::find::{find, FindLevel, FindQuery};
use super::move_::{retrieve, MoveDestination};
use dicom_ul::association::server::ServerAssociationOptions;
use dicom_ul::pdu::{PDataValue, PDataValueType, Pdu};
use std::net::TcpListener;
use std::time::Duration;

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
    let sop_bytes = encode_ui(VERIFICATION_SOP_CLASS);
    let cmd = build_command_pdu(&[
        (0x0000_0002, sop_bytes.as_slice()),
        (0x0000_0100, &encode_us(0x0030u16)),
        (0x0000_0110, &encode_us(1u16)),
        (0x0000_0800, &encode_us(NO_DATASET)),
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
    let sop_bytes = encode_ui(VERIFICATION_SOP_CLASS);
    // Build a synthetic C-ECHO-RSP.
    let cmd = build_command_pdu(&[
        (0x0000_0002, sop_bytes.as_slice()),
        (0x0000_0100, &encode_us(C_ECHO_RSP)),
        (0x0000_0120, &encode_us(1u16)),
        (0x0000_0800, &encode_us(NO_DATASET)),
        (0x0000_0900, &encode_us(STATUS_SUCCESS)),
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
    let sop_bytes = encode_ui(VERIFICATION_SOP_CLASS);
    let cmd = build_command_pdu(&[
        (0x0000_0002, sop_bytes.as_slice()),
        (0x0000_0100, &encode_us(C_ECHO_RSP)),
        (0x0000_0120, &encode_us(1u16)),
        (0x0000_0800, &encode_us(NO_DATASET)),
        (0x0000_0900, &encode_us(STATUS_SUCCESS)),
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

// ── Loopback helpers ──────────────────────────────────────────────────────────

fn loopback_config(port: u16) -> AssociationConfig {
    AssociationConfig::new(
        AeTitle::new("TEST_SCU").unwrap(),
        DicomAddress::new("127.0.0.1", port, AeTitle::new("TEST_SCP").unwrap()),
    )
    .with_connect_timeout(Duration::from_secs(5))
    .with_read_timeout(Duration::from_secs(5))
}

// ── C-ECHO loopback ───────────────────────────────────────────────────────────

#[test]
fn c_echo_loopback_returns_success_status() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();

    let scp = std::thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        let mut assoc = ServerAssociationOptions::new()
            .accept_any()
            .promiscuous(true)
            .ae_title("TEST_SCP")
            .establish(stream)
            .expect("SCP establish");

        let ctx_id = assoc
            .presentation_contexts()
            .first()
            .map(|pc| pc.id)
            .unwrap_or(1);

        loop {
            match assoc.receive() {
                Ok(Pdu::PData { data: pdv_list }) => {
                    let cmd_bytes: Vec<u8> = pdv_list
                        .iter()
                        .filter(|p| {
                            p.presentation_context_id == ctx_id
                                && p.value_type == PDataValueType::Command
                        })
                        .flat_map(|p| p.data.iter().copied())
                        .collect();
                    if cmd_bytes.is_empty() {
                        continue;
                    }
                    let cmd = parse_command_response(&cmd_bytes).unwrap();
                    let sop_bytes = encode_ui(VERIFICATION_SOP_CLASS);
                    let rsp = build_command_pdu(&[
                        (0x0000_0002, sop_bytes.as_slice()),
                        (0x0000_0100, &encode_us(C_ECHO_RSP)),
                        (0x0000_0120, &encode_us(cmd.message_id_responded)),
                        (0x0000_0800, &encode_us(NO_DATASET)),
                        (0x0000_0900, &encode_us(STATUS_SUCCESS)),
                    ]);
                    assoc
                        .send(&Pdu::PData {
                            data: vec![PDataValue {
                                presentation_context_id: ctx_id,
                                value_type: PDataValueType::Command,
                                is_last: true,
                                data: rsp,
                            }],
                        })
                        .unwrap();
                }
                Ok(Pdu::ReleaseRQ) => {
                    let _ = assoc.send(&Pdu::ReleaseRP);
                    break;
                }
                _ => break,
            }
        }
    });

    let config = loopback_config(port);
    let resp = echo(&config).expect("C-ECHO must succeed");
    assert_eq!(resp.status, STATUS_SUCCESS, "C-ECHO status must be 0x0000 SUCCESS");

    scp.join().expect("SCP thread panicked");
}

// ── C-FIND loopback ───────────────────────────────────────────────────────────

#[test]
fn c_find_loopback_returns_synthetic_study_result() {
    const EXPECTED_UID: &str = "1.2.826.0.1.3680043.10.999.1";

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();

    let scp = std::thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        let mut assoc = ServerAssociationOptions::new()
            .accept_any()
            .promiscuous(true)
            .ae_title("TEST_SCP")
            .establish(stream)
            .expect("SCP establish (FIND)");

        let ctx_id = assoc
            .presentation_contexts()
            .first()
            .map(|pc| pc.id)
            .unwrap_or(1);

        loop {
            match assoc.receive() {
                Ok(Pdu::PData { data: pdv_list }) => {
                    let cmd_bytes: Vec<u8> = pdv_list
                        .iter()
                        .filter(|p| {
                            p.presentation_context_id == ctx_id
                                && p.value_type == PDataValueType::Command
                        })
                        .flat_map(|p| p.data.iter().copied())
                        .collect();
                    if cmd_bytes.is_empty() {
                        continue;
                    }
                    let cmd = parse_command_response(&cmd_bytes).unwrap();
                    if cmd.command_field != super::command::C_FIND_RQ {
                        continue;
                    }
                    // Drain data PDV (query).
                    let _ = assoc.receive();

                    // Send pending C-FIND-RSP with synthetic result dataset.
                    let sop_bytes = encode_ui(STUDY_ROOT_FIND_SOP_CLASS);
                    let pending_rsp = build_command_pdu(&[
                        (0x0000_0002, sop_bytes.as_slice()),
                        (0x0000_0100, &encode_us(C_FIND_RSP)),
                        (0x0000_0120, &encode_us(1u16)),
                        (0x0000_0800, &encode_us(HAS_DATASET)),
                        (0x0000_0900, &encode_us(super::command::STATUS_PENDING)),
                    ]);
                    assoc
                        .send(&Pdu::PData {
                            data: vec![PDataValue {
                                presentation_context_id: ctx_id,
                                value_type: PDataValueType::Command,
                                is_last: true,
                                data: pending_rsp,
                            }],
                        })
                        .unwrap();
                    // Send result dataset with StudyInstanceUID.
                    let uid_bytes = encode_ui(EXPECTED_UID);
                    let level_bytes = encode_str("STUDY");
                    let dataset = build_dataset_ivr_le(&[
                        (0x0008_0052, level_bytes.as_slice()),
                        (0x0020_000D, uid_bytes.as_slice()),
                    ]);
                    assoc
                        .send(&Pdu::PData {
                            data: vec![PDataValue {
                                presentation_context_id: ctx_id,
                                value_type: PDataValueType::Data,
                                is_last: true,
                                data: dataset,
                            }],
                        })
                        .unwrap();
                    // Send final SUCCESS C-FIND-RSP (no dataset).
                    let final_rsp = build_command_pdu(&[
                        (0x0000_0002, sop_bytes.as_slice()),
                        (0x0000_0100, &encode_us(C_FIND_RSP)),
                        (0x0000_0120, &encode_us(1u16)),
                        (0x0000_0800, &encode_us(NO_DATASET)),
                        (0x0000_0900, &encode_us(STATUS_SUCCESS)),
                    ]);
                    assoc
                        .send(&Pdu::PData {
                            data: vec![PDataValue {
                                presentation_context_id: ctx_id,
                                value_type: PDataValueType::Command,
                                is_last: true,
                                data: final_rsp,
                            }],
                        })
                        .unwrap();
                }
                Ok(Pdu::ReleaseRQ) => {
                    let _ = assoc.send(&Pdu::ReleaseRP);
                    break;
                }
                _ => break,
            }
        }
    });

    let config = loopback_config(port);
    let query = FindQuery::new(FindLevel::Study).with_key(0x0020, 0x000D, "");
    let results = find(&config, &query).expect("C-FIND must succeed");

    assert_eq!(results.len(), 1, "exactly one pending result expected");
    let uid = results[0].get_string(0x0020, 0x000D).unwrap();
    assert_eq!(uid, EXPECTED_UID, "StudyInstanceUID must match synthetic value");

    scp.join().expect("SCP thread panicked");
}

// ── C-MOVE loopback ───────────────────────────────────────────────────────────

#[test]
fn c_move_loopback_returns_final_success_status() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();

    let scp = std::thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        let mut assoc = ServerAssociationOptions::new()
            .accept_any()
            .promiscuous(true)
            .ae_title("TEST_SCP")
            .establish(stream)
            .expect("SCP establish (MOVE)");

        let ctx_id = assoc
            .presentation_contexts()
            .first()
            .map(|pc| pc.id)
            .unwrap_or(1);

        loop {
            match assoc.receive() {
                Ok(Pdu::PData { data: pdv_list }) => {
                    let cmd_bytes: Vec<u8> = pdv_list
                        .iter()
                        .filter(|p| {
                            p.presentation_context_id == ctx_id
                                && p.value_type == PDataValueType::Command
                        })
                        .flat_map(|p| p.data.iter().copied())
                        .collect();
                    if cmd_bytes.is_empty() {
                        continue;
                    }
                    let cmd = parse_command_response(&cmd_bytes).unwrap();
                    if cmd.command_field != super::command::C_MOVE_RQ {
                        continue;
                    }
                    // Drain data PDV (query).
                    let _ = assoc.receive();

                    let sop_bytes = encode_ui(STUDY_ROOT_MOVE_SOP_CLASS);
                    // Send pending progress response.
                    let pending = build_command_pdu(&[
                        (0x0000_0002, sop_bytes.as_slice()),
                        (0x0000_0100, &encode_us(C_MOVE_RSP)),
                        (0x0000_0120, &encode_us(1u16)),
                        (0x0000_0800, &encode_us(NO_DATASET)),
                        (0x0000_0900, &encode_us(super::command::STATUS_PENDING)),
                        (0x0000_1020, &encode_us(1u16)), // remaining
                        (0x0000_1021, &encode_us(0u16)), // completed
                        (0x0000_1022, &encode_us(0u16)), // failed
                        (0x0000_1023, &encode_us(0u16)), // warning
                    ]);
                    assoc
                        .send(&Pdu::PData {
                            data: vec![PDataValue {
                                presentation_context_id: ctx_id,
                                value_type: PDataValueType::Command,
                                is_last: true,
                                data: pending,
                            }],
                        })
                        .unwrap();
                    // Send final SUCCESS response.
                    let final_rsp = build_command_pdu(&[
                        (0x0000_0002, sop_bytes.as_slice()),
                        (0x0000_0100, &encode_us(C_MOVE_RSP)),
                        (0x0000_0120, &encode_us(1u16)),
                        (0x0000_0800, &encode_us(NO_DATASET)),
                        (0x0000_0900, &encode_us(STATUS_SUCCESS)),
                        (0x0000_1020, &encode_us(0u16)), // remaining
                        (0x0000_1021, &encode_us(1u16)), // completed
                        (0x0000_1022, &encode_us(0u16)), // failed
                        (0x0000_1023, &encode_us(0u16)), // warning
                    ]);
                    assoc
                        .send(&Pdu::PData {
                            data: vec![PDataValue {
                                presentation_context_id: ctx_id,
                                value_type: PDataValueType::Command,
                                is_last: true,
                                data: final_rsp,
                            }],
                        })
                        .unwrap();
                }
                Ok(Pdu::ReleaseRQ) => {
                    let _ = assoc.send(&Pdu::ReleaseRP);
                    break;
                }
                _ => break,
            }
        }
    });

    let config = loopback_config(port);
    let dest = MoveDestination::new(AeTitle::new("MY_SCP").unwrap());
    let resp = retrieve(&config, &dest, "1.2.3.4.5").expect("C-MOVE must succeed");

    assert_eq!(resp.final_status, STATUS_SUCCESS, "final status must be SUCCESS");
    assert_eq!(resp.completed, 1, "one sub-operation completed");
    assert_eq!(resp.failed, 0, "no sub-operation failures");

    scp.join().expect("SCP thread panicked");
}
