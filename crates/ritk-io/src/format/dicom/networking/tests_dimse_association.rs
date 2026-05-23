//! Loopback integration tests for the DIMSE SCU module.
//!
//! These tests spin a minimal SCP on a random port using
//! `dicom_ul::ServerAssociationOptions` and verify real DIMSE protocol exchange
//! over a TCP loopback connection.

use super::association::{AeTitle, DicomAddress};
use super::command::{
    build_command_pdu, build_dataset_ivr_le, encode_str, encode_ui,
    parse_command_response, CommandElementValue, C_ECHO_RSP, C_FIND_RSP, C_MOVE_RSP, HAS_DATASET,
    NO_DATASET, STATUS_SUCCESS, STUDY_ROOT_FIND_SOP_CLASS, STUDY_ROOT_MOVE_SOP_CLASS,
    VERIFICATION_SOP_CLASS,
};
use super::context::AssociationConfig;
use super::echo::echo;
use super::find::{find, FindLevel, FindQuery};
use super::move_::{retrieve, MoveDestination};

use dicom_ul::association::server::ServerAssociationOptions;
use dicom_ul::pdu::{PDataValue, PDataValueType, Pdu};
use std::net::TcpListener;
use std::time::Duration;

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

                    let rsp = build_command_pdu(&[
                        (0x0000_0002, CommandElementValue::Ui(VERIFICATION_SOP_CLASS)),
                        (0x0000_0100, CommandElementValue::Us(C_ECHO_RSP)),
                        (0x0000_0120, CommandElementValue::Us(cmd.message_id_responded)),
                        (0x0000_0800, CommandElementValue::Us(NO_DATASET)),
                        (0x0000_0900, CommandElementValue::Us(STATUS_SUCCESS)),
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
    assert_eq!(
        resp.status, STATUS_SUCCESS,
        "C-ECHO status must be 0x0000 SUCCESS"
    );

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
                    let pending_rsp = build_command_pdu(&[
                        (0x0000_0002, CommandElementValue::Ui(STUDY_ROOT_FIND_SOP_CLASS)),
                        (0x0000_0100, CommandElementValue::Us(C_FIND_RSP)),
                        (0x0000_0120, CommandElementValue::Us(1)),
                        (0x0000_0800, CommandElementValue::Us(HAS_DATASET)),
                        (0x0000_0900, CommandElementValue::Us(super::command::STATUS_PENDING)),
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
                        (0x0000_0002, CommandElementValue::Ui(STUDY_ROOT_FIND_SOP_CLASS)),
                        (0x0000_0100, CommandElementValue::Us(C_FIND_RSP)),
                        (0x0000_0120, CommandElementValue::Us(1)),
                        (0x0000_0800, CommandElementValue::Us(NO_DATASET)),
                        (0x0000_0900, CommandElementValue::Us(STATUS_SUCCESS)),
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
    assert_eq!(
        uid, EXPECTED_UID,
        "StudyInstanceUID must match synthetic value"
    );

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

                    // Send pending progress response.
                    let pending = build_command_pdu(&[
                        (0x0000_0002, CommandElementValue::Ui(STUDY_ROOT_MOVE_SOP_CLASS)),
                        (0x0000_0100, CommandElementValue::Us(C_MOVE_RSP)),
                        (0x0000_0120, CommandElementValue::Us(1)),
                        (0x0000_0800, CommandElementValue::Us(NO_DATASET)),
                        (0x0000_0900, CommandElementValue::Us(super::command::STATUS_PENDING)),
                        (0x0000_1020, CommandElementValue::Us(1)), // remaining
                        (0x0000_1021, CommandElementValue::Us(0)), // completed
                        (0x0000_1022, CommandElementValue::Us(0)), // failed
                        (0x0000_1023, CommandElementValue::Us(0)), // warning
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
                        (0x0000_0002, CommandElementValue::Ui(STUDY_ROOT_MOVE_SOP_CLASS)),
                        (0x0000_0100, CommandElementValue::Us(C_MOVE_RSP)),
                        (0x0000_0120, CommandElementValue::Us(1)),
                        (0x0000_0800, CommandElementValue::Us(NO_DATASET)),
                        (0x0000_0900, CommandElementValue::Us(STATUS_SUCCESS)),
                        (0x0000_1020, CommandElementValue::Us(0)), // remaining
                        (0x0000_1021, CommandElementValue::Us(1)), // completed
                        (0x0000_1022, CommandElementValue::Us(0)), // failed
                        (0x0000_1023, CommandElementValue::Us(0)), // warning
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

    assert_eq!(
        resp.final_status, STATUS_SUCCESS,
        "final status must be SUCCESS"
    );
    assert_eq!(resp.completed, 1, "one sub-operation completed");
    assert_eq!(resp.failed, 0, "no sub-operation failures");

    scp.join().expect("SCP thread panicked");
}
