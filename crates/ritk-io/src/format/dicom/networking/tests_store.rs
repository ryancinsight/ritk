//! C-STORE loopback integration test (GAP-262-IO-02).
//!
//! Spawns a minimal SCP server thread on 127.0.0.1:0, establishes an
//! association from the SCU side via `Association::connect`, sends a
//! C-STORE-RQ, and asserts a Success (0x0000) C-STORE-RSP round-trip.

use crate::format::dicom::networking::association::{transfer_syntax, Association};
use crate::format::dicom::networking::context::{AssociationConfig, RequestedPresentationContext};
use crate::format::dicom::networking::dimse::{CommandField, DimseMessage, DimseStatus};
use crate::format::dicom::networking::pdu::{
    AssociateAcPdu, CommandType, ImplementationClassUidSubItem, ImplementationVersionNameSubItem,
    MaximumLengthSubItem, MessageControlHeader, Pdu, PresentationContextItemAc,
    PresentationDataValueItem, UserInformation, APPLICATION_CONTEXT_NAME,
    RITK_IMPLEMENTATION_CLASS_UID, RITK_IMPLEMENTATION_VERSION,
};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::time::Duration;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn rpc(uid: &str, ts: &[&str]) -> RequestedPresentationContext {
    RequestedPresentationContext {
        abstract_syntax_uid: uid.to_string(),
        transfer_syntax_uids: ts.iter().map(|s| s.to_string()).collect(),
    }
}

/// Read one complete PDU from a TCP stream (6-byte header + body).
fn read_pdu(stream: &mut dyn Read) -> std::io::Result<Vec<u8>> {
    let mut hdr = [0u8; 6];
    stream.read_exact(&mut hdr)?;
    let body_len = u32::from_be_bytes([hdr[2], hdr[3], hdr[4], hdr[5]]) as usize;
    let mut buf = Vec::with_capacity(6 + body_len);
    buf.extend_from_slice(&hdr);
    buf.resize(6 + body_len, 0);
    stream.read_exact(&mut buf[6..])?;
    Ok(buf)
}

/// Write a PDU buffer to a TCP stream.
fn write_pdu(stream: &mut dyn Write, pdu_bytes: &[u8]) -> std::io::Result<()> {
    stream.write_all(pdu_bytes)?;
    stream.flush()
}

/// Build a P-DATA-TF PDU from a list of (context-id, message-control-header, data) triples.
fn build_pdata_pdv(triples: &[(u8, MessageControlHeader, &[u8])]) -> Pdu {
    let items: Vec<PresentationDataValueItem> = triples
        .iter()
        .map(|(cid, mch, data)| PresentationDataValueItem {
            presentation_context_id: *cid,
            message_control_header: *mch,
            data: data.to_vec(),
        })
        .collect();
    Pdu::PDataTf(crate::format::dicom::networking::pdu::PDataTfPdu {
        presentation_data_value_items: items,
    })
}

// ── SCP mock server ───────────────────────────────────────────────────────────

fn scp_thread(listener: TcpListener) {
    let (mut stream, _) = listener.accept().expect("SCP accept");

    // 1. Read A-ASSOCIATE-RQ
    let rq_bytes = read_pdu(&mut stream).expect("SCP read A-ASSOCIATE-RQ");
    let rq_pdu = Pdu::decode(&rq_bytes).expect("SCP decode A-ASSOCIATE-RQ");
    let rq = match &rq_pdu {
        Pdu::AssociateRq(rq) => rq,
        _ => panic!("SCP expected A-ASSOCIATE-RQ"),
    };

    // 2. Build A-ASSOCIATE-AC accepting every presentation context
    let pc_acs: Vec<PresentationContextItemAc> = rq
        .presentation_contexts
        .iter()
        .map(|pc| PresentationContextItemAc {
            presentation_context_id: pc.presentation_context_id,
            result_reason: 0,
            transfer_syntax_uid: pc
                .transfer_syntax_uids
                .first()
                .cloned()
                .unwrap_or_else(|| transfer_syntax::IMPLICIT_VR_LE.to_string()),
        })
        .collect();
    let ac_pdu = Pdu::AssociateAc(AssociateAcPdu {
        protocol_version: 1,
        called_ae_title: rq.called_ae_title.clone(),
        calling_ae_title: rq.calling_ae_title.clone(),
        application_context_name: APPLICATION_CONTEXT_NAME.to_string(),
        presentation_contexts: pc_acs,
        user_information: UserInformation {
            maximum_length: MaximumLengthSubItem {
                maximum_length_received: 16384,
            },
            implementation_class_uid: ImplementationClassUidSubItem {
                implementation_class_uid: RITK_IMPLEMENTATION_CLASS_UID.to_string(),
            },
            implementation_version_name: Some(ImplementationVersionNameSubItem {
                implementation_version_name: RITK_IMPLEMENTATION_VERSION.to_string(),
            }),
            ..Default::default()
        },
    });

    write_pdu(&mut stream, &ac_pdu.encode()).expect("SCP write A-ASSOCIATE-AC");

    // 3. Receive P-DATA-TF PDUs, collect command + data fragments
    let mut cmd_buf = Vec::new();
    let mut data_buf = Vec::new();
    let mut cid: u8 = 0;
    let mut msg_id: u16 = 0;
    let mut sop_class_uid = String::new();
    let mut sop_instance_uid = String::new();
    let mut cmd_complete = false;

    loop {
        let pdu_bytes = read_pdu(&mut stream).expect("SCP read P-DATA-TF");
        let pdu = Pdu::decode(&pdu_bytes).expect("SCP decode PDU");

        match pdu {
            Pdu::PDataTf(pd) => {
                for pdv in &pd.presentation_data_value_items {
                    cid = pdv.presentation_context_id;
                    match pdv.message_control_header.message_type {
                        CommandType::Command => {
                            cmd_buf.extend_from_slice(&pdv.data);
                            if pdv.message_control_header.last_fragment {
                                cmd_complete = true;
                            }
                        }
                        CommandType::DataSet => {
                            data_buf.extend_from_slice(&pdv.data);
                            if pdv.message_control_header.last_fragment {
                                // Data set fully received; break out to send response.
                                break;
                            }
                        }
                    }
                }
                if cmd_complete {
                    let msg =
                        DimseMessage::decode_command_set(&cmd_buf).expect("SCP decode command set");
                    assert_eq!(
                        msg.command_field(),
                        Some(CommandField::CStoreRq),
                        "SCP expected C-STORE-RQ"
                    );
                    msg_id = msg.message_id().unwrap_or(1);
                    sop_class_uid = msg.affected_sop_class_uid().unwrap_or_default();
                    sop_instance_uid = msg.affected_sop_instance_uid().unwrap_or_default();
                    cmd_complete = false;
                    cmd_buf.clear();

                    // Check if the command set indicates no data set (0x0101).
                    // C-STORE-RQ always has a data set, but guard defensively.
                    let has_ds = msg.command_data_set_type() != Some(0x0101);
                    if !has_ds {
                        break;
                    }
                    // Continue reading data-set P-DATA-TF PDUs.
                }
                // If we have collected a complete last-fragment DataSet PDV, break.
                if pd.presentation_data_value_items.iter().any(|p| {
                    p.message_control_header.message_type == CommandType::DataSet
                        && p.message_control_header.last_fragment
                }) {
                    break;
                }
            }
            Pdu::ReleaseRq(_) => {
                // Remote released without sending C-STORE — unexpected but handle.
                let rp = Pdu::ReleaseRp(crate::format::dicom::networking::pdu::ReleaseRpPdu);
                write_pdu(&mut stream, &rp.encode()).expect("SCP write A-RELEASE-RP");
                return;
            }
            _ => panic!("SCP unexpected PDU: {:?}", pdu),
        }
    }

    // 4. Send C-STORE-RSP with Success status
    let rsp = DimseMessage::c_store_rsp(
        msg_id,
        &sop_class_uid,
        &sop_instance_uid,
        DimseStatus::Success as u16,
    );
    let rsp_cmd_bytes = rsp.encode_command_set();

    let cmd_mch = MessageControlHeader {
        message_type: CommandType::Command,
        last_fragment: true,
    };
    let pdata_cmd = build_pdata_pdv(&[(cid, cmd_mch, &rsp_cmd_bytes)]);
    write_pdu(&mut stream, &pdata_cmd.encode()).expect("SCP write C-STORE-RSP");

    // 5. Receive A-RELEASE-RQ, send A-RELEASE-RP
    let rel_bytes = read_pdu(&mut stream).expect("SCP read A-RELEASE-RQ");
    let rel_pdu = Pdu::decode(&rel_bytes).expect("SCP decode release PDU");
    match rel_pdu {
        Pdu::ReleaseRq(_) => {}
        other => panic!("SCP expected A-RELEASE-RQ, got {:?}", other),
    }
    let rp = Pdu::ReleaseRp(crate::format::dicom::networking::pdu::ReleaseRpPdu);
    write_pdu(&mut stream, &rp.encode()).expect("SCP write A-RELEASE-RP");
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[test]
fn test_c_store_loopback_success() {
    const CT_IMAGE_STORAGE: &str = "1.2.840.10008.5.1.4.1.1.2";
    const TEST_INSTANCE_UID: &str = "1.2.3.4.5.6.7.8.9.10";

    // Minimal synthetic dataset (not a valid DICOM object, but exercises
    // the transport layer — the SCP does not parse the data set bytes).
    let dataset: Vec<u8> = vec![0x08, 0x00, 0x5A, 0x44, 0x04, 0x00, 0x54, 0x45, 0x53, 0x54]; // dummy tag+VR+value

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
    let port = listener.local_addr().unwrap().port();
    let scp_handle = std::thread::spawn(move || {
        scp_thread(listener);
    });

    // SCU side
    let config = AssociationConfig {
        called_ae_title: "TESTSCP".into(),
        calling_ae_title: "RITK_TEST".into(),
        host: "127.0.0.1".into(),
        port,
        max_pdu_length: 16384,
        timeout: Duration::from_secs(10),
        presentation_contexts: vec![rpc(CT_IMAGE_STORAGE, &[transfer_syntax::IMPLICIT_VR_LE])],
        user_identity: None,
    };

    let mut assoc = Association::connect(config).expect("SCU connect");
    assert!(assoc.active, "association should be active after connect");

    let status = assoc
        .c_store(CT_IMAGE_STORAGE, TEST_INSTANCE_UID, dataset)
        .expect("SCU c_store");
    assert_eq!(
        status,
        DimseStatus::Success as u16,
        "C-STORE-RSP status must be 0x0000 (Success), got 0x{:04X}",
        status
    );

    assoc.release().expect("SCU release");

    scp_handle.join().expect("SCP thread panicked");
}

#[test]
fn test_c_store_loopback_empty_dataset() {
    const CT_IMAGE_STORAGE: &str = "1.2.840.10008.5.1.4.1.1.2";
    const TEST_INSTANCE_UID: &str = "1.2.3.4.5.6.7.8.9.11";

    // Empty dataset exercises the edge case where P-DATA-TF contains
    // a zero-length DataSet PDV.
    let dataset: Vec<u8> = vec![];

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
    let port = listener.local_addr().unwrap().port();
    let scp_handle = std::thread::spawn(move || {
        scp_thread(listener);
    });

    let config = AssociationConfig {
        called_ae_title: "TESTSCP".into(),
        calling_ae_title: "RITK_TEST".into(),
        host: "127.0.0.1".into(),
        port,
        max_pdu_length: 16384,
        timeout: Duration::from_secs(10),
        presentation_contexts: vec![rpc(CT_IMAGE_STORAGE, &[transfer_syntax::IMPLICIT_VR_LE])],
        user_identity: None,
    };

    let mut assoc = Association::connect(config).expect("SCU connect");
    let status = assoc
        .c_store(CT_IMAGE_STORAGE, TEST_INSTANCE_UID, dataset)
        .expect("SCU c_store empty dataset");
    assert_eq!(
        status,
        DimseStatus::Success as u16,
        "C-STORE-RSP status must be 0x0000 (Success), got 0x{:04X}",
        status
    );

    assoc.release().expect("SCU release");
    scp_handle.join().expect("SCP thread panicked");
}
