//! Embedded C-STORE SCP loopback integration tests.
//!
//! Each test starts the SCP with port=0 (OS-assigned), connects via
//! `Association::connect`, sends one or more C-STORE-RQ messages, then blocks
//! on `StoreScpHandle::recv_timeout` to verify the received instances.
//!
//! No external PACS is required — all tests run fully in-process.

use crate::format::dicom::networking::association::Association;
use crate::format::dicom::networking::context::transfer_syntax;
use crate::format::dicom::networking::context::{AssociationConfig, RequestedPresentationContext};
use crate::format::dicom::networking::dimse::DimseStatus;
use crate::format::dicom::networking::scp::{ScpConfig, StoreScp};
use arrayvec::ArrayString;
use std::time::Duration;

// ── SOP class constant ────────────────────────────────────────────────────────

const CT_IMAGE_STORAGE: &str = "1.2.840.10008.5.1.4.1.1.2";

// ── Helper ────────────────────────────────────────────────────────────────────

fn scu_config(port: u16) -> AssociationConfig {
    AssociationConfig {
        called_ae_title: ArrayString::from("RITKSNAP").expect("infallible: validated precondition"),
        calling_ae_title: ArrayString::from("RITK_TEST")
            .expect("infallible: validated precondition"),
        host: "127.0.0.1".into(),
        port,
        max_pdu_length: 16384,
        timeout: Duration::from_secs(10),
        presentation_contexts: vec![RequestedPresentationContext {
            abstract_syntax_uid: ArrayString::from(CT_IMAGE_STORAGE)
                .expect("infallible: validated precondition"),
            transfer_syntax_uids: vec![ArrayString::from(transfer_syntax::IMPLICIT_VR_LE)
                .expect("infallible: validated precondition")],
        }],
        user_identity: None,
    }
}

/// Synthetic IVR-LE dataset carrying a single StudyDate tag.
///
/// Format: [group:u16-LE][element:u16-LE][length:u32-LE][value:bytes]
/// DicomTag (0008,0020) = "20240115" (8 bytes, even length).
fn synthetic_dataset() -> Vec<u8> {
    let mut buf = Vec::with_capacity(16);
    buf.extend_from_slice(&0x0008u16.to_le_bytes()); // group
    buf.extend_from_slice(&0x0020u16.to_le_bytes()); // element
    buf.extend_from_slice(&8u32.to_le_bytes()); // length
    buf.extend_from_slice(b"20240115"); // value
    buf
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Positive: SCP receives a single C-STORE-RQ and exposes it via `try_recv`.
///
/// Analytical basis:
/// - SCU connects, calls `c_store`, receives C-STORE-RSP Success (0x0000).
/// - SCP emits `StoredInstance` on the channel.
/// - `try_recv` returns the instance; all fields are verified by value.
#[test]
fn test_store_scp_single_instance_received() {
    let handle = StoreScp::start(ScpConfig {
        port: 0,
        queue_capacity: 8,
        ..ScpConfig::default()
    })
    .expect("SCP start");
    let port = handle.port();
    assert_ne!(port, 0, "actual_port must be assigned by OS");

    let dataset = synthetic_dataset();
    const INSTANCE_UID: &str = "1.2.3.4.5.6.7.101";

    let mut assoc = Association::connect(scu_config(port)).expect("SCU connect");

    let status = assoc
        .c_store(CT_IMAGE_STORAGE, INSTANCE_UID, dataset.clone())
        .expect("c_store");
    assert_eq!(
        status,
        DimseStatus::Success as u16,
        "C-STORE-RSP must be Success (0x0000), got 0x{status:04X}"
    );
    assoc.release().expect("SCU release");

    // Block until the SCP thread delivers (it runs on a separate thread).
    let inst = handle
        .recv_timeout(Duration::from_secs(2))
        .expect("SCP must deliver instance within 2s");

    assert_eq!(
        inst.sop_class_uid.as_str(),
        CT_IMAGE_STORAGE,
        "sop_class_uid"
    );
    assert_eq!(
        inst.sop_instance_uid.as_str(),
        INSTANCE_UID,
        "sop_instance_uid"
    );
    assert_eq!(
        inst.dataset_bytes, dataset,
        "dataset_bytes must match exactly"
    );
    assert_eq!(
        inst.transfer_syntax_uid.as_str(),
        transfer_syntax::IMPLICIT_VR_LE,
        "transfer_syntax_uid must be the negotiated IVR-LE"
    );
}

/// Positive: SCP receives two C-STORE-RQs on the same association.
///
/// Analytical basis:
/// - `Association::c_store` does not release the association — caller may
///   call it multiple times before `release()`.
/// - SCP must process all messages in the message loop before releasing.
#[test]
fn test_store_scp_multiple_instances_same_association() {
    let handle = StoreScp::start(ScpConfig {
        port: 0,
        queue_capacity: 16,
        ..ScpConfig::default()
    })
    .expect("SCP start");
    let port = handle.port();

    let dataset_a = synthetic_dataset();
    let dataset_b = {
        let mut b = synthetic_dataset();
        b[8..16].copy_from_slice(b"20241231"); // different date
        b
    };

    const UID_A: &str = "1.2.3.4.5.6.7.201";
    const UID_B: &str = "1.2.3.4.5.6.7.202";

    let mut assoc = Association::connect(scu_config(port)).expect("SCU connect");

    let s1 = assoc
        .c_store(CT_IMAGE_STORAGE, UID_A, dataset_a.clone())
        .expect("c_store A");
    assert_eq!(
        s1,
        DimseStatus::Success as u16,
        "first C-STORE-RSP must be Success"
    );

    let s2 = assoc
        .c_store(CT_IMAGE_STORAGE, UID_B, dataset_b.clone())
        .expect("c_store B");
    assert_eq!(
        s2,
        DimseStatus::Success as u16,
        "second C-STORE-RSP must be Success"
    );

    assoc.release().expect("SCU release");

    let inst_a = handle
        .recv_timeout(Duration::from_secs(2))
        .expect("must receive first instance");
    let inst_b = handle
        .recv_timeout(Duration::from_secs(2))
        .expect("must receive second instance");

    // Order of arrival preserves send order.
    assert_eq!(
        inst_a.sop_instance_uid.as_str(),
        UID_A,
        "first instance UID"
    );
    assert_eq!(inst_a.dataset_bytes, dataset_a, "first instance dataset");
    assert_eq!(
        inst_b.sop_instance_uid.as_str(),
        UID_B,
        "second instance UID"
    );
    assert_eq!(inst_b.dataset_bytes, dataset_b, "second instance dataset");
}

/// Positive: SCP assigns an OS port when `port = 0`; `actual_port` is non-zero.
///
/// Analytical basis: `TcpListener::bind("0.0.0.0:0")` followed by
/// `local_addr().port()` returns a non-zero ephemeral port.
#[test]
fn test_store_scp_ephemeral_port_is_nonzero() {
    let handle = StoreScp::start(ScpConfig {
        port: 0,
        ..ScpConfig::default()
    })
    .expect("SCP start with port=0");
    assert_ne!(handle.port(), 0, "ephemeral port must be non-zero");
    assert_eq!(handle.ae_title(), "RITKSNAP", "ae_title must match config");
}

// ── Unit tests for SCP-LOAD-01 ───────────────────────────────────────────────

#[test]
fn received_dataset_uses_the_native_file_header() {
    // PS3.10 §7 places an Explicit VR Little Endian UL group length after DICM.
    // Independent Explicit VR Little Endian payload for Image Position (Patient).
    let dataset_bytes = b"\x20\x00\x32\x00DS\x06\x001\\2\\3 ".to_vec();
    for uid in ["1.2.3", "1.2.34"] {
        let inst = super::StoredInstance {
            sop_class_uid: ArrayString::from("1.2.840.10008.5.1.4.1.1.2")
                .expect("valid DIMSE fixture"),
            sop_instance_uid: ArrayString::from(uid).expect("valid DIMSE fixture"),
            dataset_bytes: dataset_bytes.clone(),
            transfer_syntax_uid: ArrayString::from("1.2.840.10008.1.2.1")
                .expect("valid DIMSE fixture"),
        };
        let bytes = inst
            .make_part10_bytes()
            .expect("valid association metadata");
        assert_eq!(&bytes[..128], &[0u8; 128]);
        assert_eq!(&bytes[128..140], b"DICM\x02\x00\x00\x00UL\x04\x00");
        let size =
            u32::from_le_bytes(bytes[140..144].try_into().expect("valid DIMSE fixture")) as usize;
        assert_eq!(&bytes[144 + size..], dataset_bytes);
        let tag = bytes[144..144 + size]
            .windows(6)
            .position(|w| w == b"\x02\x00\x03\x00UI")
            .expect("file meta SOP instance UID")
            + 144;
        let length = usize::from(u16::from_le_bytes(
            bytes[tag + 6..tag + 8]
                .try_into()
                .expect("valid DIMSE fixture"),
        ));
        assert_eq!(length, uid.len() + uid.len() % 2);
        assert_eq!(&bytes[tag + 8..tag + 8 + uid.len()], uid.as_bytes());
        assert!(bytes[tag + 8 + uid.len()..tag + 8 + length]
            .iter()
            .all(|&byte| byte == 0));
        let file =
            ritk_dicom::dataset::DicomFile::parse(bytes.into(), &ritk_dicom::ParseBudget::DEFAULT)
                .expect("received dataset must parse through the same provider");
        assert_eq!(file.meta().media_storage_sop_instance_uid(), uid);
        assert_eq!(
            file.element(ritk_dicom::tags::IMAGE_POSITION_PATIENT)
                .expect("valid DIMSE fixture")
                .to_numbers::<f64>()
                .expect("valid DIMSE fixture"),
            [1.0, 2.0, 3.0]
        );
    }
}
