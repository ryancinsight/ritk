//! Association unit tests (extracted from association.rs per 500-line structural limit).

use crate::format::dicom::networking::association::Association;
use crate::format::dicom::networking::context::{
    transfer_syntax, AssociationConfig, NegotiatedContext, RequestedPresentationContext,
};
use crate::format::dicom::networking::dimse::sop_class;
use crate::format::dicom::networking::pdu::{
    AssociateAcPdu, CommandType, FragmentPosition, Pdu, PresentationContextItemAc, UserInformation,
    APPLICATION_CONTEXT_NAME,
};
use arrayvec::ArrayString;
use std::net::TcpListener;
use std::net::TcpStream;

fn rpc(uid: &str, ts: &[&str]) -> RequestedPresentationContext {
    RequestedPresentationContext {
        abstract_syntax_uid: ArrayString::from(uid).unwrap(),
        transfer_syntax_uids: ts.iter().map(|s| ArrayString::from(s).unwrap()).collect(),
    }
}

#[test]
fn test_config_default() {
    let c = AssociationConfig::default();
    assert_eq!((c.port, c.max_pdu_length), (104, 16384));
    assert_eq!(c.called_ae_title.as_str(), "ANYSCP");
    assert_eq!(c.calling_ae_title.as_str(), "RITK");
}

#[test]
fn test_requested_context_odd_ids() {
    let cfg = AssociationConfig {
        presentation_contexts: vec![
            rpc(sop_class::VERIFICATION, &[transfer_syntax::IMPLICIT_VR_LE]),
            rpc(sop_class::FIND_STUDY, &[transfer_syntax::IMPLICIT_VR_LE]),
            rpc(sop_class::MOVE_STUDY, &[transfer_syntax::IMPLICIT_VR_LE]),
        ],
        ..Default::default()
    };
    if let Pdu::AssociateRq(ref rq) = Association::build_associate_rq(&cfg) {
        let ids: Vec<u8> = rq
            .presentation_contexts
            .iter()
            .map(|pc| pc.presentation_context_id)
            .collect();
        assert_eq!(ids, vec![1, 3, 5]);
        assert!(ids.iter().all(|id| id % 2 == 1));
    }
}

#[test]
fn test_build_associate_rq() {
    let cfg = AssociationConfig {
        called_ae_title: ArrayString::from("TESTSCP").unwrap(),
        presentation_contexts: vec![rpc(
            sop_class::VERIFICATION,
            &[transfer_syntax::EXPLICIT_VR_LE],
        )],
        ..Default::default()
    };
    let pdu = Association::build_associate_rq(&cfg);
    if let Pdu::AssociateRq(ref rq) = pdu {
        assert_eq!(rq.presentation_contexts.len(), 1);
        assert_eq!(rq.called_ae_title.as_str(), "TESTSCP");
        assert_eq!(rq.presentation_contexts[0].transfer_syntax_uids.len(), 2);
        assert_eq!(
            rq.presentation_contexts[0].transfer_syntax_uids[1].as_str(),
            transfer_syntax::IMPLICIT_VR_LE
        );
    }
    assert_eq!(pdu, Pdu::decode(&pdu.encode()).unwrap());
}

#[test]
fn test_fragment_pdv_single() {
    let pdvs = Association::fragment_pdvs(&[0xABu8; 100], CommandType::Command, 16378);
    assert_eq!(pdvs.len(), 1);
    assert!(pdvs[0].message_control_header.fragment_position == FragmentPosition::Last);
    assert_eq!(pdvs[0].data.len(), 100);
}

#[test]
fn test_fragment_pdv_multiple() {
    let pdvs = Association::fragment_pdvs(&vec![0xCDu8; 40000], CommandType::DataSet, 8186);
    assert!(pdvs.len() >= 5);
    for (i, p) in pdvs.iter().enumerate() {
        assert_eq!(
            p.message_control_header.fragment_position == FragmentPosition::Last,
            i == pdvs.len() - 1
        );
    }
    assert_eq!(pdvs.iter().map(|p| p.data.len()).sum::<usize>(), 40000);
}

#[test]
fn test_find_context() {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let a = Association {
        stream: TcpStream::connect(l.local_addr().unwrap()).unwrap(),
        config: AssociationConfig::default(),
        negotiated_contexts: vec![
            NegotiatedContext {
                presentation_context_id: 1,
                abstract_syntax_uid: ArrayString::from(sop_class::VERIFICATION).unwrap(),
                transfer_syntax_uid: ArrayString::from(transfer_syntax::IMPLICIT_VR_LE).unwrap(),
            },
            NegotiatedContext {
                presentation_context_id: 3,
                abstract_syntax_uid: ArrayString::from(sop_class::FIND_STUDY).unwrap(),
                transfer_syntax_uid: ArrayString::from(transfer_syntax::EXPLICIT_VR_LE).unwrap(),
            },
        ],
        next_context_id: 7,
        remote_max_pdu_length: 16384,
        state: crate::format::dicom::networking::association::DicomAssociationState::Active,
    };
    assert_eq!(
        a.find_context(sop_class::VERIFICATION)
            .unwrap()
            .presentation_context_id,
        1
    );
    assert!(a.find_context("1.2.3.4.5.6").is_none());
}

#[test]
fn test_negotiated_context_from_ac() {
    let ac = AssociateAcPdu {
        protocol_version: 1,
        called_ae_title: ArrayString::from("SCP").unwrap(),
        calling_ae_title: ArrayString::from("RITK").unwrap(),
        application_context_name: ArrayString::from(APPLICATION_CONTEXT_NAME).unwrap(),
        presentation_contexts: vec![
            PresentationContextItemAc {
                presentation_context_id: 1,
                result_reason: 0,
                transfer_syntax_uid: ArrayString::from(transfer_syntax::IMPLICIT_VR_LE).unwrap(),
            },
            PresentationContextItemAc {
                presentation_context_id: 3,
                result_reason: 1,
                transfer_syntax_uid: ArrayString::from(transfer_syntax::EXPLICIT_VR_LE).unwrap(),
            },
        ],
        user_information: UserInformation::default(),
    };
    let mut m = std::collections::HashMap::new();
    m.insert(1u8, ArrayString::from(sop_class::VERIFICATION).unwrap());
    m.insert(3u8, ArrayString::from(sop_class::FIND_STUDY).unwrap());
    let n = Association::negotiated_contexts_from_ac(&ac, &m);
    assert_eq!(n.len(), 1);
    assert_eq!(
        n[0],
        NegotiatedContext {
            presentation_context_id: 1,
            abstract_syntax_uid: ArrayString::from(sop_class::VERIFICATION).unwrap(),
            transfer_syntax_uid: ArrayString::from(transfer_syntax::IMPLICIT_VR_LE).unwrap()
        }
    );
}

#[test]
fn test_transfer_syntax_uids() {
    for uid in [
        transfer_syntax::IMPLICIT_VR_LE,
        transfer_syntax::EXPLICIT_VR_LE,
        transfer_syntax::EXPLICIT_VR_BE,
        transfer_syntax::JPEG_BASELINE,
        transfer_syntax::JPEG_LOSSLESS,
        transfer_syntax::JPEG_LS_LOSSLESS,
        transfer_syntax::JPEG_2000_LOSSLESS,
        transfer_syntax::JPEG_2000,
    ] {
        assert!(!uid.is_empty());
        assert!(
            uid.starts_with("1.2.840.10008"),
            "{} must start with 1.2.840.10008",
            uid
        );
    }
}
