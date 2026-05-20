use super::*;

fn sample_rq() -> AssociateRqPdu {
    AssociateRqPdu {
        protocol_version: 1, called_ae_title: "SCP".into(), calling_ae_title: "SCU".into(),
        application_context_name: APPLICATION_CONTEXT_NAME.into(),
        presentation_contexts: vec![PresentationContextItemRq {
            presentation_context_id: 1, abstract_syntax_uid: "1.2.840.10008.1.1".into(),
            transfer_syntax_uids: vec!["1.2.840.10008.1.2".into()],
        }],
        user_information: UserInformation {
            maximum_length: MaximumLengthSubItem { maximum_length_received: 16384 },
            implementation_class_uid: ImplementationClassUidSubItem {
                implementation_class_uid: RITK_IMPLEMENTATION_CLASS_UID.into(),
            },
            implementation_version_name: Some(ImplementationVersionNameSubItem {
                implementation_version_name: RITK_IMPLEMENTATION_VERSION.into(),
            }),
            ..Default::default()
        },
    }
}

#[test]
fn test_associate_rq_round_trip() {
    let pdu = Pdu::AssociateRq(sample_rq());
    let enc = pdu.encode();
    let dec = Pdu::decode(&enc).unwrap();
    assert_eq!(pdu, dec);
}

#[test]
fn test_pdu_type_byte() {
    let check = |pdu: Pdu, expected: u8| { assert_eq!(pdu.encode()[0], expected); };
    check(Pdu::AssociateRq(sample_rq()), 0x01);
    let ac = AssociateAcPdu {
        protocol_version: 1, called_ae_title: "SCP".into(), calling_ae_title: "SCU".into(),
        application_context_name: APPLICATION_CONTEXT_NAME.into(),
        presentation_contexts: vec![PresentationContextItemAc {
            presentation_context_id: 1, result_reason: 0,
            transfer_syntax_uid: "1.2.840.10008.1.2".into(),
        }],
        user_information: UserInformation::default(),
    };
    check(Pdu::AssociateAc(ac), 0x02);
    check(Pdu::AssociateRj(AssociateRjPdu {
        result: AssociationRejectResult::RejectedTransient,
        source: AssociationRejectSource::DicomUlServiceUser, reason: 1,
    }), 0x03);
    check(Pdu::PDataTf(PDataTfPdu { presentation_data_value_items: vec![] }), 0x04);
    check(Pdu::ReleaseRq(ReleaseRqPdu), 0x05);
    check(Pdu::ReleaseRp(ReleaseRpPdu), 0x06);
    check(Pdu::Abort(AbortPdu { source: AbortSource::DicomUlServiceUser }), 0x07);
}

#[test]
fn test_ae_title_padding() {
    let pdu = Pdu::AssociateRq(sample_rq());
    let enc = pdu.encode();
    // AE titles at offset 10 (6-byte header + 2 version + 2 reserved)
    let called = &enc[10..26]; let calling = &enc[26..42];
    assert_eq!(&called[..3], b"SCP"); assert!(called[3..].iter().all(|&b| b == b' '));
    assert_eq!(&calling[..3], b"SCU"); assert!(calling[3..].iter().all(|&b| b == b' '));
    let dec = Pdu::decode(&enc).unwrap();
    if let Pdu::AssociateRq(rq) = dec {
        assert_eq!(rq.called_ae_title, "SCP");
        assert_eq!(rq.calling_ae_title, "SCU");
    }
}

#[test]
fn test_p_data_tf_round_trip() {
    let pdu = Pdu::PDataTf(PDataTfPdu { presentation_data_value_items: vec![
        PresentationDataValueItem {
            presentation_context_id: 1,
            message_control_header: MessageControlHeader { message_type: CommandType::Command, last_fragment: true },
            data: vec![0x00, 0x00, 0x00, 0x01],
        },
        PresentationDataValueItem {
            presentation_context_id: 1,
            message_control_header: MessageControlHeader { message_type: CommandType::DataSet, last_fragment: false },
            data: vec![0xDE, 0xAD, 0xBE, 0xEF],
        },
        PresentationDataValueItem {
            presentation_context_id: 3,
            message_control_header: MessageControlHeader { message_type: CommandType::DataSet, last_fragment: true },
            data: (0..128).map(|i| i as u8).collect(),
        },
    ]});
    let enc = pdu.encode(); let dec = Pdu::decode(&enc).unwrap(); assert_eq!(pdu, dec);
}

#[test]
fn test_release_round_trip() {
    let rq = Pdu::ReleaseRq(ReleaseRqPdu); let enc = rq.encode();
    assert_eq!(enc.len(), 6);
    assert_eq!(Pdu::decode(&enc).unwrap(), rq);
    let rp = Pdu::ReleaseRp(ReleaseRpPdu);
    assert_eq!(Pdu::decode(&rp.encode()).unwrap(), rp);
}

#[test]
fn test_abort_round_trip() {
    let pdu = Pdu::Abort(AbortPdu { source: AbortSource::DicomUlServiceProviderAcse });
    assert_eq!(Pdu::decode(&pdu.encode()).unwrap(), pdu);
}

#[test]
fn test_associate_rj_round_trip() {
    let pdu = Pdu::AssociateRj(AssociateRjPdu {
        result: AssociationRejectResult::RejectedPermanent,
        source: AssociationRejectSource::DicomUlServiceProviderPresentation, reason: 3,
    });
    assert_eq!(Pdu::decode(&pdu.encode()).unwrap(), pdu);
}

#[test]
fn test_maximum_length_default() {
    assert_eq!(MaximumLengthSubItem::default().maximum_length_received, 16384);
    assert_eq!(DEFAULT_MAXIMUM_LENGTH, 16384);
}
