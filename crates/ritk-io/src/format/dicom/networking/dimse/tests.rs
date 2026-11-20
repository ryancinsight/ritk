use super::*;

#[test]
fn test_c_echo_rq_encode_decode() {
    let msg = DimseMessage::c_echo_rq(7);
    let encoded = msg.encode_command_set();
    let decoded =
        DimseMessage::decode_command_set(&encoded).expect("infallible: validated precondition");
    assert_eq!(decoded.command_field(), Some(CommandField::CEchoRq));
    assert_eq!(decoded.message_id(), Some(7));
    assert_eq!(
        decoded.affected_sop_class_uid().as_deref(),
        Some(sop_class::VERIFICATION_SOP_CLASS)
    );
}

#[test]
fn test_c_echo_rsp_status() {
    let msg = DimseMessage::c_echo_rsp(7, DimseStatus::Success as u16);
    assert_eq!(msg.status(), Some(0x0000));
    assert_eq!(msg.command_field(), Some(CommandField::CEchoRsp));
}

#[test]
fn test_c_find_rq_has_data_set() {
    let identifier = vec![0x08, 0x00, 0x52, 0x00];
    let msg = DimseMessage::c_find_rq(1, sop_class::FIND_STUDY, identifier.clone());
    assert_eq!(msg.command_data_set_type(), Some(HAS_DATASET));
    // The identifier must survive construction verbatim: `is_some` would hold
    // for an empty or substituted data set while the command still claims one.
    assert_eq!(msg.data_set.as_deref(), Some(identifier.as_slice()));
    let encoded = msg.encode_command_set();
    let decoded =
        DimseMessage::decode_command_set(&encoded).expect("infallible: validated precondition");
    assert_eq!(decoded.command_field(), Some(CommandField::CFindRq));
}

#[test]
fn test_c_store_rq_round_trip() {
    let ds = vec![0xDE, 0xAD, 0xBE, 0xEF];
    let msg = DimseMessage::c_store_rq(42, "1.2.840.10008.5.1.4.1.1.2", "1.2.3.4.5.6", 0x0000, ds);
    let encoded = msg.encode_command_set();
    let decoded =
        DimseMessage::decode_command_set(&encoded).expect("infallible: validated precondition");
    assert_eq!(
        decoded.affected_sop_class_uid().as_deref(),
        Some("1.2.840.10008.5.1.4.1.1.2")
    );
    assert_eq!(decoded.message_id(), Some(42));
    assert_eq!(decoded.command_field(), Some(CommandField::CStoreRq));
}

#[test]
fn test_c_move_rq_encode() {
    let identifier = vec![0x00, 0x01, 0x02];
    let msg = DimseMessage::c_move_rq(5, sop_class::MOVE_STUDY, "PACS_SCP", identifier.clone());
    assert_eq!(msg.move_destination().as_deref(), Some("PACS_SCP"));
    assert_eq!(msg.command_field(), Some(CommandField::CMoveRq));
    assert_eq!(msg.data_set.as_deref(), Some(identifier.as_slice()));
}

#[test]
fn test_command_group_length() {
    let msg = DimseMessage::c_echo_rq(1);
    let encoded = msg.encode_command_set();
    let decoded =
        DimseMessage::decode_command_set(&encoded).expect("infallible: validated precondition");

    let gl_elem = decoded
        .find_element(TAG_CMD_GROUP_LENGTH)
        .expect("group length element present");
    let group_length = u32::from_le_bytes([
        gl_elem.value.as_bytes()[0],
        gl_elem.value.as_bytes()[1],
        gl_elem.value.as_bytes()[2],
        gl_elem.value.as_bytes()[3],
    ]);

    assert_eq!(&encoded[..8], &[0, 0, 0, 0, 4, 0, 0, 0]);
    assert_eq!(group_length as usize, encoded.len() - 12);
    assert_eq!(&encoded[12..16], &[0, 0, 2, 0]);
    assert!(DimseMessage::decode_command_set(&encoded[..encoded.len() - 1]).is_err());
}

#[test]
fn test_sop_class_uids() {
    let uids = [
        sop_class::VERIFICATION_SOP_CLASS,
        sop_class::FIND_STUDY,
        sop_class::FIND_PATIENT,
        sop_class::STUDY_ROOT_FIND_SOP_CLASS,
        sop_class::FIND_INSTANCE,
        sop_class::MOVE_STUDY,
        sop_class::MOVE_PATIENT,
        sop_class::STUDY_ROOT_MOVE_SOP_CLASS,
        sop_class::GET_STUDY,
        sop_class::STORAGE_COMMITMENT,
    ];
    for uid in &uids {
        assert!(!uid.is_empty(), "UID must not be empty");
        assert!(
            uid.as_bytes()[0].is_ascii_digit(),
            "UID must start with digit: {}",
            uid
        );
        assert!(
            uid.chars().all(|c| c.is_ascii_digit() || c == '.'),
            "UID must contain only digits and dots: {}",
            uid
        );
    }
}

#[test]
fn test_pending_status_decode() {
    let msg = DimseMessage::c_find_rsp(3, sop_class::FIND_STUDY, 0xFF00, None);
    let encoded = msg.encode_command_set();
    let decoded =
        DimseMessage::decode_command_set(&encoded).expect("infallible: validated precondition");
    assert_eq!(decoded.status(), Some(0xFF00));
    assert_eq!(decoded.command_field(), Some(CommandField::CFindRsp));
    assert!(decoded.data_set.is_none());
}
