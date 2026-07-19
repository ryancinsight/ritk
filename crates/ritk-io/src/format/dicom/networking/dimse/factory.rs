//! DIMSE message factory methods per DICOM PS 3.7.

use super::*;

// ── Factory methods ───────────────────────────────────────────────────────────

impl DimseMessage {
    /// C-ECHO-RQ. PS3.7 §9.1.5.1.
    pub fn c_echo_rq(message_id: u16) -> Self {
        Self {
            command_set: vec![
                CommandElement {
                    tag: TAG_AFFECTED_SOP_CLASS,
                    vr: CommandVr::Ui,
                    value: CommandValue::ui(sop_class::VERIFICATION),
                },
                CommandElement {
                    tag: TAG_COMMAND_FIELD,
                    vr: CommandVr::Us,
                    value: CommandValue::us(CommandField::CEchoRq as u16),
                },
                CommandElement {
                    tag: TAG_MESSAGE_ID,
                    vr: CommandVr::Us,
                    value: CommandValue::us(message_id),
                },
                CommandElement {
                    tag: TAG_CMD_DATA_SET_TYPE,
                    vr: CommandVr::Us,
                    value: CommandValue::us(NO_DATASET),
                },
            ],
            data_set: None,
        }
    }

    /// C-ECHO-RSP. PS3.7 §9.1.5.2.
    pub fn c_echo_rsp(message_id: u16, status: u16) -> Self {
        Self {
            command_set: vec![
                CommandElement {
                    tag: TAG_AFFECTED_SOP_CLASS,
                    vr: CommandVr::Ui,
                    value: CommandValue::ui(sop_class::VERIFICATION),
                },
                CommandElement {
                    tag: TAG_COMMAND_FIELD,
                    vr: CommandVr::Us,
                    value: CommandValue::us(CommandField::CEchoRsp as u16),
                },
                CommandElement {
                    tag: TAG_MESSAGE_ID_RESP,
                    vr: CommandVr::Us,
                    value: CommandValue::us(message_id),
                },
                CommandElement {
                    tag: TAG_CMD_DATA_SET_TYPE,
                    vr: CommandVr::Us,
                    value: CommandValue::us(NO_DATASET),
                },
                CommandElement {
                    tag: TAG_STATUS,
                    vr: CommandVr::Us,
                    value: CommandValue::us(status),
                },
            ],
            data_set: None,
        }
    }

    /// C-FIND-RQ. PS3.7 §9.1.2.1.
    pub fn c_find_rq(message_id: u16, sop_class_uid: &str, identifier: Vec<u8>) -> Self {
        Self {
            command_set: vec![
                CommandElement {
                    tag: TAG_AFFECTED_SOP_CLASS,
                    vr: CommandVr::Ui,
                    value: CommandValue::ui(sop_class_uid),
                },
                CommandElement {
                    tag: TAG_COMMAND_FIELD,
                    vr: CommandVr::Us,
                    value: CommandValue::us(CommandField::CFindRq as u16),
                },
                CommandElement {
                    tag: TAG_MESSAGE_ID,
                    vr: CommandVr::Us,
                    value: CommandValue::us(message_id),
                },
                CommandElement {
                    tag: TAG_PRIORITY,
                    vr: CommandVr::Us,
                    value: CommandValue::us(0x0000),
                },
                CommandElement {
                    tag: TAG_CMD_DATA_SET_TYPE,
                    vr: CommandVr::Us,
                    value: CommandValue::us(HAS_DATASET),
                },
                CommandElement {
                    tag: TAG_NUM_SUBOPS,
                    vr: CommandVr::Us,
                    value: CommandValue::us(1u16),
                },
            ],
            data_set: Some(identifier),
        }
    }

    /// C-FIND-RSP. PS3.7 §9.1.2.2.
    pub fn c_find_rsp(
        message_id: u16,
        sop_class_uid: &str,
        status: u16,
        identifier: Option<Vec<u8>>,
    ) -> Self {
        let has_ds = identifier.is_some();
        Self {
            command_set: vec![
                CommandElement {
                    tag: TAG_AFFECTED_SOP_CLASS,
                    vr: CommandVr::Ui,
                    value: CommandValue::ui(sop_class_uid),
                },
                CommandElement {
                    tag: TAG_COMMAND_FIELD,
                    vr: CommandVr::Us,
                    value: CommandValue::us(CommandField::CFindRsp as u16),
                },
                CommandElement {
                    tag: TAG_MESSAGE_ID_RESP,
                    vr: CommandVr::Us,
                    value: CommandValue::us(message_id),
                },
                CommandElement {
                    tag: TAG_CMD_DATA_SET_TYPE,
                    vr: CommandVr::Us,
                    value: CommandValue::us(if has_ds { HAS_DATASET } else { NO_DATASET }),
                },
                CommandElement {
                    tag: TAG_STATUS,
                    vr: CommandVr::Us,
                    value: CommandValue::us(status),
                },
            ],
            data_set: identifier,
        }
    }

    /// C-STORE-RQ. PS3.7 §9.1.1.1.
    pub fn c_store_rq(
        message_id: u16,
        sop_class_uid: &str,
        sop_instance_uid: &str,
        priority: u16,
        data_set: Vec<u8>,
    ) -> Self {
        Self {
            command_set: vec![
                CommandElement {
                    tag: TAG_AFFECTED_SOP_CLASS,
                    vr: CommandVr::Ui,
                    value: CommandValue::ui(sop_class_uid),
                },
                CommandElement {
                    tag: TAG_AFFECTED_SOP_INSTANCE,
                    vr: CommandVr::Ui,
                    value: CommandValue::ui(sop_instance_uid),
                },
                CommandElement {
                    tag: TAG_COMMAND_FIELD,
                    vr: CommandVr::Us,
                    value: CommandValue::us(CommandField::CStoreRq as u16),
                },
                CommandElement {
                    tag: TAG_MESSAGE_ID,
                    vr: CommandVr::Us,
                    value: CommandValue::us(message_id),
                },
                CommandElement {
                    tag: TAG_PRIORITY,
                    vr: CommandVr::Us,
                    value: CommandValue::us(priority),
                },
                CommandElement {
                    tag: TAG_CMD_DATA_SET_TYPE,
                    vr: CommandVr::Us,
                    value: CommandValue::us(HAS_DATASET),
                },
            ],
            data_set: Some(data_set),
        }
    }

    /// C-STORE-RSP. PS3.7 §9.1.1.2.
    pub fn c_store_rsp(
        message_id: u16,
        sop_class_uid: &str,
        sop_instance_uid: &str,
        status: u16,
    ) -> Self {
        Self {
            command_set: vec![
                CommandElement {
                    tag: TAG_AFFECTED_SOP_CLASS,
                    vr: CommandVr::Ui,
                    value: CommandValue::ui(sop_class_uid),
                },
                CommandElement {
                    tag: TAG_AFFECTED_SOP_INSTANCE,
                    vr: CommandVr::Ui,
                    value: CommandValue::ui(sop_instance_uid),
                },
                CommandElement {
                    tag: TAG_COMMAND_FIELD,
                    vr: CommandVr::Us,
                    value: CommandValue::us(CommandField::CStoreRsp as u16),
                },
                CommandElement {
                    tag: TAG_MESSAGE_ID_RESP,
                    vr: CommandVr::Us,
                    value: CommandValue::us(message_id),
                },
                CommandElement {
                    tag: TAG_CMD_DATA_SET_TYPE,
                    vr: CommandVr::Us,
                    value: CommandValue::us(NO_DATASET),
                },
                CommandElement {
                    tag: TAG_STATUS,
                    vr: CommandVr::Us,
                    value: CommandValue::us(status),
                },
            ],
            data_set: None,
        }
    }

    /// C-MOVE-RQ. PS3.7 §9.1.3.1.
    pub fn c_move_rq(
        message_id: u16,
        sop_class_uid: &str,
        move_destination: &str,
        identifier: Vec<u8>,
    ) -> Self {
        Self {
            command_set: vec![
                CommandElement {
                    tag: TAG_AFFECTED_SOP_CLASS,
                    vr: CommandVr::Ui,
                    value: CommandValue::ui(sop_class_uid),
                },
                CommandElement {
                    tag: TAG_COMMAND_FIELD,
                    vr: CommandVr::Us,
                    value: CommandValue::us(CommandField::CMoveRq as u16),
                },
                CommandElement {
                    tag: TAG_MESSAGE_ID,
                    vr: CommandVr::Us,
                    value: CommandValue::us(message_id),
                },
                CommandElement {
                    tag: TAG_MOVE_DESTINATION,
                    vr: CommandVr::Ae,
                    value: CommandValue::ae(move_destination),
                },
                CommandElement {
                    tag: TAG_PRIORITY,
                    vr: CommandVr::Us,
                    value: CommandValue::us(0x0000),
                },
                CommandElement {
                    tag: TAG_CMD_DATA_SET_TYPE,
                    vr: CommandVr::Us,
                    value: CommandValue::us(HAS_DATASET),
                },
            ],
            data_set: Some(identifier),
        }
    }

    /// C-MOVE-RSP. PS3.7 §9.1.3.2.
    pub fn c_move_rsp(
        message_id: u16,
        sop_class_uid: &str,
        status: u16,
        identifier: Option<Vec<u8>>,
    ) -> Self {
        let has_ds = identifier.is_some();
        Self {
            command_set: vec![
                CommandElement {
                    tag: TAG_AFFECTED_SOP_CLASS,
                    vr: CommandVr::Ui,
                    value: CommandValue::ui(sop_class_uid),
                },
                CommandElement {
                    tag: TAG_COMMAND_FIELD,
                    vr: CommandVr::Us,
                    value: CommandValue::us(CommandField::CMoveRsp as u16),
                },
                CommandElement {
                    tag: TAG_MESSAGE_ID_RESP,
                    vr: CommandVr::Us,
                    value: CommandValue::us(message_id),
                },
                CommandElement {
                    tag: TAG_CMD_DATA_SET_TYPE,
                    vr: CommandVr::Us,
                    value: CommandValue::us(if has_ds { HAS_DATASET } else { NO_DATASET }),
                },
                CommandElement {
                    tag: TAG_STATUS,
                    vr: CommandVr::Us,
                    value: CommandValue::us(status),
                },
            ],
            data_set: identifier,
        }
    }
}
