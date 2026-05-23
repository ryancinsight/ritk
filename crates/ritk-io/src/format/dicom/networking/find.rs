//! C-FIND SCU — Study Root Query/Retrieve: FIND (PS3.4 §C.4.1).

use super::association::{FindResult, NetworkingError};
use super::context::AssociationConfig;
use super::command::{
    build_command_pdu, build_dataset_ivr_le, encode_str,
    parse_command_response, CommandElementValue, C_FIND_RQ, C_FIND_RSP, HAS_DATASET,
    IMPLICIT_VR_LE_TS, NO_DATASET, PRIORITY_MEDIUM, STATUS_PENDING, STATUS_PENDING_WARN,
    STATUS_SUCCESS, STUDY_ROOT_FIND_SOP_CLASS,
};
use super::echo::{find_ctx_id, receive_command_pdv, receive_data_pdv};
use dicom_ul::association::client::ClientAssociationOptions;
use dicom_ul::pdu::{PDataValue, PDataValueType, Pdu};

/// DICOM query retrieve level for C-FIND (PS3.4 §C.3.4 Table C.3-1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FindLevel {
    Patient,
    Study,
    Series,
    Image,
}

impl FindLevel {
    fn as_cs_str(self) -> &'static str {
        match self {
            Self::Patient => "PATIENT",
            Self::Study => "STUDY",
            Self::Series => "SERIES",
            Self::Image => "IMAGE",
        }
    }
}

/// A C-FIND query dataset (key-value pairs for Study Root QR level).
#[derive(Debug, Clone)]
pub struct FindQuery {
    pub level: FindLevel,
    pub keys: Vec<(u16, u16, String)>,
}

impl FindQuery {
    pub fn new(level: FindLevel) -> Self {
        Self { level, keys: Vec::new() }
    }

    pub fn with_key(mut self, group: u16, element: u16, value: impl Into<String>) -> Self {
        self.keys.push((group, element, value.into()));
        self
    }
}

/// Execute a C-FIND SCU operation against the configured PACS endpoint.
pub fn find(
    config: &AssociationConfig,
    query: &FindQuery,
) -> Result<Vec<FindResult>, NetworkingError> {
    let mut assoc = ClientAssociationOptions::new()
        .calling_ae_title(config.calling_ae_title.as_str())
        .called_ae_title(config.called_ae_title.as_str())
        .with_presentation_context(STUDY_ROOT_FIND_SOP_CLASS, vec![IMPLICIT_VR_LE_TS])
        .establish(&format!("{}:{}", config.host, config.port))
        .map_err(|e| NetworkingError::Protocol(e.to_string()))?;

    let ctx_id = find_ctx_id(&assoc)?;

    let cmd_bytes = build_command_pdu(&[
        (0x0000_0002, CommandElementValue::Ui(STUDY_ROOT_FIND_SOP_CLASS)),
        (0x0000_0100, CommandElementValue::Us(C_FIND_RQ)),
        (0x0000_0110, CommandElementValue::Us(1u16)),
        (0x0000_0700, CommandElementValue::Us(PRIORITY_MEDIUM)),
        (0x0000_0800, CommandElementValue::Us(HAS_DATASET)),
    ]);

    let level_bytes = encode_str(query.level.as_cs_str());
    let mut dataset_elements: Vec<(u32, Vec<u8>)> = vec![(0x0008_0052, level_bytes)];
    for (group, element, value) in &query.keys {
        let tag = ((*group as u32) << 16) | (*element as u32);
        dataset_elements.push((tag, encode_str(value)));
    }
    dataset_elements.sort_by_key(|(tag, _)| *tag);
    let dataset_refs: Vec<(u32, &[u8])> = dataset_elements
        .iter()
        .map(|(tag, v)| (*tag, v.as_slice()))
        .collect();
    let dataset_bytes = build_dataset_ivr_le(&dataset_refs);

    assoc
        .send(&Pdu::PData {
            data: vec![PDataValue {
                presentation_context_id: ctx_id,
                value_type: PDataValueType::Command,
                is_last: true,
                data: cmd_bytes,
            }],
        })
        .map_err(|e| NetworkingError::Protocol(e.to_string()))?;

    assoc
        .send(&Pdu::PData {
            data: vec![PDataValue {
                presentation_context_id: ctx_id,
                value_type: PDataValueType::Data,
                is_last: true,
                data: dataset_bytes,
            }],
        })
        .map_err(|e| NetworkingError::Protocol(e.to_string()))?;

    let mut results = Vec::new();
    loop {
        let rsp_cmd_bytes = receive_command_pdv(&mut assoc, ctx_id)?;
        let rsp = parse_command_response(&rsp_cmd_bytes)?;

        if rsp.command_field != C_FIND_RSP {
            return Err(NetworkingError::Protocol(format!(
                "expected C-FIND-RSP (0x{:04X}), got 0x{:04X}",
                C_FIND_RSP, rsp.command_field
            )));
        }

        match rsp.status {
            STATUS_PENDING | STATUS_PENDING_WARN => {
                if rsp.data_set_type != NO_DATASET {
                    let data_bytes = receive_data_pdv(&mut assoc, ctx_id)?;
                    results.push(FindResult { matches: vec![data_bytes], status: rsp.status });
                }
            }
            STATUS_SUCCESS | 0xA700 | 0xA900 | 0xC000..=0xCFFF => break,
            other => return Err(NetworkingError::UnexpectedStatus(other)),
        }
    }

    assoc
        .release()
        .map_err(|e| NetworkingError::Protocol(e.to_string()))?;

    Ok(results)
}
