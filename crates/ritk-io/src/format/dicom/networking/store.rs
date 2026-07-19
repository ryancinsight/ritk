//! C-STORE SCU â€” DICOM Storage Service Class (PS3.4 Â§B).

use super::association::{release_client_association, NetworkingError, StoreResponse};
use super::command::{
    build_command_pdu, parse_command_response, CommandElementValue, C_STORE_RQ, C_STORE_RSP,
    EXPLICIT_VR_LE_TS, HAS_DATASET, PRIORITY_MEDIUM,
};
use super::context::AssociationConfig;
use super::echo::{find_ctx_id, receive_command_pdv};
use dicom::dictionary_std::tags;
use dicom_transfer_syntax_registry::entries::EXPLICIT_VR_LITTLE_ENDIAN;
use dicom_ul::association::client::ClientAssociationOptions;
use dicom_ul::pdu::{PDataValue, PDataValueType, Pdu};
use std::net::TcpStream;
use std::path::Path;

const MAX_FRAGMENT_SIZE: usize = 16_384;

/// Send a DICOM object file via C-STORE to the configured PACS endpoint.
pub fn store(config: &AssociationConfig, path: &Path) -> Result<StoreResponse, NetworkingError> {
    let file_obj = dicom::object::open_file(path)
        .map_err(|e| NetworkingError::Protocol(format!("failed to open DICOM file: {}", e)))?;

    let sop_class_uid = file_obj
        .element(tags::SOP_CLASS_UID)
        .ok()
        .and_then(|e| e.to_str().ok().map(|s| s.to_string()))
        .ok_or_else(|| NetworkingError::Protocol("missing SOP Class UID in file".to_owned()))?;

    let sop_instance_uid = file_obj
        .element(tags::SOP_INSTANCE_UID)
        .ok()
        .and_then(|e| e.to_str().ok().map(|s| s.to_string()))
        .ok_or_else(|| NetworkingError::Protocol("missing SOP Instance UID in file".to_owned()))?;

    let inner: &dicom::object::InMemDicomObject = &file_obj;
    let ts = EXPLICIT_VR_LITTLE_ENDIAN.erased();
    let mut dataset_bytes = Vec::with_capacity(4096);
    inner
        .write_dataset_with_ts(&mut dataset_bytes, &ts)
        .map_err(|e| NetworkingError::Protocol(format!("failed to encode dataset: {}", e)))?;

    let mut assoc = ClientAssociationOptions::new()
        .calling_ae_title(config.calling_ae_title.as_str())
        .called_ae_title(config.called_ae_title.as_str())
        .with_presentation_context(sop_class_uid.as_str(), vec![EXPLICIT_VR_LE_TS])
        .establish(format!("{}:{}", config.host, config.port))
        .map_err(|e| NetworkingError::Protocol(e.to_string()))?;

    let ctx_id = find_ctx_id(&assoc)?;

    let cmd_bytes = build_command_pdu(&[
        (0x0000_0002, CommandElementValue::Ui(&sop_class_uid)),
        (0x0000_0100, CommandElementValue::Us(C_STORE_RQ)),
        (0x0000_0110, CommandElementValue::Us(1u16)),
        (0x0000_0700, CommandElementValue::Us(PRIORITY_MEDIUM)),
        (0x0000_0800, CommandElementValue::Us(HAS_DATASET)),
        (0x0000_1000, CommandElementValue::Ui(&sop_instance_uid)),
    ]);

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

    send_fragmented_data(&mut assoc, ctx_id, &dataset_bytes)?;

    let rsp_bytes = receive_command_pdv(&mut assoc, ctx_id)?;
    let rsp = parse_command_response(&rsp_bytes)?;

    if rsp.command_field != C_STORE_RSP {
        return Err(NetworkingError::Protocol(format!(
            "expected C-STORE-RSP (0x{:04X}), got 0x{:04X}",
            C_STORE_RSP, rsp.command_field
        )));
    }

    release_client_association(assoc)?;

    Ok(StoreResponse {
        status: rsp.status,
        affected_sop_instance_uid: rsp.affected_sop_instance_uid,
    })
}

pub(super) fn send_fragmented_data(
    assoc: &mut dicom_ul::association::client::ClientAssociation<TcpStream>,
    ctx_id: u8,
    data: &[u8],
) -> Result<(), NetworkingError> {
    if data.is_empty() {
        assoc
            .send(&Pdu::PData {
                data: vec![PDataValue {
                    presentation_context_id: ctx_id,
                    value_type: PDataValueType::Data,
                    is_last: true,
                    data: Vec::new(),
                }],
            })
            .map_err(|e| NetworkingError::Protocol(e.to_string()))?;
        return Ok(());
    }

    let chunks: Vec<&[u8]> = data.chunks(MAX_FRAGMENT_SIZE).collect();
    let last_idx = chunks.len() - 1;
    for (i, chunk) in chunks.iter().enumerate() {
        assoc
            .send(&Pdu::PData {
                data: vec![PDataValue {
                    presentation_context_id: ctx_id,
                    value_type: PDataValueType::Data,
                    is_last: i == last_idx,
                    data: chunk.to_vec(),
                }],
            })
            .map_err(|e| NetworkingError::Protocol(e.to_string()))?;
    }
    Ok(())
}
