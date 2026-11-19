//! C-STORE SCU — DICOM Storage Service Class (PS3.4 §B).

use super::association::{networking_error, Association, NetworkingError, StoreResponse};
use super::command::EXPLICIT_VR_LE_TS;
use super::context::AssociationConfig;
use dicom::dictionary_std::tags;
use dicom_transfer_syntax_registry::entries::EXPLICIT_VR_LITTLE_ENDIAN;
use std::path::Path;

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

    let mut association = Association::for_service(config, &sop_class_uid, EXPLICIT_VR_LE_TS)?;
    let response = association
        .store_response(
            &sop_class_uid,
            &sop_instance_uid,
            dataset_bytes,
            &[EXPLICIT_VR_LE_TS],
        )
        .map_err(networking_error)?;
    let status = response
        .status()
        .ok_or_else(|| NetworkingError::Protocol("missing Status".to_owned()))?;
    association.release().map_err(networking_error)?;
    Ok(StoreResponse {
        status,
        affected_sop_instance_uid: response.affected_sop_instance_uid(),
    })
}
