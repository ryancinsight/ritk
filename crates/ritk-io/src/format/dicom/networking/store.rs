//! C-STORE SCU — DICOM Storage Service Class (PS3.4 §B).

use super::association::{networking_error, Association, NetworkingError, StoreResponse};
use super::context::AssociationConfig;
use ritk_dicom::{tags, TransferSyntaxKind};
use std::path::Path;

/// Send a DICOM object file via C-STORE to the configured PACS endpoint.
pub fn store(config: &AssociationConfig, path: &Path) -> Result<StoreResponse, NetworkingError> {
    let file_obj = ritk_dicom::dataset::DicomFile::open(path)
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

    let source_syntax = TransferSyntaxKind::from_uid(file_obj.meta().transfer_syntax());
    let syntax = if source_syntax.is_encapsulated() {
        source_syntax
    } else {
        TransferSyntaxKind::ExplicitVrLittleEndian
    };
    let mut dataset_bytes = Vec::new();
    file_obj
        .write_dataset(&mut dataset_bytes, &syntax)
        .map_err(|error| NetworkingError::Protocol(format!("failed to encode dataset: {error}")))?;

    let mut association = Association::for_service(config, &sop_class_uid, syntax.uid())?;
    let response = association
        .store_response(
            &sop_class_uid,
            &sop_instance_uid,
            dataset_bytes,
            &[syntax.uid()],
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
