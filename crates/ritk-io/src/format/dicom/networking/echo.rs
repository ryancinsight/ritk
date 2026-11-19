//! C-ECHO SCU for the verification service class.

use super::association::{networking_error, Association, EchoResponse, NetworkingError};
use super::command::{EXPLICIT_VR_LE_TS, VERIFICATION_SOP_CLASS};
use super::context::AssociationConfig;

/// Send a C-ECHO to the configured PACS endpoint and return the response.
pub fn echo(config: &AssociationConfig) -> Result<EchoResponse, NetworkingError> {
    let mut association =
        Association::for_service(config, VERIFICATION_SOP_CLASS, EXPLICIT_VR_LE_TS)?;
    let status = association.c_echo().map_err(networking_error)?;
    association.release().map_err(networking_error)?;
    Ok(EchoResponse { status })
}
