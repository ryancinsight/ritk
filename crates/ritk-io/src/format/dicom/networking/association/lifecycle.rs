//! Association lifecycle operations for the `dicom-ul` SCU boundary.

use super::NetworkingError;
use dicom_ul::association::client::ClientAssociation;
use dicom_ul::pdu::Pdu;
use std::net::TcpStream;

/// Complete the DICOM A-RELEASE handshake without a second transport close.
///
/// `dicom-ul` 0.10 treats a macOS `NotConnected` result from the post-release
/// TCP shutdown as a protocol failure even after receiving `A-RELEASE-RP`.
/// The DICOM protocol completes at that reply; consuming the association then
/// drops the socket. The upstream defect is tracked in Enet4/dicom-rs#811.
pub(crate) fn release_client_association(
    mut association: ClientAssociation<TcpStream>,
) -> Result<(), NetworkingError> {
    association
        .send(&Pdu::ReleaseRQ)
        .map_err(|error| NetworkingError::Protocol(error.to_string()))?;

    match association
        .receive()
        .map_err(|error| NetworkingError::Protocol(error.to_string()))?
    {
        Pdu::ReleaseRP => Ok(()),
        pdu => Err(NetworkingError::Protocol(format!(
            "expected A-RELEASE-RP, got {pdu:?}"
        ))),
    }
}
