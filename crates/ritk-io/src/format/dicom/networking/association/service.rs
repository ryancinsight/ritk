//! Shared association setup for the one-shot DIMSE service APIs.

use super::{Association, NetworkingError};
use crate::format::dicom::networking::context::{AssociationConfig, RequestedPresentationContext};

pub(in crate::format::dicom::networking) fn networking_error(
    error: anyhow::Error,
) -> NetworkingError {
    match error.downcast::<NetworkingError>() {
        Ok(error) => error,
        Err(error) => NetworkingError::Protocol(format!("{error:#}")),
    }
}

impl Association {
    pub(in crate::format::dicom::networking) fn for_service(
        config: &AssociationConfig,
        sop_class: &str,
        syntax: &str,
    ) -> Result<Self, NetworkingError> {
        let mut config = config.clone();
        let abstract_syntax_uid = sop_class
            .try_into()
            .map_err(|_| NetworkingError::Protocol("SOP class UID exceeds 64 bytes".to_owned()))?;
        let transfer_syntax = syntax.try_into().map_err(|_| {
            NetworkingError::Protocol("transfer syntax UID exceeds 64 bytes".to_owned())
        })?;
        config.presentation_contexts = vec![RequestedPresentationContext {
            abstract_syntax_uid,
            transfer_syntax_uids: vec![transfer_syntax],
        }];
        Self::connect(config).map_err(networking_error)
    }
}
