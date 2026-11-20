//! C-FIND SCU — Study Root Query/Retrieve: FIND (PS3.4 §C.4.1).

use super::association::{networking_error, Association, FindResult, NetworkingError};
use super::command::{
    encode_query_identifier, IMPLICIT_VR_LE_TS, STATUS_SUCCESS, STUDY_ROOT_FIND_SOP_CLASS,
};
use super::context::AssociationConfig;

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
        Self {
            level,
            keys: Vec::with_capacity(8),
        }
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
    let dataset = encode_query_identifier(
        query.level.as_cs_str(),
        query
            .keys
            .iter()
            .map(|(group, element, value)| (*group, *element, value.as_str())),
    );
    let mut association =
        Association::for_service(config, STUDY_ROOT_FIND_SOP_CLASS, IMPLICIT_VR_LE_TS)?;
    let mut results = Vec::new();
    let status = association
        .find_responses(STUDY_ROOT_FIND_SOP_CLASS, dataset, |status, data| {
            results.push(FindResult {
                matches: vec![data],
                status,
            });
        })
        .map_err(networking_error)?;
    if !matches!(status, STATUS_SUCCESS | 0xA700 | 0xA900 | 0xC000..=0xCFFF) {
        return Err(NetworkingError::UnexpectedStatus(status));
    }
    association.release().map_err(networking_error)?;
    Ok(results)
}
