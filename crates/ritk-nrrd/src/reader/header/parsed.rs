use std::collections::HashMap;

/// Parsed NRRD fields, comments, and custom key/value metadata.
#[derive(Debug)]
pub struct NrrdHeader {
    pub(in crate::reader) fields: HashMap<String, String>,
    pub(in crate::reader) key_values: HashMap<String, String>,
    pub(in crate::reader) key_value_records: Vec<NrrdKeyValueRecord>,
    pub(in crate::reader) comments: Vec<String>,
    pub(in crate::reader) format_version: u8,
}

/// One decoded NRRD custom key/value record from the header.
#[derive(Debug, Eq, PartialEq)]
pub struct NrrdKeyValueRecord {
    key: String,
    value: String,
}

impl NrrdKeyValueRecord {
    pub(in crate::reader) fn new(key: String, value: String) -> Self {
        Self { key, value }
    }

    /// Returns this record's decoded key.
    #[must_use]
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Returns this record's decoded value.
    #[must_use]
    pub fn value(&self) -> &str {
        &self.value
    }
}

impl NrrdHeader {
    /// Returns standard fields by their lowercase canonical names.
    #[must_use]
    pub fn fields(&self) -> &HashMap<String, String> {
        &self.fields
    }

    /// Returns each custom key's effective value; a repeated key uses its
    /// last value, as specified by NRRD.
    #[must_use]
    pub fn key_values(&self) -> &HashMap<String, String> {
        &self.key_values
    }

    /// Returns decoded custom key/value records in source order, including
    /// repeated keys.
    #[must_use]
    pub fn key_value_records(&self) -> &[NrrdKeyValueRecord] {
        &self.key_value_records
    }

    /// Returns comment lines in source order, including each leading `#`.
    #[must_use]
    pub fn comments(&self) -> &[String] {
        &self.comments
    }

    /// Returns the version declared by the NRRD magic line.
    #[must_use]
    pub const fn format_version(&self) -> u8 {
        self.format_version
    }
}
