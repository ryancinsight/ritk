use std::io::Write;

use anyhow::{bail, Result};

use crate::DicomTag;

use super::super::{Dataset, PrimitiveValue, VR};

fn text(dataset: &Dataset, tag: DicomTag) -> Option<String> {
    dataset
        .get(tag)?
        .to_str()
        .ok()
        .map(|value| value.trim_matches([' ', '\0']).to_owned())
}

const META: u16 = 2;
const SOP_CLASS: DicomTag = DicomTag::new(8, 0x16);
const SOP_INSTANCE: DicomTag = DicomTag::new(8, 0x18);
pub const IMPLEMENTATION_CLASS_UID: &str = "1.2.826.0.1.3690043.9.7433.1.1";

#[derive(Debug, Clone, PartialEq)]
pub struct FileMeta {
    pub media_storage_sop_class_uid: String,
    pub media_storage_sop_instance_uid: String,
    pub transfer_syntax: String,
    pub implementation_class_uid: String,
    pub implementation_version_name: Option<String>,
    pub source_application_entity_title: Option<String>,
    pub sending_application_entity_title: Option<String>,
    pub receiving_application_entity_title: Option<String>,
    pub private_information_creator_uid: Option<String>,
    pub private_information: Option<Vec<u8>>,
    version: [u8; 2],
    extra: Dataset,
}

impl Default for FileMeta {
    fn default() -> Self {
        Self {
            media_storage_sop_class_uid: String::new(),
            media_storage_sop_instance_uid: String::new(),
            transfer_syntax: String::new(),
            implementation_class_uid: IMPLEMENTATION_CLASS_UID.to_owned(),
            implementation_version_name: None,
            source_application_entity_title: None,
            sending_application_entity_title: None,
            receiving_application_entity_title: None,
            private_information_creator_uid: None,
            private_information: None,
            version: [0, 1],
            extra: Dataset::new(),
        }
    }
}

impl FileMeta {
    pub fn media_storage_sop_class_uid(&self) -> &str {
        self.media_storage_sop_class_uid
            .trim_end_matches([' ', '\0'])
    }

    pub fn media_storage_sop_instance_uid(&self) -> &str {
        self.media_storage_sop_instance_uid
            .trim_end_matches([' ', '\0'])
    }

    pub fn transfer_syntax(&self) -> &str {
        self.transfer_syntax.trim_end_matches([' ', '\0'])
    }

    pub(crate) fn from_builder(builder: FileMetaBuilder, dataset: &Dataset) -> Result<Self> {
        let FileMetaBuilder(mut meta) = builder;
        if meta.media_storage_sop_class_uid.is_empty() {
            meta.media_storage_sop_class_uid = text(dataset, SOP_CLASS).unwrap_or_default();
        }
        if meta.media_storage_sop_instance_uid.is_empty() {
            meta.media_storage_sop_instance_uid = text(dataset, SOP_INSTANCE).unwrap_or_default();
        }
        FileMetaBuilder(meta).build()
    }

    /// Write the Part 10 preamble and File Meta Information of a DIMSE dataset.
    ///
    /// # Errors
    /// Returns an error for invalid metadata or an output failure.
    pub fn write_header(&self, writer: impl Write) -> Result<()> {
        super::wire::write::file_header(self, Some(&[0; 128]), writer)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        for (name, uid) in [
            (
                "Media Storage SOP Class UID",
                self.media_storage_sop_class_uid(),
            ),
            (
                "Media Storage SOP Instance UID",
                self.media_storage_sop_instance_uid(),
            ),
            ("Transfer Syntax UID", self.transfer_syntax()),
            ("Implementation Class UID", &self.implementation_class_uid),
        ] {
            if uid.is_empty() || uid.len() > 64 || !uid.is_ascii() {
                bail!("invalid or missing DICOM file metadata {name}");
            }
        }
        Ok(())
    }

    pub(crate) fn from_dataset(mut dataset: Dataset) -> Result<Self> {
        let mut take_text = |element| -> Result<Option<String>> {
            dataset
                .remove_element(DicomTag::new(META, element))
                .map(|field| field.to_str().map(|value| value.into_owned()))
                .transpose()
        };
        let mut meta = Self {
            media_storage_sop_class_uid: take_text(0x0002)?.unwrap_or_default(),
            media_storage_sop_instance_uid: take_text(0x0003)?.unwrap_or_default(),
            transfer_syntax: take_text(0x0010)?.unwrap_or_default(),
            implementation_class_uid: take_text(0x0012)?
                .unwrap_or_else(|| IMPLEMENTATION_CLASS_UID.to_owned()),
            implementation_version_name: take_text(0x0013)?,
            source_application_entity_title: take_text(0x0016)?,
            sending_application_entity_title: take_text(0x0017)?,
            receiving_application_entity_title: take_text(0x0018)?,
            private_information_creator_uid: take_text(0x0100)?,
            ..Self::default()
        };
        if let Some(entry) = dataset.remove_element(DicomTag::new(META, 1)) {
            meta.version = entry.value().to_bytes()?.as_ref().try_into()?;
        }
        meta.private_information = dataset
            .remove_element(DicomTag::new(META, 0x0102))
            .map(|entry| entry.value().to_bytes().map(|value| value.into_owned()))
            .transpose()?;
        meta.extra = dataset;
        Ok(meta)
    }

    pub(crate) fn to_dataset(&self) -> Result<Dataset> {
        self.validate()?;
        let mut entries = self.extra.clone();
        entries.put_value(DicomTag::new(META, 1), VR::OB, self.version);
        for (element, vr, value) in [
            (0x0002, VR::UI, Some(self.media_storage_sop_class_uid())),
            (0x0003, VR::UI, Some(self.media_storage_sop_instance_uid())),
            (0x0010, VR::UI, Some(self.transfer_syntax())),
            (0x0012, VR::UI, Some(self.implementation_class_uid.as_str())),
            (0x0013, VR::SH, self.implementation_version_name.as_deref()),
            (
                0x0016,
                VR::AE,
                self.source_application_entity_title.as_deref(),
            ),
            (
                0x0017,
                VR::AE,
                self.sending_application_entity_title.as_deref(),
            ),
            (
                0x0018,
                VR::AE,
                self.receiving_application_entity_title.as_deref(),
            ),
            (
                0x0100,
                VR::UI,
                self.private_information_creator_uid.as_deref(),
            ),
        ] {
            if let Some(value) = value {
                entries.put_value(DicomTag::new(META, element), vr, value);
            }
        }
        if let Some(value) = &self.private_information {
            entries.put_value(
                DicomTag::new(META, 0x0102),
                VR::OB,
                PrimitiveValue::from(value.clone()),
            );
        }
        Ok(entries)
    }
}

#[derive(Debug, Default)]
pub struct FileMetaBuilder(FileMeta);

impl FileMetaBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// # Errors
    /// Returns an error for missing or invalid required file metadata.
    pub fn build(self) -> Result<FileMeta> {
        self.0.validate()?;
        Ok(self.0)
    }

    #[must_use]
    pub fn transfer_syntax(mut self, uid: impl Into<String>) -> Self {
        self.0.transfer_syntax = uid.into();
        self
    }

    #[must_use]
    pub fn media_storage_sop_class_uid(mut self, uid: impl Into<String>) -> Self {
        self.0.media_storage_sop_class_uid = uid.into();
        self
    }

    #[must_use]
    pub fn media_storage_sop_instance_uid(mut self, uid: impl Into<String>) -> Self {
        self.0.media_storage_sop_instance_uid = uid.into();
        self
    }

    #[must_use]
    pub fn implementation_class_uid(mut self, uid: impl Into<String>) -> Self {
        self.0.implementation_class_uid = uid.into();
        self
    }
}
