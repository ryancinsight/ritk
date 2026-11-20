use std::io::{BufWriter, Write};
use std::ops::{Deref, DerefMut};
use std::path::Path;

use anyhow::{Context, Result};
use bytes::Bytes;
use consus_core::ParseBudget;

use super::FileMeta;
use crate::dataset::{wire, Dataset};
use crate::TransferSyntaxKind;

#[derive(Debug, Clone, PartialEq)]
pub struct DicomFile {
    pub(crate) preamble: Option<Box<[u8; 128]>>,
    meta: FileMeta,
    dataset: Dataset,
}

impl DicomFile {
    pub fn new(meta: FileMeta, dataset: Dataset) -> Self {
        Self {
            preamble: Some(Box::new([0; 128])),
            meta,
            dataset,
        }
    }

    /// # Errors
    /// Returns file, framing, and resource-budget failures.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_budget(path, &ParseBudget::DEFAULT)
    }

    /// # Errors
    /// Returns file, framing, and resource-budget failures.
    pub fn open_with_budget(path: impl AsRef<Path>, budget: &ParseBudget) -> Result<Self> {
        let path = path.as_ref();
        let input = crate::read_file_with_budget(path, budget)?;
        Self::parse(input.into(), budget).with_context(|| format!("parse DICOM {}", path.display()))
    }

    /// Parse Part 10. Primitive and fragment bytes may retain the input.
    ///
    /// # Errors
    /// Returns framing, text codec, and resource-budget failures.
    pub fn parse(input: Bytes, budget: &ParseBudget) -> Result<Self> {
        crate::backend::parse_budget::materialize(input, budget)
    }

    pub const fn meta(&self) -> &FileMeta {
        &self.meta
    }

    pub fn update_meta(&mut self, update: impl FnOnce(&mut FileMeta)) {
        update(&mut self.meta);
    }

    /// # Errors
    /// Returns metadata, encoding, and writer failures.
    pub fn write_all(&self, writer: impl Write) -> Result<()> {
        wire::write::Part10::plan(self)?.write(writer)
    }

    /// # Errors
    /// Returns metadata, encoding, and filesystem failures.
    pub fn write_to_file(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let context = || format!("write DICOM {}", path.display());
        let plan = wire::write::Part10::plan(self).with_context(context)?;
        let mut output = BufWriter::new(std::fs::File::create(path).with_context(context)?);
        plan.write(&mut output).with_context(context)?;
        output.flush()?;
        Ok(())
    }

    /// Write the DIMSE dataset payload without file metadata.
    ///
    /// # Errors
    /// Returns an error when the data does not fit the requested transfer
    /// syntax, an element length is not representable, or the writer fails.
    pub fn write_dataset(&self, writer: impl Write, syntax: &TransferSyntaxKind) -> Result<()> {
        wire::write::dataset(self, writer, syntax)
    }
}

impl Deref for DicomFile {
    type Target = Dataset;
    fn deref(&self) -> &Self::Target {
        &self.dataset
    }
}

impl DerefMut for DicomFile {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.dataset
    }
}
