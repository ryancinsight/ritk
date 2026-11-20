use std::collections::VecDeque;
use std::io::Write;

use anyhow::Result;

use super::encoding::{defined_length, emit, encoded_dataset, plan_dataset};
use super::fields::attribute_header;
use crate::dataset::wire::{charset::CharacterSet, syntax::Encoding};
use crate::dataset::{DicomFile, FileMeta, VR};
use crate::{DicomTag, TransferSyntaxKind};

pub(crate) struct Part10<'a> {
    file: &'a DicomFile,
    syntax: TransferSyntaxKind,
    encoding: Encoding,
    lengths: VecDeque<u32>,
}

impl<'a> Part10<'a> {
    pub(crate) fn plan(file: &'a DicomFile) -> Result<Self> {
        let syntax = TransferSyntaxKind::from_uid(file.meta().transfer_syntax());
        let (encoding, lengths) = plan_dataset(file, &syntax)?;
        file.meta().validate()?;
        Ok(Self {
            file,
            syntax,
            encoding,
            lengths,
        })
    }

    pub(crate) fn write(mut self, mut out: impl Write) -> Result<()> {
        file_header(self.file.meta(), self.file.preamble.as_deref(), &mut out)?;
        encoded_dataset(
            self.file,
            out,
            &self.syntax,
            self.encoding,
            &mut self.lengths,
        )
    }
}

pub(crate) fn file_header(
    meta: &FileMeta,
    preamble: Option<&[u8; 128]>,
    mut out: impl Write,
) -> Result<()> {
    let meta = meta.to_dataset()?;
    let mut metadata = Vec::new();
    emit(
        &meta,
        Encoding::EXPLICIT_LE,
        &CharacterSet::default(),
        &mut VecDeque::new(),
        &mut metadata,
    )?;
    let size = defined_length(metadata.len())?;
    if let Some(preamble) = preamble {
        out.write_all(preamble.as_slice())?;
    }
    out.write_all(b"DICM")?;
    attribute_header(
        &mut out,
        Encoding::EXPLICIT_LE,
        DicomTag::new(2, 0),
        VR::UL,
        4,
    )?;
    out.write_all(&size.to_le_bytes())?;
    out.write_all(&metadata)?;
    Ok(())
}
