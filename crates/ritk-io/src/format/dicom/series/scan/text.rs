//! DICOM text decoding for the selected catalog attributes.

use std::borrow::Cow;

use anyhow::{Context, Result};
use dicom::encoding::text::{SpecificCharacterSet, TextCodec};
use ritk_dicom::{tags, DicomTag, Part10Attributes};

pub(super) struct AttributeText<'a, 'data, const N: usize> {
    attributes: &'a Part10Attributes<'data, N>,
    charset: SpecificCharacterSet,
}

impl<'a, 'data, const N: usize> AttributeText<'a, 'data, N> {
    pub(super) fn new(attributes: &'a Part10Attributes<'data, N>) -> Result<Self> {
        let charset = match attributes.primitive(tags::SPECIFIC_CHARACTER_SET)? {
            Some(bytes) => {
                let designation = std::str::from_utf8(bytes)
                    .context("DICOM character-set designation is invalid")?;
                // The existing decoder selects the first declared repertoire
                // and retains the default when it cannot identify that value.
                let first = designation
                    .split('\\')
                    .next()
                    .unwrap_or_default()
                    .trim_end_matches([' ', '\0']);
                SpecificCharacterSet::from_code(first).unwrap_or_default()
            }
            None => SpecificCharacterSet::default(),
        };
        Ok(Self {
            attributes,
            charset,
        })
    }

    pub(super) fn default_text(&self, tag: DicomTag) -> Result<Option<Cow<'data, str>>> {
        self.read_text(tag, &SpecificCharacterSet::ISO_IR_6)
    }

    pub(super) fn local_text(&self, tag: DicomTag) -> Result<Option<Cow<'data, str>>> {
        self.read_text(tag, &self.charset)
    }

    fn read_text(
        &self,
        tag: DicomTag,
        charset: &SpecificCharacterSet,
    ) -> Result<Option<Cow<'data, str>>> {
        self.attributes
            .primitive(tag)?
            .map(|bytes| {
                let remove_padding = |c: char| c == ' ' || c == '\0';
                if *charset == SpecificCharacterSet::ISO_IR_6 && bytes.is_ascii() {
                    return Ok(Cow::Borrowed(
                        std::str::from_utf8(bytes)?.trim_end_matches(remove_padding),
                    ));
                }
                let mut decoded = charset.decode(bytes).with_context(|| {
                    format!("decode DICOM text ({:04X},{:04X})", tag.group, tag.element)
                })?;
                decoded.truncate(decoded.trim_end_matches(remove_padding).len());
                Ok(Cow::Owned(decoded))
            })
            .transpose()
    }
}
