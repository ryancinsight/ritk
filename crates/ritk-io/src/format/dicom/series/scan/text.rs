//! DICOM text decoding for the selected catalog attributes.

use std::borrow::Cow;

use anyhow::{Context, Result};
use ritk_dicom::{tags, CharacterSet, DicomTag, Part10Attributes};

pub(super) struct AttributeText<'a, 'data, const N: usize> {
    attributes: &'a Part10Attributes<'data, N>,
    charset: CharacterSet,
}

impl<'a, 'data, const N: usize> AttributeText<'a, 'data, N> {
    pub(super) fn new(attributes: &'a Part10Attributes<'data, N>) -> Result<Self> {
        let charset = match attributes.primitive(tags::SPECIFIC_CHARACTER_SET)? {
            Some(bytes) => {
                let designation = std::str::from_utf8(bytes)
                    .context("DICOM character-set designation is invalid")?;
                CharacterSet::from_declaration(designation)?
            }
            None => CharacterSet::default(),
        };
        Ok(Self {
            attributes,
            charset,
        })
    }

    pub(super) fn default_text(&self, tag: DicomTag) -> Result<Option<Cow<'_, str>>> {
        self.read_text(tag, &CharacterSet::default())
    }

    pub(super) fn local_text(&self, tag: DicomTag) -> Result<Option<Cow<'_, str>>> {
        self.read_text(tag, &self.charset)
    }

    fn read_text(&self, tag: DicomTag, charset: &CharacterSet) -> Result<Option<Cow<'_, str>>> {
        self.attributes
            .primitive(tag)?
            .map(|bytes| {
                let decoded = charset
                    .decode(bytes)
                    .with_context(|| format!("decode DICOM text {tag}"))?;
                let trim = |value: &str| value.trim_end_matches([' ', '\0']).len();
                Ok(match decoded {
                    Cow::Borrowed(text) => Cow::Borrowed(&text[..trim(text)]),
                    Cow::Owned(mut text) => {
                        text.truncate(trim(&text));
                        Cow::Owned(text)
                    }
                })
            })
            .transpose()
    }
}
