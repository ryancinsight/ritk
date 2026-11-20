//! DICOM Specific Character Set and inherited dataset text representation.

use std::borrow::Cow;

use anyhow::{anyhow, bail, Result};
use encoding_rs::Encoding;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Repertoire {
    Ascii,
    Latin,
    Encoded(&'static Encoding),
}

#[derive(Debug, Clone)]
pub struct CharacterSet {
    first: Repertoire,
    extensions: super::charset::extension::Designations,
}

mod extension;

impl Default for CharacterSet {
    fn default() -> Self {
        Self {
            first: Repertoire::Ascii,
            extensions: extension::Designations::default(),
        }
    }
}

impl CharacterSet {
    /// Interpret (0008,0005). An empty first value designates ISO IR 6.
    ///
    /// # Errors
    /// A repertoire name must designate a DICOM single repertoire or a valid
    /// set of code-extension repertoires.
    pub fn from_declaration(declaration: &str) -> Result<Self> {
        let mut codes = declaration
            .split('\\')
            .map(|value| value.trim_matches([' ', '\0']));
        let first_code = codes.next().unwrap_or("");
        let first = repertoire(first_code)?;
        let mut extensions = extension::Designations::default();
        if first_code.starts_with("ISO 2022") {
            extensions.insert(first_code)?;
        }
        for code in codes {
            extensions.insert(code)?;
        }
        Ok(Self { first, extensions })
    }

    /// # Errors
    /// Returns an error for invalid bytes in the declared character set.
    pub fn decode<'a>(&self, bytes: &'a [u8]) -> Result<Cow<'a, str>> {
        if !self.extensions.is_empty() && bytes.contains(&0x1b) {
            return self.extensions.decode(self.first, bytes).map(Cow::Owned);
        }
        decode(self.first, bytes)
    }

    pub(crate) fn encode<'a>(&self, text: &'a str) -> Result<Cow<'a, [u8]>> {
        if !self.extensions.is_empty() && !text.is_ascii() {
            return self.extensions.encode(self.first, text).map(Cow::Owned);
        }
        encode(self.first, text)
    }
}

// ISO_IR 100 is ISO-8859-1. Its C1 positions are not Windows-1252
// characters. Handle this repertoire directly.
fn repertoire(code: &str) -> Result<Repertoire> {
    let number = code
        .strip_prefix("ISO 2022 IR ")
        .or_else(|| code.strip_prefix("ISO_IR "))
        .or_else(|| code.strip_prefix("ISO_IR_"));
    if matches!(code, "" | "Default") || number == Some("6") {
        return Ok(Repertoire::Ascii);
    }
    if number == Some("100") {
        return Ok(Repertoire::Latin);
    }
    let label: &[u8] = match (number, code) {
        (Some("13"), _) => b"shift_jis",
        (Some("87" | "159"), _) => b"euc-jp",
        (Some("101"), _) => b"iso-8859-2",
        (Some("109"), _) => b"iso-8859-3",
        (Some("110"), _) => b"iso-8859-4",
        (Some("144"), _) => b"iso-8859-5",
        (Some("127"), _) => b"iso-8859-6",
        (Some("126"), _) => b"iso-8859-7",
        (Some("138"), _) => b"iso-8859-8",
        (Some("148"), _) => b"iso-8859-9",
        (Some("149"), _) => b"euc-kr",
        (Some("166"), _) => b"tis-620",
        (Some("192"), _) => b"utf-8",
        (Some("58"), _) | (_, "GBK" | "GB2312") => b"gbk",
        (_, "GB18030") => b"gb18030",
        _ => bail!("unsupported DICOM Specific Character Set {code:?}"),
    };
    Encoding::for_label(label)
        .map(Repertoire::Encoded)
        .ok_or_else(|| anyhow!("missing codec for DICOM Specific Character Set {code:?}"))
}

fn decode(repertoire: Repertoire, bytes: &[u8]) -> Result<Cow<'_, str>> {
    match repertoire {
        Repertoire::Ascii if bytes.is_ascii() => Ok(Cow::Borrowed(std::str::from_utf8(bytes)?)),
        Repertoire::Ascii => bail!("non-ASCII byte in a DICOM default-repertoire value"),
        Repertoire::Latin if bytes.is_ascii() => Ok(Cow::Borrowed(std::str::from_utf8(bytes)?)),
        Repertoire::Latin => Ok(Cow::Owned(bytes.iter().copied().map(char::from).collect())),
        Repertoire::Encoded(codec) => codec
            .decode_without_bom_handling_and_without_replacement(bytes)
            .ok_or_else(|| anyhow!("invalid {} DICOM text", codec.name())),
    }
}

fn encode(repertoire: Repertoire, text: &str) -> Result<Cow<'_, [u8]>> {
    match repertoire {
        Repertoire::Ascii if text.is_ascii() => Ok(Cow::Borrowed(text.as_bytes())),
        Repertoire::Ascii => bail!("DICOM default repertoire cannot encode non-ASCII text"),
        Repertoire::Latin => Ok(Cow::Owned(
            text.chars()
                .map(|ch| u8::try_from(u32::from(ch)))
                .collect::<Result<Vec<_>, _>>()?,
        )),
        Repertoire::Encoded(codec) => {
            let (bytes, _, failed) = codec.encode(text);
            if failed {
                bail!("DICOM text cannot be encoded in {}", codec.name());
            }
            Ok(bytes)
        }
    }
}
