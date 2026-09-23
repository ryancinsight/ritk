//! Attribute and text-field parsing for the GIFTI elements.

use quick_xml::XmlVersion;
use quick_xml::events::BytesStart;

use super::Element;
use crate::GiftiError;

/// The attributes of one start tag.
pub(super) struct Attributes(Vec<(String, String)>);

impl Attributes {
    pub(super) fn of(start: &BytesStart<'_>, element: Element) -> Result<Self, GiftiError> {
        let mut pairs = Vec::new();
        for attribute in start.attributes() {
            let attribute = attribute
                .map_err(|error| GiftiError::structure(element.tag(), error.to_string()))?;
            let value = attribute
                .normalized_value(XmlVersion::Implicit1_0)
                .map_err(|error| GiftiError::structure(element.tag(), error.to_string()))?;
            let key = String::from_utf8_lossy(attribute.key.local_name().as_ref()).into_owned();
            pairs.push((key, value.into_owned()));
        }
        Ok(Self(pairs))
    }

    pub(super) fn optional(&self, name: &str) -> Option<&str> {
        self.0
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }

    pub(super) fn required(&self, element: &'static str, name: &str) -> Result<&str, GiftiError> {
        self.optional(name)
            .ok_or_else(|| GiftiError::structure(element, format!("missing {name}")))
    }
}

/// `Key` (or the pre-1.0 `Index`, section 2.6.3.1) and the colour.
pub(super) fn parse_label(attributes: &Attributes) -> Result<(u32, Option<[f32; 4]>), GiftiError> {
    let key = attributes
        .optional("Key")
        .or_else(|| attributes.optional("Index"))
        .ok_or_else(|| GiftiError::structure("Label", "missing Key"))?;
    let key = key
        .parse::<u32>()
        .map_err(|_| GiftiError::structure("Label", format!("Key {key:?}")))?;
    let components = ["Red", "Green", "Blue", "Alpha"].map(|name| attributes.optional(name));
    if components.iter().all(Option::is_none) {
        return Ok((key, None));
    }
    let mut rgba = [0.0_f32; 4];
    for (slot, (name, text)) in rgba.iter_mut().zip(
        ["Red", "Green", "Blue", "Alpha"]
            .into_iter()
            .zip(components),
    ) {
        let text = text.ok_or_else(|| {
            GiftiError::structure("Label", format!("key {key}: colour given without {name}"))
        })?;
        *slot = text
            .parse()
            .map_err(|_| GiftiError::structure("Label", format!("key {key}: {name} {text:?}")))?;
    }
    Ok((key, Some(rgba)))
}

/// Sixteen numbers, rows first (section 2.8).
pub(super) fn parse_matrix(text: &str) -> Result<[[f64; 4]; 4], GiftiError> {
    let values = text
        .split_ascii_whitespace()
        .map(str::parse::<f64>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| GiftiError::structure("MatrixData", error.to_string()))?;
    let [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p] = values.as_slice() else {
        return Err(GiftiError::structure(
            "MatrixData",
            format!("{} values, expected 16", values.len()),
        ));
    };
    Ok([
        [*a, *b, *c, *d],
        [*e, *f, *g, *h],
        [*i, *j, *k, *l],
        [*m, *n, *o, *p],
    ])
}
