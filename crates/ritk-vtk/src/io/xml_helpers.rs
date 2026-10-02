//! Shared XML parsing helpers for VTK XML formats (VTI, VTP, VTU).

use crate::domain::vtk_data_object::AttributeArray;
use crate::io::xml_array::{attribute_from_values, component_count, decode_ascii_attribute};
use anyhow::{Context, Result};
use std::collections::HashMap;

/// Default VTK origin when the attribute is absent.
pub(crate) const DEFAULT_ORIGIN_STR: &str = "0 0 0";
/// Default VTK spacing when the attribute is absent.
pub(crate) const DEFAULT_SPACING_STR: &str = "1 1 1";

/// Return the opening tag string for the first occurrence of `<tag ...>` or
/// `<tag>` in `s`, including the closing `>`.
pub(crate) fn find_tag(s: &str, tag: &str) -> Option<String> {
    let open = format!("<{}", tag);
    let start = s.find(&open)?;
    let end = s[start..].find('>')? + 1;
    Some(s[start..start + end].to_string())
}

/// Return the substring from the first `<tag` to the matching `</tag>` (inclusive).
pub(crate) fn find_section(s: &str, tag: &str) -> Option<String> {
    let open = format!("<{}", tag);
    let close = format!("</{}>", tag);
    let start = s.find(&open)?;
    let end_offset = s[start..].find(&close)? + close.len();
    Some(s[start..start + end_offset].to_string())
}

/// Parse the `name="value"` attribute from an XML tag string.
pub(crate) fn attr_val(tag: &str, name: &str) -> Option<String> {
    let mut pat = name.to_string();
    pat.push('=');
    pat.push('"');
    let start = tag.find(&pat)? + pat.len();
    let rest = &tag[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Parse a `usize` attribute from an XML tag string.
pub(crate) fn attr_usize(tag: &str, name: &str) -> Result<usize> {
    let v = attr_val(tag, name)
        .ok_or_else(|| anyhow::anyhow!("attribute '{}' not found in tag: {}", name, tag))?;
    v.parse()
        .with_context(|| format!("cannot parse attribute '{}' as usize: {}", name, v))
}

/// Parse every whitespace-separated token of `text` as a `T`.
///
/// `what` names the field in the error, which also names the offending token:
/// a token that is not a `T` is never dropped.
///
/// # Errors
///
/// Returns an error for the first token that does not parse.
pub(crate) fn parse_values<T>(text: &str, what: &str) -> Result<Vec<T>>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    text.split_whitespace()
        .map(|token| {
            token.parse().with_context(|| {
                format!("{what}: bad {} token '{token}'", std::any::type_name::<T>())
            })
        })
        .collect()
}

/// Parse `text` as exactly `N` whitespace-separated values of `T`.
///
/// # Errors
///
/// Returns an error for a token that does not parse or a count other than `N`.
pub(crate) fn parse_array<T, const N: usize>(text: &str, what: &str) -> Result<[T; N]>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    let values = parse_values::<T>(text, what)?;
    let count = values.len();
    values
        .try_into()
        .map_err(|_| anyhow::anyhow!("{what} must hold {N} values, got {count}"))
}

/// Extract the text content of the first `<DataArray ...>...</DataArray>` in `section`.
pub(crate) fn extract_da_content(section: &str) -> String {
    let da_start = match section.find("<DataArray") {
        Some(p) => p,
        None => return String::new(),
    };
    let rest = &section[da_start..];
    let gt = match rest.find('>') {
        Some(p) => p + 1,
        None => return String::new(),
    };
    let lt = rest[gt..].find("</").map(|p| gt + p).unwrap_or(rest.len());
    rest[gt..lt].trim().to_string()
}

/// Decode the first `<DataArray>` of `section` as `f32` values, in the type its
/// tag declares ([`decode_ascii_attribute`]).
///
/// # Errors
///
/// Returns an error when `section` holds no `<DataArray>` or it fails to decode.
pub(crate) fn first_array_values(section: &str) -> Result<Vec<f32>> {
    let start = section
        .find("<DataArray")
        .ok_or_else(|| anyhow::anyhow!("section holds no <DataArray>"))?;
    let rest = &section[start..];
    let tag_end = rest
        .find('>')
        .ok_or_else(|| anyhow::anyhow!("unterminated <DataArray> tag"))?
        + 1;
    decode_ascii_attribute(&rest[..tag_end], &extract_da_content(section))
}

/// Parse the content of a `<DataArray>` element (as returned by
/// [`named_da`]) as integers.
///
/// # Errors
///
/// Returns an error for a token that is not an `i64`.
pub(crate) fn da_integers(da: &str, what: &str) -> Result<Vec<i64>> {
    parse_values(&extract_da_content(da), what)
}

/// The `UInt32` indices held by the `<DataArray>` element `da`, as
/// [`da_integers`] parses them.
///
/// # Errors
///
/// Returns an error for a token that is not an `i64` or whose value is negative
/// or exceeds `u32::MAX`.
pub(crate) fn index_values(da: &str, field: &str) -> Result<Vec<u32>> {
    da_integers(da, field)?
        .into_iter()
        .map(|value| {
            u32::try_from(value).with_context(|| {
                format!("{field} value {value} is not a non-negative UInt32 index")
            })
        })
        .collect()
}

/// Find a named `<DataArray Name="name" ...>...</DataArray>` within `section`.
pub(crate) fn named_da(section: &str, name: &str) -> Option<String> {
    let mut np = String::from("Name=\"");
    np.push_str(name);
    np.push('"');
    let attr_pos = section.find(&np)?;
    let da_start = section[..attr_pos].rfind("<DataArray")?;
    let rest = &section[da_start..];
    let close = "</DataArray>";
    let end = rest.find(close)? + close.len();
    Some(rest[..end].to_string())
}

/// Parse all ASCII `<DataArray>` elements in a PointData/CellData section into
/// an attribute map.
///
/// Each array decodes in its declared `type` and converts to `f32`
/// ([`crate::io::xml_array`]); the component count selects the attribute kind
/// as [`attribute_from_values`] describes.
///
/// # Errors
///
/// Returns an error for an array that is not ASCII, has an unsupported `type`,
/// a malformed token, or a value count that contradicts its tag.
pub(crate) fn parse_attrs(section: &str) -> Result<HashMap<String, AttributeArray>> {
    let mut map = HashMap::new();
    let mut rest = section;
    let close = "</DataArray>";
    while let Some(start) = rest.find("<DataArray") {
        rest = &rest[start..];
        let Some(te) = rest.find('>').map(|e| e + 1) else {
            break;
        };
        let tag = &rest[..te];
        let Some(de) = rest.find(close) else {
            break;
        };
        let name = attr_val(tag, "Name").unwrap_or_default();
        if !name.is_empty() {
            let values = decode_ascii_attribute(tag, rest[te..de].trim())
                .with_context(|| format!("DataArray '{name}'"))?;
            map.insert(
                name.clone(),
                attribute_from_values(&name, component_count(tag)?, values)?,
            );
        }
        rest = &rest[de + close.len()..];
    }
    Ok(map)
}
