//! Shared XML parsing helpers for VTK XML formats (VTI, VTP, VTU).

use crate::domain::vtk_data_object::AttributeArray;
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

/// Parse space/newline-delimited values from a string slice.
///
/// Generic over any type parseable via [`std::str::FromStr`].
/// Tokens that fail to parse are silently dropped.
pub(crate) fn parse_floats<T>(s: &str) -> Vec<T>
where
    T: std::str::FromStr,
{
    s.split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect()
}

/// Parse space/newline-delimited `i64` values from a string slice.
pub(crate) fn parse_i64s(s: &str) -> Vec<i64> {
    parse_floats(s)
}

/// Parse space/newline-delimited `i32` values from a string slice.
pub(crate) fn parse_ints(s: &str) -> Vec<i32> {
    parse_floats(s)
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

/// Parse all `<DataArray>` elements in a PointData/CellData section into an
/// attribute map.
///
/// - `NumberOfComponents="3"` â†’ `Vectors` (or `Normals` when name contains "normal").
/// - `NumberOfComponents="2"` â†’ `TextureCoords` with `dim=2`.
/// - All other component counts â†’ `Scalars` with that `num_components`.
pub(crate) fn parse_attrs(section: &str) -> HashMap<String, AttributeArray> {
    let mut map = HashMap::new();
    let mut rest = section;
    let close = "</DataArray>";
    while let Some(start) = rest.find("<DataArray") {
        rest = &rest[start..];
        let te = match rest.find('>') {
            Some(e) => e + 1,
            None => break,
        };
        let tag = &rest[..te];
        let name = attr_val(tag, "Name").unwrap_or_default();
        let ncomp: usize = attr_val(tag, "NumberOfComponents")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let de = match rest.find(close) {
            Some(e) => e,
            None => break,
        };
        let data = rest[te..de].trim().to_string();
        let floats = parse_floats::<f32>(&data);
        if !name.is_empty() {
            let attr = match ncomp {
                3 => {
                    let v3: Vec<[f32; 3]> =
                        floats.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
                    if name.to_lowercase().contains("normal") {
                        AttributeArray::Normals { values: v3 }
                    } else {
                        AttributeArray::Vectors { values: v3 }
                    }
                }
                2 => AttributeArray::TextureCoords {
                    values: floats,
                    dim: 2,
                },
                n => AttributeArray::Scalars {
                    values: floats,
                    num_components: n,
                },
            };
            map.insert(name, attr);
        }
        rest = &rest[de + close.len()..];
    }
    map
}
