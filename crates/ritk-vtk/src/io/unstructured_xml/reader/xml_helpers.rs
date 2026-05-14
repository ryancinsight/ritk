//! XML parsing helpers for the VTU ASCII reader.

use anyhow::{Context, Result};
use std::collections::HashMap;
use crate::domain::vtk_data_object::AttributeArray;

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
    let dq = char::from(34u8); // "
    let mut pat = name.to_string();
    pat.push(char::from(61u8)); // =
    pat.push(dq);
    let start = tag.find(&pat)? + pat.len();
    let rest = &tag[start..];
    let end = rest.find(dq)?;
    Some(rest[..end].to_string())
}

/// Parse a `usize` attribute from an XML tag string.
pub(crate) fn attr_usize(tag: &str, name: &str) -> Result<usize> {
    let v = attr_val(tag, name)
        .ok_or_else(|| anyhow::anyhow!("attribute '{}' not found in tag: {}", name, tag))?;
    v.parse()
        .with_context(|| format!("cannot parse attribute '{}' as usize: {}", name, v))
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
    let dq = char::from(34u8);
    let mut np = String::from("Name=");
    np.push(dq);
    np.push_str(name);
    np.push(dq);
    let attr_pos = section.find(&np)?;
    let da_start = section[..attr_pos].rfind("<DataArray")?;
    let rest = &section[da_start..];
    let close = "</DataArray>";
    let end = rest.find(close)? + close.len();
    Some(rest[..end].to_string())
}

/// Parse space/newline-delimited f32 values from a string slice.
pub(crate) fn parse_floats(s: &str) -> Vec<f32> {
    s.split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect()
}

/// Parse space/newline-delimited i32 values from a string slice.
pub(crate) fn parse_ints(s: &str) -> Vec<i32> {
    s.split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect()
}

/// Parse all `<DataArray>` elements in a PointData/CellData section into an
/// attribute map.  Arrays with `NumberOfComponents="3"` are decoded as Vectors
/// (or Normals if the name contains "normal"); others are decoded as Scalars.
pub(crate) fn parse_attrs(section: &str) -> HashMap<String, AttributeArray> {
    let mut map = HashMap::new();
    let mut rest = section;
    let close = "</DataArray>";
    loop {
        let start = match rest.find("<DataArray") {
            Some(s) => s,
            None => break,
        };
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
        let floats = parse_floats(&data);
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
