//! VTK XML ImageData (.vti) reader (ASCII inline format).
//!
//! # Format Reference
//! VTK XML Format Specification v0.1, Kitware Inc.
//!
//! # Parsing Contract
//! - Finds the first `<ImageData>` tag and reads `WholeExtent`, `Origin`, `Spacing`.
//! - `<Piece>` tag is required; returns `Err` if absent.
//! - `<PointData>` and `<CellData>` are optional; absent sections yield empty maps.
//! - DataArrays with `NumberOfComponents="3"` are decoded as `Vectors` (or `Normals`
//!   when the name contains "normal"); all others are decoded as `Scalars`.
//! - No external XML parser dependency; parsing uses `str::find` / `str::rfind`.

use crate::domain::vtk_data_object::{AttributeArray, VtkImageData};
use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::path::Path;

/// Read a VTI XML (ASCII inline) file from disk into a [`VtkImageData`].
pub fn read_vti_image_data<P: AsRef<Path>>(path: P) -> Result<VtkImageData> {
    let s = std::fs::read_to_string(path.as_ref())
        .with_context(|| format!("cannot open VTI: {}", path.as_ref().display()))?;
    parse_vti(&s)
}

/// Parse an ASCII-inline VTI XML string into a [`VtkImageData`].
pub(crate) fn parse_vti(input: &str) -> Result<VtkImageData> {
    // ── ImageData opening tag ────────────────────────────────────────────────
    let image_tag = find_tag(input, "ImageData")
        .ok_or_else(|| anyhow::anyhow!("missing <ImageData> tag in VTI document"))?;

    let extent_str = attr_val(&image_tag, "WholeExtent")
        .ok_or_else(|| anyhow::anyhow!("missing WholeExtent attribute in <ImageData> tag"))?;
    let extent_vals = parse_i64s(&extent_str);
    if extent_vals.len() < 6 {
        bail!(
            "WholeExtent must contain 6 integers, got {}",
            extent_vals.len()
        );
    }
    let mut whole_extent = [0i64; 6];
    whole_extent.copy_from_slice(&extent_vals[..6]);

    let origin_str = attr_val(&image_tag, "Origin").unwrap_or_else(|| "0 0 0".to_string());
    let origin_vals = parse_f64s(&origin_str);
    let mut origin = [0.0f64; 3];
    for i in 0..3 {
        origin[i] = origin_vals.get(i).copied().unwrap_or(0.0);
    }

    let spacing_str = attr_val(&image_tag, "Spacing").unwrap_or_else(|| "1 1 1".to_string());
    let spacing_vals = parse_f64s(&spacing_str);
    let mut spacing = [1.0f64; 3];
    for i in 0..3 {
        spacing[i] = spacing_vals.get(i).copied().unwrap_or(1.0);
    }

    // ── Piece tag (required) ─────────────────────────────────────────────────
    let _piece = find_tag(input, "Piece")
        .ok_or_else(|| anyhow::anyhow!("missing <Piece> tag in VTI document"))?;

    // ── Attribute sections (optional) ────────────────────────────────────────
    let point_data = find_section(input, "PointData")
        .map(|sec| parse_attrs(&sec))
        .unwrap_or_default();
    let cell_data = find_section(input, "CellData")
        .map(|sec| parse_attrs(&sec))
        .unwrap_or_default();

    Ok(VtkImageData {
        whole_extent,
        origin,
        spacing,
        point_data,
        cell_data,
    })
}

// ── XML helpers ───────────────────────────────────────────────────────────────

/// Return the opening tag string for the first occurrence of `<tag ...>` or
/// `<tag>` in `s`, including the closing `>`.
fn find_tag(s: &str, tag: &str) -> Option<String> {
    let open = format!("<{}", tag);
    let start = s.find(&open)?;
    let end = s[start..].find('>')? + 1;
    Some(s[start..start + end].to_string())
}

/// Return the substring from the first `<tag` to the matching `</tag>` (inclusive).
fn find_section(s: &str, tag: &str) -> Option<String> {
    let open = format!("<{}", tag);
    let close = format!("</{}>", tag);
    let start = s.find(&open)?;
    let end_offset = s[start..].find(&close)? + close.len();
    Some(s[start..start + end_offset].to_string())
}

/// Parse the `name="value"` attribute from an XML tag string.
fn attr_val(tag: &str, name: &str) -> Option<String> {
    let dq = char::from(34u8); // "
    let mut pat = name.to_string();
    pat.push(char::from(61u8)); // =
    pat.push(dq);
    let start = tag.find(&pat)? + pat.len();
    let rest = &tag[start..];
    let end = rest.find(dq)?;
    Some(rest[..end].to_string())
}

/// Parse space/newline-delimited f32 values from a string slice.
fn parse_floats(s: &str) -> Vec<f32> {
    s.split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect()
}

/// Parse space/newline-delimited f64 values from a string slice.
fn parse_f64s(s: &str) -> Vec<f64> {
    s.split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect()
}

/// Parse space/newline-delimited i64 values from a string slice.
fn parse_i64s(s: &str) -> Vec<i64> {
    s.split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect()
}

/// Extract the text content of the first `<DataArray ...>...</DataArray>` in `section`.
///
/// Specified as a required helper for VTI format support; available for
/// future extensions that need to extract individual named DataArray content.
#[allow(dead_code)]
fn extract_da_content(section: &str) -> String {
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
///
/// Specified as a required helper for VTI format support; available for
/// future extensions that need to locate a specific named DataArray by name.
#[allow(dead_code)]
fn named_da(section: &str, name: &str) -> Option<String> {
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

/// Parse all `<DataArray>` elements in a PointData/CellData section into an
/// attribute map.
///
/// - `NumberOfComponents="3"` → `Vectors` (or `Normals` when name contains "normal").
/// - `NumberOfComponents="2"` → `TextureCoords` with `dim=2`.
/// - All other component counts → `Scalars` with that `num_components`.
fn parse_attrs(section: &str) -> HashMap<String, AttributeArray> {
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

// ── Binary-appended reader ────────────────────────────────────────────────────

/// Parse all appended-format `<DataArray>` elements in a PointData/CellData
/// section into an attribute map, reading binary values from `binary_block`.
///
/// Each DataArray tag must carry `format="appended"` and `offset="N"`.
/// The binary block layout at each offset: `uint32 LE` byte count followed
/// by that many bytes of `float32 LE` values.
///
/// Component interpretation mirrors [`parse_attrs`]:
/// - `NumberOfComponents="3"` → `Vectors` (or `Normals` when name contains "normal").
/// - `NumberOfComponents="2"` → `TextureCoords` with `dim=2`.
/// - All other counts → `Scalars` with that `num_components`.
fn parse_appended_attrs(
    section: &str,
    binary_block: &[u8],
) -> Result<HashMap<String, AttributeArray>> {
    let mut map = HashMap::new();
    let mut rest = section;
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
        let format = attr_val(tag, "format").unwrap_or_default();
        let offset: usize = attr_val(tag, "offset")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        if !name.is_empty() && format == "appended" {
            if offset + 4 > binary_block.len() {
                bail!(
                    "DataArray '{}': offset {} + 4 exceeds binary block length {}",
                    name,
                    offset,
                    binary_block.len()
                );
            }
            let n_bytes = u32::from_le_bytes([
                binary_block[offset],
                binary_block[offset + 1],
                binary_block[offset + 2],
                binary_block[offset + 3],
            ]) as usize;
            if offset + 4 + n_bytes > binary_block.len() {
                bail!(
                    "DataArray '{}': data region [{}..{}] exceeds binary block length {}",
                    name,
                    offset + 4,
                    offset + 4 + n_bytes,
                    binary_block.len()
                );
            }
            let data_bytes = &binary_block[offset + 4..offset + 4 + n_bytes];
            let n_floats = n_bytes / 4;
            let mut floats: Vec<f32> = Vec::with_capacity(n_floats);
            for chunk in data_bytes.chunks_exact(4) {
                floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
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
        // Advance past the current DataArray opening tag (self-closing or otherwise).
        rest = &rest[te..];
    }
    Ok(map)
}

/// Parse a binary-appended VTI byte buffer into a [`VtkImageData`].
///
/// # Format expected
/// VTK XML ImageData with `<AppendedData encoding="raw">` section.
/// The binary region begins immediately after the `_` marker that follows the
/// AppendedData opening tag.  Each DataArray block: `uint32 LE` byte count
/// then that many bytes of `float32 LE` values.
///
/// # Errors
/// Returns `Err` if the `<AppendedData>` or `_` marker is absent, if the
/// `<ImageData>` tag or its required attributes are missing, if any DataArray
/// offset/length is out of range, or if the header is not valid UTF-8.
pub fn read_vti_binary_appended_bytes(data: &[u8]) -> Result<VtkImageData> {
    // ── Locate the AppendedData block and the `_` binary marker ─────────────
    let ad_needle = b"<AppendedData";
    let ad_pos = data
        .windows(ad_needle.len())
        .position(|w| w == ad_needle)
        .ok_or_else(|| anyhow::anyhow!("no <AppendedData> tag found in binary VTI document"))?;

    // Find the closing `>` of the <AppendedData ...> opening tag.
    let gt_rel = data[ad_pos..]
        .iter()
        .position(|&b| b == b'>')
        .ok_or_else(|| anyhow::anyhow!("<AppendedData> opening tag has no closing `>`"))?;
    let after_gt = ad_pos + gt_rel + 1;

    // The `_` marker is the first `_` byte after the `>` (typically `\n_`).
    let us_rel = data[after_gt..]
        .iter()
        .position(|&b| b == b'_')
        .ok_or_else(|| {
            anyhow::anyhow!("no `_` marker found in AppendedData block after opening tag `>`")
        })?;
    let underscore_pos = after_gt + us_rel;

    // Header: all bytes strictly before the `_` marker (valid UTF-8 XML text).
    let header_bytes = &data[..underscore_pos];
    // Binary block: all bytes strictly after the `_` marker.
    let binary_block = &data[underscore_pos + 1..];

    let header_str = std::str::from_utf8(header_bytes)
        .context("VTI binary-appended header is not valid UTF-8")?;

    // ── Parse ImageData attributes ───────────────────────────────────────────
    let image_tag = find_tag(header_str, "ImageData")
        .ok_or_else(|| anyhow::anyhow!("missing <ImageData> tag in binary VTI document"))?;

    let extent_str = attr_val(&image_tag, "WholeExtent")
        .ok_or_else(|| anyhow::anyhow!("missing WholeExtent attribute in <ImageData>"))?;
    let extent_vals = parse_i64s(&extent_str);
    if extent_vals.len() < 6 {
        bail!(
            "WholeExtent must contain 6 integers, got {}",
            extent_vals.len()
        );
    }
    let mut whole_extent = [0i64; 6];
    whole_extent.copy_from_slice(&extent_vals[..6]);

    let origin_str = attr_val(&image_tag, "Origin").unwrap_or_else(|| "0 0 0".to_string());
    let origin_vals = parse_f64s(&origin_str);
    let mut origin = [0.0f64; 3];
    for i in 0..3 {
        origin[i] = origin_vals.get(i).copied().unwrap_or(0.0);
    }

    let spacing_str = attr_val(&image_tag, "Spacing").unwrap_or_else(|| "1 1 1".to_string());
    let spacing_vals = parse_f64s(&spacing_str);
    let mut spacing = [1.0f64; 3];
    for i in 0..3 {
        spacing[i] = spacing_vals.get(i).copied().unwrap_or(1.0);
    }

    // ── Parse PointData and CellData sections ────────────────────────────────
    let point_data = find_section(header_str, "PointData")
        .map(|sec| parse_appended_attrs(&sec, binary_block))
        .transpose()?
        .unwrap_or_default();

    let cell_data = find_section(header_str, "CellData")
        .map(|sec| parse_appended_attrs(&sec, binary_block))
        .transpose()?
        .unwrap_or_default();

    Ok(VtkImageData {
        whole_extent,
        origin,
        spacing,
        point_data,
        cell_data,
    })
}

/// Read a binary-appended VTI XML file from disk into a [`VtkImageData`].
///
/// Reads the entire file into memory, then delegates to
/// [`read_vti_binary_appended_bytes`].
pub fn read_vti_binary_appended<P: AsRef<Path>>(path: P) -> Result<VtkImageData> {
    let bytes = std::fs::read(path.as_ref()).with_context(|| {
        format!(
            "cannot open binary-appended VTI: {}",
            path.as_ref().display()
        )
    })?;
    read_vti_binary_appended_bytes(&bytes)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::vtk_data_object::AttributeArray;
    use crate::format::vtk::image_xml::writer::write_vti_image_data;
    use crate::format::vtk::image_xml::writer::write_vti_str;
    use tempfile::NamedTempFile;

    /// Build a minimal valid VTI XML string with the given extent, origin, spacing,
    /// and optional inline PointData/CellData blocks.
    fn make_vti(
        extent: &str,
        origin: &str,
        spacing: &str,
        point_data_block: &str,
        cell_data_block: &str,
    ) -> String {
        let mut s = String::new();
        s.push_str("<?xml version=\"1.0\"?>\n");
        s.push_str("<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
        s.push_str(&format!(
            "  <ImageData WholeExtent=\"{}\" Origin=\"{}\" Spacing=\"{}\">\n",
            extent, origin, spacing
        ));
        s.push_str(&format!("    <Piece Extent=\"{}\">\n", extent));
        s.push_str(point_data_block);
        s.push_str(cell_data_block);
        s.push_str("    </Piece>\n");
        s.push_str("  </ImageData>\n");
        s.push_str("</VTKFile>\n");
        s
    }

    #[test]
    fn test_parse_vti_whole_extent() {
        let vti = make_vti("0 3 0 4 0 5", "0 0 0", "1 1 1", "", "");
        let img = parse_vti(&vti).expect("parse must succeed");
        assert_eq!(img.whole_extent, [0, 3, 0, 4, 0, 5]);
    }

    #[test]
    fn test_parse_vti_origin_and_spacing() {
        let vti = make_vti("0 1 0 1 0 1", "1.5 2.5 3.5", "0.25 0.5 0.75", "", "");
        let img = parse_vti(&vti).expect("parse must succeed");
        assert!(
            (img.origin[0] - 1.5).abs() < 1e-9,
            "origin[0] = {} expected 1.5",
            img.origin[0]
        );
        assert!(
            (img.origin[1] - 2.5).abs() < 1e-9,
            "origin[1] = {} expected 2.5",
            img.origin[1]
        );
        assert!(
            (img.origin[2] - 3.5).abs() < 1e-9,
            "origin[2] = {} expected 3.5",
            img.origin[2]
        );
        assert!(
            (img.spacing[0] - 0.25).abs() < 1e-9,
            "spacing[0] = {} expected 0.25",
            img.spacing[0]
        );
        assert!(
            (img.spacing[1] - 0.5).abs() < 1e-9,
            "spacing[1] = {} expected 0.5",
            img.spacing[1]
        );
        assert!(
            (img.spacing[2] - 0.75).abs() < 1e-9,
            "spacing[2] = {} expected 0.75",
            img.spacing[2]
        );
    }

    #[test]
    fn test_parse_vti_point_data_scalars() {
        // extent "0 1 0 1 0 1" → n_points = 2*2*2 = 8
        let pd = concat!(
            "      <PointData>\n",
            "        <DataArray type=\"Float32\" Name=\"intensity\"",
            " NumberOfComponents=\"1\" format=\"ascii\">\n",
            "          1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0\n",
            "        </DataArray>\n",
            "      </PointData>\n"
        );
        let vti = make_vti("0 1 0 1 0 1", "0 0 0", "1 1 1", pd, "");
        let img = parse_vti(&vti).expect("parse must succeed");
        assert_eq!(img.point_data.len(), 1, "one point array expected");
        match img
            .point_data
            .get("intensity")
            .expect("intensity must be present")
        {
            AttributeArray::Scalars {
                values,
                num_components,
            } => {
                assert_eq!(*num_components, 1);
                assert_eq!(values.len(), 8);
                assert!(
                    (values[0] - 1.0f32).abs() < 1e-5,
                    "values[0] = {} expected 1.0",
                    values[0]
                );
                assert!(
                    (values[7] - 8.0f32).abs() < 1e-5,
                    "values[7] = {} expected 8.0",
                    values[7]
                );
            }
            other => panic!("expected Scalars, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_vti_multicomponent_vectors() {
        // extent "0 0 0 0 0 0" → n_points = 1; 3-component DataArray → Vectors
        let pd = concat!(
            "      <PointData>\n",
            "        <DataArray type=\"Float32\" Name=\"velocity\"",
            " NumberOfComponents=\"3\" format=\"ascii\">\n",
            "          1.0 2.0 3.0\n",
            "        </DataArray>\n",
            "      </PointData>\n"
        );
        let vti = make_vti("0 0 0 0 0 0", "0 0 0", "1 1 1", pd, "");
        let img = parse_vti(&vti).expect("parse must succeed");
        match img
            .point_data
            .get("velocity")
            .expect("velocity must be present")
        {
            AttributeArray::Vectors { values } => {
                assert_eq!(values.len(), 1, "one vector for one point");
                assert!(
                    (values[0][0] - 1.0f32).abs() < 1e-5,
                    "vx = {} expected 1.0",
                    values[0][0]
                );
                assert!(
                    (values[0][1] - 2.0f32).abs() < 1e-5,
                    "vy = {} expected 2.0",
                    values[0][1]
                );
                assert!(
                    (values[0][2] - 3.0f32).abs() < 1e-5,
                    "vz = {} expected 3.0",
                    values[0][2]
                );
            }
            other => panic!("expected Vectors, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_vti_cell_data() {
        // extent "0 1 0 1 0 1" → n_cells = 1*1*1 = 1
        let cd = concat!(
            "      <CellData>\n",
            "        <DataArray type=\"Float32\" Name=\"pressure\"",
            " NumberOfComponents=\"1\" format=\"ascii\">\n",
            "          42.0\n",
            "        </DataArray>\n",
            "      </CellData>\n"
        );
        let vti = make_vti("0 1 0 1 0 1", "0 0 0", "1 1 1", "", cd);
        let img = parse_vti(&vti).expect("parse must succeed");
        assert_eq!(img.cell_data.len(), 1, "one cell array expected");
        match img
            .cell_data
            .get("pressure")
            .expect("pressure must be present")
        {
            AttributeArray::Scalars {
                values,
                num_components,
            } => {
                assert_eq!(*num_components, 1);
                assert_eq!(values.len(), 1);
                assert!(
                    (values[0] - 42.0f32).abs() < 1e-5,
                    "values[0] = {} expected 42.0",
                    values[0]
                );
            }
            other => panic!("expected Scalars, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_vti_empty_point_data() {
        // No PointData block; point_data map must be empty.
        let vti = make_vti("0 2 0 2 0 2", "0 0 0", "1 1 1", "", "");
        let img = parse_vti(&vti).expect("parse must succeed");
        assert!(
            img.point_data.is_empty(),
            "point_data must be empty when PointData section is absent"
        );
        assert!(
            img.cell_data.is_empty(),
            "cell_data must be empty when CellData section is absent"
        );
    }

    #[test]
    fn test_read_vti_file_roundtrip() {
        // Write via write_vti_str, parse with parse_vti, verify full round-trip.
        // extent [0,1,0,1,0,1] → n_points = 8
        let mut img = VtkImageData {
            whole_extent: [0, 1, 0, 1, 0, 1],
            origin: [1.0, 2.0, 3.0],
            spacing: [0.5, 0.5, 0.5],
            ..Default::default()
        };
        img.point_data.insert(
            "scalars".to_string(),
            AttributeArray::Scalars {
                values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                num_components: 1,
            },
        );
        let xml = write_vti_str(&img);
        let parsed = parse_vti(&xml).expect("round-trip parse must succeed");

        assert_eq!(parsed.whole_extent, img.whole_extent, "extent mismatch");
        for i in 0..3 {
            assert!(
                (parsed.origin[i] - img.origin[i]).abs() < 1e-5,
                "origin[{i}] mismatch: {} vs {}",
                parsed.origin[i],
                img.origin[i]
            );
            assert!(
                (parsed.spacing[i] - img.spacing[i]).abs() < 1e-5,
                "spacing[{i}] mismatch: {} vs {}",
                parsed.spacing[i],
                img.spacing[i]
            );
        }
        match parsed
            .point_data
            .get("scalars")
            .expect("scalars must be present")
        {
            AttributeArray::Scalars {
                values,
                num_components,
            } => {
                assert_eq!(*num_components, 1);
                assert_eq!(values.len(), 8);
                let expected = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
                for (i, (&got, &exp)) in values.iter().zip(expected.iter()).enumerate() {
                    assert!(
                        (got - exp).abs() < 1e-5,
                        "scalars[{i}] = {got} expected {exp}"
                    );
                }
            }
            other => panic!("expected Scalars, got {:?}", other),
        }
    }

    #[test]
    fn test_from_file_roundtrip() {
        // Write to NamedTempFile via write_vti_image_data, read back via read_vti_image_data.
        let mut img = VtkImageData {
            whole_extent: [0, 2, 0, 3, 0, 1],
            origin: [0.0, 0.0, 0.0],
            spacing: [1.0, 1.0, 1.0],
            ..Default::default()
        };
        // n_points = (2+1)*(3+1)*(1+1) = 3*4*2 = 24
        img.point_data.insert(
            "density".to_string(),
            AttributeArray::Scalars {
                values: (0..24).map(|i| i as f32 * 0.5).collect(),
                num_components: 1,
            },
        );

        let tmp = NamedTempFile::new().expect("temp file creation must succeed");
        write_vti_image_data(tmp.path(), &img).expect("write must succeed");
        let loaded = read_vti_image_data(tmp.path()).expect("read must succeed");

        assert_eq!(loaded.whole_extent, img.whole_extent, "extent round-trip");
        let loaded_vals = match loaded
            .point_data
            .get("density")
            .expect("density must exist")
        {
            AttributeArray::Scalars { values, .. } => values.clone(),
            other => panic!("expected Scalars, got {:?}", other),
        };
        assert_eq!(loaded_vals.len(), 24, "24 scalar values expected");
        for i in 0..24 {
            let exp = i as f32 * 0.5;
            assert!(
                (loaded_vals[i] - exp).abs() < 1e-5,
                "density[{i}] = {} expected {exp}",
                loaded_vals[i]
            );
        }
    }

    #[test]
    fn test_missing_piece_tag_error() {
        // Valid ImageData tag but no Piece element — parse must return Err.
        let vti = concat!(
            "<?xml version=\"1.0\"?>\n",
            "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n",
            "  <ImageData WholeExtent=\"0 1 0 1 0 1\"",
            " Origin=\"0 0 0\" Spacing=\"1 1 1\">\n",
            "  </ImageData>\n",
            "</VTKFile>\n"
        );
        let result = parse_vti(vti);
        assert!(result.is_err(), "missing Piece tag must return Err");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("Piece"),
            "error message must mention Piece, got: {msg}"
        );
    }

    #[test]
    fn test_nonexistent_file_error() {
        let result = read_vti_image_data("/nonexistent/path/that/does/not/exist.vti");
        assert!(result.is_err(), "nonexistent file must return Err");
    }

    /// Invariant: the reader correctly parses a CellData array from
    /// binary-appended format with no PointData present.
    #[test]
    fn test_read_vti_binary_appended_cell_data_roundtrip() {
        use crate::format::vtk::image_xml::writer::write_vti_binary_appended_bytes;

        // extent [0,1,0,1,0,1] → n_cells = 1×1×1 = 1; n_points = 2×2×2 = 8
        let grid = VtkImageData {
            whole_extent: [0, 1, 0, 1, 0, 1],
            origin: [0.0, 0.0, 0.0],
            spacing: [1.0, 1.0, 1.0],
            point_data: std::collections::HashMap::new(),
            cell_data: {
                let mut m = std::collections::HashMap::new();
                m.insert(
                    "pressure".to_string(),
                    AttributeArray::Scalars {
                        values: vec![42.0f32],
                        num_components: 1,
                    },
                );
                m
            },
        };

        let bytes = write_vti_binary_appended_bytes(&grid)
            .expect("write_vti_binary_appended_bytes must succeed on cell-data-only grid");
        let parsed = read_vti_binary_appended_bytes(&bytes)
            .expect("read_vti_binary_appended_bytes must succeed on cell-data-only bytes");

        assert!(
            parsed.cell_data.contains_key("pressure"),
            "parsed must contain 'pressure' CellData key"
        );
        let values = match parsed.cell_data.get("pressure").unwrap() {
            AttributeArray::Scalars { values, .. } => values.clone(),
            other => panic!("expected Scalars variant for 'pressure', got {:?}", other),
        };
        assert_eq!(values.len(), 1, "pressure CellData must have 1 value");
        assert!(
            (values[0] - 42.0f32).abs() < 1e-6,
            "pressure[0]: expected 42.0, got {} (diff {})",
            values[0],
            (values[0] - 42.0f32).abs()
        );
    }

    /// Invariant: the reader preserves both PointData and CellData sections
    /// from a mixed binary-appended file with all values intact.
    #[test]
    fn test_read_vti_binary_appended_preserves_both_sections() {
        use crate::format::vtk::image_xml::writer::write_vti_binary_appended_bytes;

        // extent [0,1,0,1,0,0] → n_points = 2×2×1 = 4; n_cells = 1×1×1 = 1
        let grid = VtkImageData {
            whole_extent: [0, 1, 0, 1, 0, 0],
            origin: [0.0, 0.0, 0.0],
            spacing: [1.0, 1.0, 1.0],
            point_data: {
                let mut m = std::collections::HashMap::new();
                m.insert(
                    "temperature".to_string(),
                    AttributeArray::Scalars {
                        values: vec![100.0f32, 200.0, 300.0, 400.0],
                        num_components: 1,
                    },
                );
                m
            },
            cell_data: {
                let mut m = std::collections::HashMap::new();
                m.insert(
                    "flux".to_string(),
                    AttributeArray::Scalars {
                        values: vec![3.14f32],
                        num_components: 1,
                    },
                );
                m
            },
        };

        let bytes = write_vti_binary_appended_bytes(&grid)
            .expect("write_vti_binary_appended_bytes must succeed on mixed grid");
        let parsed = read_vti_binary_appended_bytes(&bytes)
            .expect("read_vti_binary_appended_bytes must succeed on mixed bytes");

        assert!(
            parsed.point_data.contains_key("temperature"),
            "parsed must contain 'temperature' PointData key"
        );
        let temp_vals = match parsed.point_data.get("temperature").unwrap() {
            AttributeArray::Scalars { values, .. } => values.clone(),
            other => panic!("expected Scalars for 'temperature', got {:?}", other),
        };
        assert_eq!(temp_vals.len(), 4, "temperature must have 4 values");
        assert!(
            (temp_vals[3] - 400.0f32).abs() < 1e-6,
            "temperature[3]: expected 400.0, got {} (diff {})",
            temp_vals[3],
            (temp_vals[3] - 400.0f32).abs()
        );

        assert!(
            parsed.cell_data.contains_key("flux"),
            "parsed must contain 'flux' CellData key"
        );
        let flux_vals = match parsed.cell_data.get("flux").unwrap() {
            AttributeArray::Scalars { values, .. } => values.clone(),
            other => panic!("expected Scalars for 'flux', got {:?}", other),
        };
        assert_eq!(flux_vals.len(), 1, "flux must have 1 value");
        assert!(
            (flux_vals[0] - 3.14f32).abs() < 1e-5,
            "flux[0]: expected 3.14, got {} (diff {})",
            flux_vals[0],
            (flux_vals[0] - 3.14f32).abs()
        );
    }
}
