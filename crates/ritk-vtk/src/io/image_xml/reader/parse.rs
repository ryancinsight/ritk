//! ASCII-inline VTI reader: `read_vti_image_data`, `parse_vti`, `parse_attrs`.

use super::xml_helpers::{attr_val, find_section, find_tag, parse_f64s, parse_floats, parse_i64s};
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
    let origin = std::array::from_fn(|i| origin_vals.get(i).copied().unwrap_or(0.0));

    let spacing_str = attr_val(&image_tag, "Spacing").unwrap_or_else(|| "1 1 1".to_string());
    let spacing_vals = parse_f64s(&spacing_str);
    let spacing = std::array::from_fn(|i| spacing_vals.get(i).copied().unwrap_or(1.0));

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

/// Parse all `<DataArray>` elements in a PointData/CellData section into an
/// attribute map.
///
/// - `NumberOfComponents="3"` → `Vectors` (or `Normals` when name contains "normal").
/// - `NumberOfComponents="2"` → `TextureCoords` with `dim=2`.
/// - All other component counts → `Scalars` with that `num_components`.
pub(super) fn parse_attrs(section: &str) -> HashMap<String, AttributeArray> {
    let mut map = HashMap::new();
    let mut rest = section;
    let close = "</DataArray>";
    #[allow(clippy::while_let_loop)]
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
