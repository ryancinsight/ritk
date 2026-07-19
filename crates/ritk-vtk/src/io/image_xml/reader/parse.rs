//! ASCII-inline VTI reader: `read_vti_image_data`, `parse_vti`, `parse_attrs`.

use super::xml_helpers::{
    attr_val, find_section, find_tag, parse_attrs, parse_floats, parse_i64s, DEFAULT_ORIGIN_STR,
    DEFAULT_SPACING_STR,
};
use crate::domain::vtk_data_object::VtkImageData;
use anyhow::{bail, Context, Result};
use std::path::Path;

/// Read a VTI XML (ASCII inline) file from disk into a [`VtkImageData`].
pub fn read_vti_image_data<P: AsRef<Path>>(path: P) -> Result<VtkImageData> {
    let s = std::fs::read_to_string(path.as_ref())
        .with_context(|| format!("cannot open VTI: {}", path.as_ref().display()))?;
    parse_vti(&s)
}

/// Parse an ASCII-inline VTI XML string into a [`VtkImageData`].
pub(crate) fn parse_vti(input: &str) -> Result<VtkImageData> {
    // â”€â”€ ImageData opening tag â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

    let origin_str =
        attr_val(&image_tag, "Origin").unwrap_or_else(|| DEFAULT_ORIGIN_STR.to_string());
    let origin_vals: Vec<f64> = parse_floats(&origin_str);
    let mut origin = [0.0f64; 3];
    for (i, dst) in origin.iter_mut().enumerate() {
        *dst = origin_vals.get(i).copied().unwrap_or(0.0);
    }

    let spacing_str =
        attr_val(&image_tag, "Spacing").unwrap_or_else(|| DEFAULT_SPACING_STR.to_string());
    let spacing_vals: Vec<f64> = parse_floats(&spacing_str);
    let mut spacing = [1.0f64; 3];
    for (i, dst) in spacing.iter_mut().enumerate() {
        *dst = spacing_vals.get(i).copied().unwrap_or(1.0);
    }

    // â”€â”€ Piece tag (required) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    let _piece = find_tag(input, "Piece")
        .ok_or_else(|| anyhow::anyhow!("missing <Piece> tag in VTI document"))?;

    // â”€â”€ Attribute sections (optional) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
