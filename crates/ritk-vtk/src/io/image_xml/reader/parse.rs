//! ASCII-inline VTI reader: `read_vti_image_data`, `parse_vti`, `parse_attrs`.

use super::xml_helpers::{
    attr_val, find_section, find_tag, parse_array, parse_attrs, DEFAULT_ORIGIN_STR,
    DEFAULT_SPACING_STR,
};
use crate::domain::vtk_data_object::VtkImageData;
use anyhow::{Context, Result};
use std::path::Path;

/// Read a VTI XML (ASCII inline) file from disk into a [`VtkImageData`].
pub fn read_vti_image_data<P: AsRef<Path>>(path: P) -> Result<VtkImageData> {
    let s = std::fs::read_to_string(path.as_ref())
        .with_context(|| format!("cannot open VTI: {}", path.as_ref().display()))?;
    parse_vti(&s)
}

/// The `WholeExtent`, `Origin`, and `Spacing` of an `<ImageData>` opening tag.
///
/// `Origin` defaults to `0 0 0` and `Spacing` to `1 1 1` when absent; each
/// attribute that is present must hold exactly its 6 or 3 values.
///
/// # Errors
///
/// Returns an error for a missing `WholeExtent` or a malformed attribute.
pub(super) fn image_geometry(image_tag: &str) -> Result<([i64; 6], [f64; 3], [f64; 3])> {
    let extent_str = attr_val(image_tag, "WholeExtent")
        .ok_or_else(|| anyhow::anyhow!("missing WholeExtent attribute in <ImageData> tag"))?;
    let whole_extent: [i64; 6] = parse_array(&extent_str, "WholeExtent")?;

    let origin_str =
        attr_val(image_tag, "Origin").unwrap_or_else(|| DEFAULT_ORIGIN_STR.to_string());
    let origin: [f64; 3] = parse_array(&origin_str, "Origin")?;

    let spacing_str =
        attr_val(image_tag, "Spacing").unwrap_or_else(|| DEFAULT_SPACING_STR.to_string());
    let spacing: [f64; 3] = parse_array(&spacing_str, "Spacing")?;
    Ok((whole_extent, origin, spacing))
}

/// Parse an ASCII-inline VTI XML string into a [`VtkImageData`].
pub(crate) fn parse_vti(input: &str) -> Result<VtkImageData> {
    // ── ImageData opening tag ────────────────────────────────────────────────
    let image_tag = find_tag(input, "ImageData")
        .ok_or_else(|| anyhow::anyhow!("missing <ImageData> tag in VTI document"))?;

    let (whole_extent, origin, spacing) = image_geometry(&image_tag)?;

    // ── Piece tag (required) ─────────────────────────────────────────────────
    let _piece = find_tag(input, "Piece")
        .ok_or_else(|| anyhow::anyhow!("missing <Piece> tag in VTI document"))?;

    // ── Attribute sections (optional) ────────────────────────────────────────
    let point_data = find_section(input, "PointData")
        .map(|sec| parse_attrs(&sec))
        .transpose()?
        .unwrap_or_default();
    let cell_data = find_section(input, "CellData")
        .map(|sec| parse_attrs(&sec))
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
