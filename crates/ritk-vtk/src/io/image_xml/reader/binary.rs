//! Binary-appended VTI reader: `read_vti_binary_appended_bytes`, helpers.

use super::xml_helpers::{attr_val, find_section, find_tag, parse_f64s, parse_i64s};
use crate::domain::vtk_data_object::{AttributeArray, VtkImageData};
use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::path::Path;

/// Parse all appended-format `<DataArray>` elements in a PointData/CellData
/// section into an attribute map, reading binary values from `binary_block`.
///
/// Each DataArray tag must carry `format="appended"` and `offset="N"`.
/// The binary block layout at each offset: `uint32 LE` byte count followed
/// by that many bytes of `float32 LE` values.
///
/// Component interpretation mirrors `parse_attrs`:
/// - `NumberOfComponents="3"` → `Vectors` (or `Normals` when name contains "normal").
/// - `NumberOfComponents="2"` → `TextureCoords` with `dim=2`.
/// - All other counts → `Scalars` with that `num_components`.
fn parse_appended_attrs(
    section: &str,
    binary_block: &[u8],
) -> Result<HashMap<String, AttributeArray>> {
    let mut map = HashMap::new();
    let mut rest = section;
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
    let origin = std::array::from_fn(|i| origin_vals.get(i).copied().unwrap_or(0.0));

    let spacing_str = attr_val(&image_tag, "Spacing").unwrap_or_else(|| "1 1 1".to_string());
    let spacing_vals = parse_f64s(&spacing_str);
    let spacing = std::array::from_fn(|i| spacing_vals.get(i).copied().unwrap_or(1.0));

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
