//! Binary-appended VTI reader: `read_vti_binary_appended_bytes`, helpers.

use super::parse::image_geometry;
use super::xml_helpers::{attr_val, find_section, find_tag};
use crate::domain::vtk_data_object::{AttributeArray, VtkImageData};
use crate::io::xml_array::{
    attribute_from_values, attribute_values, check_value_count, component_count,
    declared_sample_type,
};
use anyhow::{anyhow, bail, Context, Result};
use consus_core::ByteOrder;
use ritk_codecs::sample::SampleBuffer;
use std::collections::HashMap;
use std::path::Path;

/// Byte order and length-prefix width of an appended block, from `<VTKFile>`.
///
/// Sources (Kitware/VTK at 1fb31d6): the XML file-format documentation
/// (`Documentation/docs/vtk_file_formats/vtkxml_file_format.md`) defines
/// `byte_order` as the order of the stored data and places each appended array
/// at its `offset` after the `_` that opens `<AppendedData>`;
/// `IO/XMLParser/vtkXMLDataParser.cxx` lines 229-262 accept `BigEndian` and
/// `LittleEndian` for `byte_order` and `UInt32` and `UInt64` for `header_type`,
/// the width of the length prefix, and refuse any other value.
#[derive(Clone, Copy)]
struct AppendedLayout {
    order: ByteOrder,
    /// Bytes of the length prefix: 4 for `header_type="UInt32"`, 8 for `"UInt64"`.
    header_width: usize,
}

impl AppendedLayout {
    /// Read `byte_order` (default `LittleEndian`) and `header_type` (default
    /// `UInt32`) from the `<VTKFile>` tag of `header`.
    ///
    /// # Errors
    ///
    /// Returns an error for any other value of either attribute, and for a
    /// `compressor`, which this reader does not implement.
    fn from_header(header: &str) -> Result<Self> {
        let tag = find_tag(header, "VTKFile").unwrap_or_default();
        let order = match attr_val(&tag, "byte_order").as_deref() {
            None | Some("LittleEndian") => ByteOrder::LittleEndian,
            Some("BigEndian") => ByteOrder::BigEndian,
            Some(other) => bail!("unsupported VTK XML byte_order: {other}"),
        };
        let header_width = match attr_val(&tag, "header_type").as_deref() {
            None | Some("UInt32") => 4,
            Some("UInt64") => 8,
            Some(other) => bail!("unsupported VTK XML header_type: {other}"),
        };
        if let Some(compressor) = attr_val(&tag, "compressor") {
            bail!("compressed appended data (compressor=\"{compressor}\") is not implemented");
        }
        Ok(Self {
            order,
            header_width,
        })
    }

    /// The byte count stored in the length prefix `bytes`.
    fn byte_count(self, bytes: &[u8]) -> Result<usize> {
        let count = match (self.header_width, self.order) {
            (4, ByteOrder::LittleEndian) => u64::from(u32::from_le_bytes(bytes.try_into()?)),
            (4, ByteOrder::BigEndian) => u64::from(u32::from_be_bytes(bytes.try_into()?)),
            (_, ByteOrder::LittleEndian) => u64::from_le_bytes(bytes.try_into()?),
            (_, ByteOrder::BigEndian) => u64::from_be_bytes(bytes.try_into()?),
        };
        usize::try_from(count).context("appended block length exceeds usize")
    }
}

/// Parse all appended-format `<DataArray>` elements in a PointData/CellData
/// section into an attribute map, reading binary values from `binary_block`.
///
/// Each DataArray tag must carry `format="appended"` and `offset="N"`. The
/// binary block layout at each offset: a byte count in the file's header type
/// and byte order, followed by that many bytes of samples in the array's
/// declared `type`, which convert to `f32`. The decoded value count must be a
/// whole number of tuples and equal `NumberOfTuples * NumberOfComponents` when
/// the tag declares `NumberOfTuples` ([`check_value_count`]).
///
/// Component interpretation is that of [`attribute_from_values`]:
/// - `NumberOfComponents="3"` → `Vectors` (or `Normals` when name contains "normal").
/// - `NumberOfComponents="2"` → `TextureCoords` with `dim=2`.
/// - All other counts → `Scalars` with that `num_components`.
fn parse_appended_attrs(
    section: &str,
    binary_block: &[u8],
    layout: AppendedLayout,
) -> Result<HashMap<String, AttributeArray>> {
    let mut map = HashMap::new();
    let mut rest = section;
    while let Some(start) = rest.find("<DataArray") {
        rest = &rest[start..];
        let Some(te) = rest.find('>').map(|e| e + 1) else {
            break;
        };
        let tag = &rest[..te];
        let name = attr_val(tag, "Name").unwrap_or_default();
        if !name.is_empty() {
            let format = attr_val(tag, "format").unwrap_or_default();
            if format != "appended" {
                bail!(
                    "DataArray '{name}' has format=\"{format}\"; the appended reader supports format=\"appended\" only"
                );
            }
            let offset: usize = attr_val(tag, "offset")
                .ok_or_else(|| anyhow!("DataArray '{name}' has no offset attribute"))?
                .parse()
                .with_context(|| format!("DataArray '{name}': bad offset"))?;
            let values = read_appended_attribute(binary_block, offset, layout, tag)
                .with_context(|| format!("DataArray '{name}'"))?;
            map.insert(
                name.clone(),
                attribute_from_values(&name, component_count(tag)?, values)?,
            );
        }
        // Advance past the current DataArray opening tag (self-closing or otherwise).
        rest = &rest[te..];
    }
    Ok(map)
}

/// The samples of the appended block at `offset`, decoded in the type `tag`
/// declares and converted to `f32`.
fn read_appended_attribute(
    binary_block: &[u8],
    offset: usize,
    layout: AppendedLayout,
    tag: &str,
) -> Result<Vec<f32>> {
    let data_start = offset
        .checked_add(layout.header_width)
        .filter(|&end| end <= binary_block.len())
        .ok_or_else(|| {
            anyhow!(
                "offset {offset} + {} exceeds binary block length {}",
                layout.header_width,
                binary_block.len()
            )
        })?;
    let n_bytes = layout.byte_count(&binary_block[offset..data_start])?;
    let data_end = data_start
        .checked_add(n_bytes)
        .filter(|&end| end <= binary_block.len())
        .ok_or_else(|| {
            anyhow!(
                "data region [{data_start}..+{n_bytes}] exceeds binary block length {}",
                binary_block.len()
            )
        })?;
    let samples = SampleBuffer::decode(
        &binary_block[data_start..data_end],
        declared_sample_type(tag)?,
        layout.order,
    )?;
    check_value_count(tag, samples.len())
        .with_context(|| format!("appended block of {n_bytes} bytes at offset {offset}"))?;
    attribute_values(samples)
}

/// Parse a binary-appended VTI byte buffer into a [`VtkImageData`].
///
/// # Format expected
/// VTK XML ImageData with `<AppendedData encoding="raw">` section.
/// The binary region begins immediately after the `_` marker that follows the
/// AppendedData opening tag.  Each DataArray block: a byte count (`header_type`,
/// default `UInt32`, in the file's `byte_order`, default little-endian) then
/// that many bytes of samples in the array's declared `type`, converted to
/// `f32`.
///
/// # Errors
/// Returns `Err` if the `<AppendedData>` or `_` marker is absent, if the
/// `<ImageData>` tag or its required attributes are missing, if any DataArray
/// offset/length is out of range, if a decoded value count contradicts its
/// `DataArray` tag, if the header is not valid UTF-8, or if the file uses base64
/// appended data or a compressor.
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
    let appended_tag = String::from_utf8_lossy(&data[ad_pos..after_gt]);
    if let Some(encoding) = attr_val(&appended_tag, "encoding") {
        if encoding != "raw" {
            bail!("AppendedData encoding=\"{encoding}\" is not implemented; only \"raw\" is");
        }
    }
    let layout = AppendedLayout::from_header(header_str)?;

    // ── Parse ImageData attributes ───────────────────────────────────────────
    let image_tag = find_tag(header_str, "ImageData")
        .ok_or_else(|| anyhow::anyhow!("missing <ImageData> tag in binary VTI document"))?;

    let (whole_extent, origin, spacing) = image_geometry(&image_tag)?;

    // ── Parse PointData and CellData sections ────────────────────────────────
    let point_data = find_section(header_str, "PointData")
        .map(|sec| parse_appended_attrs(&sec, binary_block, layout))
        .transpose()?
        .unwrap_or_default();

    let cell_data = find_section(header_str, "CellData")
        .map(|sec| parse_appended_attrs(&sec, binary_block, layout))
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
