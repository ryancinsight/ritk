//! Binary-appended VTI writer: `write_vti_binary_appended_bytes`, helpers.

use crate::domain::vtk_data_object::{AttributeArray, VtkImageData};
use anyhow::{anyhow, Context, Result};
use std::fmt::Write;
use std::path::Path;

const APPENDED_CLOSING_TAG: &[u8] = b"\n  </AppendedData>\n</VTKFile>\n";

fn attr_ncomp(attr: &AttributeArray) -> usize {
    match attr {
        AttributeArray::Scalars { num_components, .. } => *num_components,
        AttributeArray::Vectors { .. } | AttributeArray::Normals { .. } => 3,
        AttributeArray::TextureCoords { dim, .. } => *dim,
    }
}

fn attr_value_len(attr: &AttributeArray) -> Result<usize> {
    match attr {
        AttributeArray::Scalars { values, .. } | AttributeArray::TextureCoords { values, .. } => {
            Ok(values.len())
        }
        AttributeArray::Vectors { values } | AttributeArray::Normals { values } => values
            .len()
            .checked_mul(3)
            .ok_or_else(|| anyhow!("VTI appended vector component count overflow")),
    }
}

fn attr_block_len(attr: &AttributeArray) -> Result<usize> {
    let value_bytes = attr_value_len(attr)?
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| anyhow!("VTI appended DataArray byte count overflow"))?;
    if value_bytes > u32::MAX as usize {
        return Err(anyhow!(
            "VTI appended DataArray has {} bytes, exceeding UInt32 header limit {}",
            value_bytes,
            u32::MAX
        ));
    }
    value_bytes
        .checked_add(std::mem::size_of::<u32>())
        .ok_or_else(|| anyhow!("VTI appended DataArray block length overflow"))
}

fn write_da_appended_tag(s: &mut String, name: &str, ncomp: usize, offset: usize) -> Result<()> {
    let dq = '"';
    s.push_str("        <DataArray type=");
    s.push(dq);
    s.push_str("Float32");
    s.push(dq);
    s.push_str(" Name=");
    s.push(dq);
    s.push_str(name);
    s.push(dq);
    s.push_str(" NumberOfComponents=");
    s.push(dq);
    write!(s, "{}", ncomp)?;
    s.push(dq);
    s.push_str(" format=");
    s.push(dq);
    s.push_str("appended");
    s.push(dq);
    s.push_str(" offset=");
    s.push(dq);
    write!(s, "{}", offset)?;
    s.push(dq);
    s.push_str("/>\n");
    Ok(())
}

fn write_attr_appended_block(out: &mut Vec<u8>, attr: &AttributeArray) -> Result<()> {
    let value_bytes = attr_value_len(attr)?
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| anyhow!("VTI appended DataArray byte count overflow"))?;
    let value_bytes_u32 = u32::try_from(value_bytes).map_err(|_| {
        anyhow!(
            "VTI appended DataArray has {} bytes, exceeding UInt32 header limit {}",
            value_bytes,
            u32::MAX
        )
    })?;
    out.extend_from_slice(&value_bytes_u32.to_le_bytes());
    match attr {
        AttributeArray::Scalars { values, .. } | AttributeArray::TextureCoords { values, .. } => {
            write_f32_values(out, values.iter().copied());
        }
        AttributeArray::Vectors { values } | AttributeArray::Normals { values } => {
            for value in values {
                write_f32_values(out, value.iter().copied());
            }
        }
    }
    Ok(())
}

fn write_f32_values(out: &mut Vec<u8>, values: impl IntoIterator<Item = f32>) {
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
}

/// Serialize a [`VtkImageData`] to a binary-appended VTI byte buffer.
///
/// # Format
/// Produces a VTK XML ImageData document with `encoding="raw"` AppendedData.
/// Each DataArray block in the binary region is: `uint32 LE` byte-count header
/// followed by that many bytes of `float32 LE` values.
///
/// # Invariants
/// - Validates the grid before serialization; returns `Err` on any violation.
/// - DataArrays within each section are sorted by name (lexicographic) for
///   deterministic offset computation and reproducible output.
/// - Offsets satisfy: `offset[0] = 0`,
///   `offset[i+1] = offset[i] + 4 + value_count[i] * 4`.
/// - Appended blocks are streamed from the source attribute arrays; no
///   duplicate flattened `Vec<f32>` is allocated for offset computation or
///   binary emission.
pub fn write_vti_binary_appended_bytes(grid: &VtkImageData) -> Result<Vec<u8>> {
    grid.validate().map_err(|e| anyhow!("{}", e))?;

    let e = &grid.whole_extent;
    let extent_str = format!("{} {} {} {} {} {}", e[0], e[1], e[2], e[3], e[4], e[5]);
    let origin_str = format!(
        "{:.6} {:.6} {:.6}",
        grid.origin[0], grid.origin[1], grid.origin[2]
    );
    let spacing_str = format!(
        "{:.6} {:.6} {:.6}",
        grid.spacing[0], grid.spacing[1], grid.spacing[2]
    );

    let mut pd: Vec<(&str, &AttributeArray)> = grid
        .point_data
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect();
    pd.sort_unstable_by(|a, b| a.0.cmp(b.0));

    let mut cd: Vec<(&str, &AttributeArray)> = grid
        .cell_data
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect();
    cd.sort_unstable_by(|a, b| a.0.cmp(b.0));

    let all: Vec<(&str, &AttributeArray)> = pd.iter().chain(cd.iter()).copied().collect();
    let mut offsets: Vec<usize> = Vec::with_capacity(all.len() + 1);
    offsets.push(0);
    for (_, attr) in &all {
        let prev = *offsets.last().unwrap();
        let next = prev
            .checked_add(attr_block_len(attr)?)
            .ok_or_else(|| anyhow!("VTI appended offset overflow"))?;
        offsets.push(next);
    }
    let appended_len = *offsets.last().unwrap_or(&0);

    let mut xml = String::new();
    writeln!(xml, "<?xml version=\"1.0\"?>").unwrap();
    writeln!(
        xml,
        "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\" header_type=\"UInt32\">"
    )
    .unwrap();
    writeln!(
        xml,
        "  <ImageData WholeExtent=\"{}\" Origin=\"{}\" Spacing=\"{}\">",
        extent_str, origin_str, spacing_str
    )
    .unwrap();
    writeln!(xml, "    <Piece Extent=\"{}\">", extent_str).unwrap();

    let pd_len = pd.len();
    if !pd.is_empty() {
        writeln!(xml, "      <PointData>").unwrap();
        for (i, (name, attr)) in pd.iter().enumerate() {
            write_da_appended_tag(&mut xml, name, attr_ncomp(attr), offsets[i])?;
        }
        writeln!(xml, "      </PointData>").unwrap();
    }

    if !cd.is_empty() {
        writeln!(xml, "      <CellData>").unwrap();
        for (i, (name, attr)) in cd.iter().enumerate() {
            write_da_appended_tag(&mut xml, name, attr_ncomp(attr), offsets[pd_len + i])?;
        }
        writeln!(xml, "      </CellData>").unwrap();
    }

    writeln!(xml, "    </Piece>").unwrap();
    writeln!(xml, "  </ImageData>").unwrap();
    writeln!(xml, "  <AppendedData encoding=\"raw\">").unwrap();

    let mut result: Vec<u8> =
        Vec::with_capacity(xml.len() + 1 + appended_len + APPENDED_CLOSING_TAG.len());
    result.extend_from_slice(xml.as_bytes());
    result.push(b'_');
    for (_, attr) in &all {
        write_attr_appended_block(&mut result, attr)?;
    }
    result.extend_from_slice(APPENDED_CLOSING_TAG);

    Ok(result)
}

/// Write a [`VtkImageData`] to a binary-appended VTI XML file.
///
/// Validates the grid before writing. Uses `encoding="raw"` AppendedData format.
/// Returns `Err` on validation failure or I/O error.
pub fn write_vti_binary_appended_to_file<P: AsRef<Path>>(
    path: P,
    grid: &VtkImageData,
) -> Result<()> {
    let bytes = write_vti_binary_appended_bytes(grid)?;
    std::fs::write(path.as_ref(), &bytes).with_context(|| {
        format!(
            "cannot write binary-appended VTI: {}",
            path.as_ref().display()
        )
    })
}
