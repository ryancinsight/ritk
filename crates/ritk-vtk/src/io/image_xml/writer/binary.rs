//! Binary-appended VTI writer: `write_vti_binary_appended_bytes`, helpers.

use crate::domain::vtk_data_object::{AttributeArray, VtkImageData};
use anyhow::{anyhow, Context, Result};
use std::fmt::Write;
use std::path::Path;

fn attr_ncomp(attr: &AttributeArray) -> usize {
    match attr {
        AttributeArray::Scalars { num_components, .. } => *num_components,
        AttributeArray::Vectors { .. } | AttributeArray::Normals { .. } => 3,
        AttributeArray::TextureCoords { dim, .. } => *dim,
    }
}

fn flatten_attr(attr: &AttributeArray) -> Vec<f32> {
    match attr {
        AttributeArray::Scalars { values, .. } => values.clone(),
        AttributeArray::Vectors { values } => {
            values.iter().flat_map(|v| v.iter().copied()).collect()
        }
        AttributeArray::Normals { values } => {
            values.iter().flat_map(|v| v.iter().copied()).collect()
        }
        AttributeArray::TextureCoords { values, .. } => values.clone(),
    }
}

fn write_da_appended_tag(s: &mut String, name: &str, ncomp: usize, offset: usize) {
    let dq = char::from(34u8); // "
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
    write!(s, "{}", ncomp).unwrap();
    s.push(dq);
    s.push_str(" format=");
    s.push(dq);
    s.push_str("appended");
    s.push(dq);
    s.push_str(" offset=");
    s.push(dq);
    write!(s, "{}", offset).unwrap();
    s.push(dq);
    s.push_str("/>\n");
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
///   `offset[i+1] = offset[i] + 4 + flat_values[i].len() * 4`.
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
    let flat: Vec<Vec<f32>> = all.iter().map(|(_, a)| flatten_attr(a)).collect();
    let mut offsets: Vec<usize> = Vec::with_capacity(all.len() + 1);
    offsets.push(0);
    for fv in &flat {
        let prev = *offsets.last().unwrap();
        offsets.push(prev + 4 + fv.len() * 4);
    }

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
            write_da_appended_tag(&mut xml, name, attr_ncomp(attr), offsets[i]);
        }
        writeln!(xml, "      </PointData>").unwrap();
    }

    if !cd.is_empty() {
        writeln!(xml, "      <CellData>").unwrap();
        for (i, (name, attr)) in cd.iter().enumerate() {
            write_da_appended_tag(&mut xml, name, attr_ncomp(attr), offsets[pd_len + i]);
        }
        writeln!(xml, "      </CellData>").unwrap();
    }

    writeln!(xml, "    </Piece>").unwrap();
    writeln!(xml, "  </ImageData>").unwrap();
    writeln!(xml, "  <AppendedData encoding=\"raw\">").unwrap();

    let mut result: Vec<u8> = xml.into_bytes();
    result.push(b'_');
    for fv in &flat {
        let n_bytes = (fv.len() * 4) as u32;
        result.extend_from_slice(&n_bytes.to_le_bytes());
        for &v in fv {
            result.extend_from_slice(&v.to_le_bytes());
        }
    }
    result.extend_from_slice(b"\n  </AppendedData>\n</VTKFile>\n");

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
