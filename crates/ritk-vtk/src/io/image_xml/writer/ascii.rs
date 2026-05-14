//! ASCII-inline VTI writer: `write_vti_image_data`, `write_vti_str`, `write_attr`.

use crate::domain::vtk_data_object::{AttributeArray, VtkImageData};
use anyhow::{anyhow, Context, Result};
use std::fmt::Write;
use std::path::Path;

/// Write a [`VtkImageData`] to an ASCII-inline VTI XML file.
///
/// Validates the grid before writing; returns `Err` on any invariant violation
/// or I/O failure.
pub fn write_vti_image_data<P: AsRef<Path>>(path: P, grid: &VtkImageData) -> Result<()> {
    grid.validate().map_err(|e| anyhow!("{}", e))?;
    std::fs::write(path.as_ref(), write_vti_str(grid).as_bytes())
        .with_context(|| format!("cannot write VTI: {}", path.as_ref().display()))
}

/// Serialize a [`VtkImageData`] to an ASCII-inline VTI XML string.
///
/// Does not validate the grid; use `write_vti_image_data` for validated
/// file output.
pub fn write_vti_str(grid: &VtkImageData) -> String {
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

    let mut s = String::new();
    writeln!(s, "<?xml version=\"1.0\"?>").unwrap();
    writeln!(
        s,
        "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">"
    )
    .unwrap();
    writeln!(
        s,
        "  <ImageData WholeExtent=\"{}\" Origin=\"{}\" Spacing=\"{}\">",
        extent_str, origin_str, spacing_str
    )
    .unwrap();
    writeln!(s, "    <Piece Extent=\"{}\">", extent_str).unwrap();

    if !grid.point_data.is_empty() {
        writeln!(s, "      <PointData>").unwrap();
        for (name, attr) in &grid.point_data {
            write_attr(&mut s, name, attr);
        }
        writeln!(s, "      </PointData>").unwrap();
    }

    if !grid.cell_data.is_empty() {
        writeln!(s, "      <CellData>").unwrap();
        for (name, attr) in &grid.cell_data {
            write_attr(&mut s, name, attr);
        }
        writeln!(s, "      </CellData>").unwrap();
    }

    writeln!(s, "    </Piece>").unwrap();
    writeln!(s, "  </ImageData>").unwrap();
    writeln!(s, "</VTKFile>").unwrap();
    s
}

/// Emit a single attribute `<DataArray>` element into `s`.
pub(super) fn write_attr(s: &mut String, name: &str, attr: &AttributeArray) {
    let dq = char::from(34u8); // "
    let gt = char::from(62u8); // >

    let hdr = |ncomp: usize| -> String {
        let mut h = String::from("        <DataArray type=");
        h.push(dq);
        h.push_str("Float32");
        h.push(dq);
        h.push_str(" Name=");
        h.push(dq);
        h.push_str(name);
        h.push(dq);
        h.push_str(" NumberOfComponents=");
        h.push(dq);
        h.push_str(&ncomp.to_string());
        h.push(dq);
        h.push_str(" format=");
        h.push(dq);
        h.push_str("ascii");
        h.push(dq);
        h.push(gt);
        h
    };

    match attr {
        AttributeArray::Scalars {
            values,
            num_components,
        } => {
            writeln!(s, "{}", hdr(*num_components)).unwrap();
            write!(s, "       ").unwrap();
            for x in values {
                write!(s, " {:.6}", x).unwrap();
            }
            writeln!(s).unwrap();
            writeln!(s, "        </DataArray>").unwrap();
        }
        AttributeArray::Vectors { values } => {
            writeln!(s, "{}", hdr(3)).unwrap();
            write!(s, "       ").unwrap();
            for [x, y, z] in values {
                write!(s, " {:.6} {:.6} {:.6}", x, y, z).unwrap();
            }
            writeln!(s).unwrap();
            writeln!(s, "        </DataArray>").unwrap();
        }
        AttributeArray::Normals { values } => {
            writeln!(s, "{}", hdr(3)).unwrap();
            write!(s, "       ").unwrap();
            for [x, y, z] in values {
                write!(s, " {:.6} {:.6} {:.6}", x, y, z).unwrap();
            }
            writeln!(s).unwrap();
            writeln!(s, "        </DataArray>").unwrap();
        }
        AttributeArray::TextureCoords { values, dim } => {
            writeln!(s, "{}", hdr(*dim)).unwrap();
            write!(s, "       ").unwrap();
            for x in values {
                write!(s, " {:.6}", x).unwrap();
            }
            writeln!(s).unwrap();
            writeln!(s, "        </DataArray>").unwrap();
        }
    }
}
