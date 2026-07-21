//! ASCII-inline VTI writer: `write_vti_image_data`, `write_vti_str`.

use crate::domain::vtk_data_object::VtkImageData;
use crate::io::xml_write_attr::write_attr_xml;
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
    writeln!(s, "<?xml version=\"1.0\"?>").expect("infallible write");
    writeln!(
        s,
        "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">"
    )
    .expect("infallible: validated precondition");
    writeln!(
        s,
        "  <ImageData WholeExtent=\"{}\" Origin=\"{}\" Spacing=\"{}\">",
        extent_str, origin_str, spacing_str
    )
    .expect("infallible: validated precondition");
    writeln!(s, "    <Piece Extent=\"{}\">", extent_str).expect("infallible write");

    if !grid.point_data.is_empty() {
        writeln!(s, "      <PointData>").expect("infallible write");
        for (name, attr) in &grid.point_data {
            write_attr_xml(&mut s, name, attr);
        }
        writeln!(s, "      </PointData>").expect("infallible write");
    }

    if !grid.cell_data.is_empty() {
        writeln!(s, "      <CellData>").expect("infallible write");
        for (name, attr) in &grid.cell_data {
            write_attr_xml(&mut s, name, attr);
        }
        writeln!(s, "      </CellData>").expect("infallible write");
    }

    writeln!(s, "    </Piece>").expect("infallible write");
    writeln!(s, "  </ImageData>").expect("infallible write");
    writeln!(s, "</VTKFile>").expect("infallible write");
    s
}
