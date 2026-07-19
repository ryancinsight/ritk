//! OBJ ASCII writer â† VtkPolyData.
//!
//! Output format:
//! ```text
//! # Written by RITK
//!
//! v x y z
//! â€¦
//! [vn nx ny nz â€¦]   (only if point_data["Normals"] is present)
//!
//! f v1[//n1] v2[//n2] v3[//n3] â€¦
//! â€¦
//! ```
//! Indices are 1-based as required by the OBJ specification.

use crate::domain::vtk_data_object::{AttributeArray, VtkPolyData};
use anyhow::Result;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Write `poly` to an OBJ file at `path`.
pub fn write_obj_mesh(path: impl AsRef<Path>, poly: &VtkPolyData) -> Result<()> {
    let file = std::fs::File::create(path.as_ref())?;
    write_obj_to_writer(&mut BufWriter::new(file), poly)
}

/// Write OBJ to any [`Write`] sink.  Exposed for in-memory testing.
pub(crate) fn write_obj_to_writer(w: &mut impl Write, poly: &VtkPolyData) -> Result<()> {
    writeln!(w, "# Written by RITK")?;
    writeln!(w)?;

    for [x, y, z] in &poly.points {
        writeln!(w, "v {x} {y} {z}")?;
    }

    // Extract normals if present; emit a `vn` line per point.
    let normal_values: Option<&Vec<[f32; 3]>> =
        poly.point_data.get("Normals").and_then(|a| match a {
            AttributeArray::Normals { values } => Some(values),
            _ => None,
        });

    if let Some(normals) = normal_values {
        writeln!(w)?;
        for [nx, ny, nz] in normals {
            writeln!(w, "vn {nx} {ny} {nz}")?;
        }
    }

    let with_normals = normal_values.is_some();
    writeln!(w)?;

    for cell in &poly.polygons {
        write!(w, "f")?;
        for &idx in cell {
            let one = idx + 1;
            if with_normals {
                // Vertex index and normal index are always identical in our
                // convention (i-th point â†’ i-th normal).
                write!(w, " {one}//{one}")?;
            } else {
                write!(w, " {one}")?;
            }
        }
        writeln!(w)?;
    }

    w.flush()?;
    Ok(())
}
