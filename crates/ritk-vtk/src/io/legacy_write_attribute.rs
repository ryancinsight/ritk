//! Shared legacy VTK attribute writer for VTK ASCII/legacy formats.
//!
//! Writes a single attribute section (SCALARS, VECTORS, NORMALS,
//! TEXTURE_COORDINATES) into a `dyn Write` sink using the VTK legacy
//! keyword format.

use crate::domain::vtk_data_object::AttributeArray;
use anyhow::Result;
use std::io::Write as IoWrite;

/// Emit a single attribute section in VTK legacy ASCII format.
///
/// # Format
/// - Scalars: `SCALARS <name> float <ncomp>` + `LOOKUP_TABLE default` + values
/// - Vectors: `VECTORS <name> float` + triplets
/// - Normals: `NORMALS <name> float` + triplets
/// - TextureCoords: `TEXTURE_COORDINATES <name> float <dim>` + chunks
pub(crate) fn write_attribute_legacy(
    w: &mut dyn IoWrite,
    name: &str,
    attr: &AttributeArray,
) -> Result<()> {
    match attr {
        AttributeArray::Scalars {
            values,
            num_components,
        } => {
            writeln!(w, "SCALARS {} float {}", name, num_components)?;
            writeln!(w, "LOOKUP_TABLE default")?;
            for v in values {
                writeln!(w, "{}", v)?;
            }
        }
        AttributeArray::Vectors { values } => {
            writeln!(w, "VECTORS {} float", name)?;
            for [x, y, z] in values {
                writeln!(w, "{} {} {}", x, y, z)?;
            }
        }
        AttributeArray::Normals { values } => {
            writeln!(w, "NORMALS {} float", name)?;
            for [x, y, z] in values {
                writeln!(w, "{} {} {}", x, y, z)?;
            }
        }
        AttributeArray::TextureCoords { values, dim } => {
            writeln!(w, "TEXTURE_COORDINATES {} float {}", name, dim)?;
            for chunk in values.chunks(*dim) {
                let parts: Vec<String> = chunk.iter().map(|v| v.to_string()).collect();
                writeln!(w, "{}", parts.join(" "))?;
            }
        }
    }
    Ok(())
}
