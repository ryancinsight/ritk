//! Shared XML attribute writer for VTK XML formats (VTI, VTP, VTU).
//!
//! Emits a single `<DataArray>` element for an [`AttributeArray`] in ASCII-inline format.

use crate::domain::vtk_data_object::AttributeArray;
use std::fmt::Write;

/// Emit a single attribute `<DataArray>` element into `s`.
pub(crate) fn write_attr_xml(s: &mut String, name: &str, attr: &AttributeArray) {
    let hdr = |ncomp: usize| -> String {
        format!(
            " <DataArray type=\"Float32\" Name=\"{}\" NumberOfComponents=\"{}\" format=\"ascii\">",
            name, ncomp
        )
    };

    match attr {
        AttributeArray::Scalars {
            values,
            num_components,
        } => {
            writeln!(s, "{}", hdr(*num_components)).expect("infallible write");
            write!(s, " ").expect("infallible write");
            for x in values {
                write!(s, " {:.6}", x).expect("infallible write");
            }
            writeln!(s).expect("infallible write");
            writeln!(s, " </DataArray>").expect("infallible write");
        }
        AttributeArray::Vectors { values } => {
            writeln!(s, "{}", hdr(3)).expect("infallible write");
            write!(s, " ").expect("infallible write");
            for [x, y, z] in values {
                write!(s, " {:.6} {:.6} {:.6}", x, y, z).expect("infallible write");
            }
            writeln!(s).expect("infallible write");
            writeln!(s, " </DataArray>").expect("infallible write");
        }
        AttributeArray::Normals { values } => {
            writeln!(s, "{}", hdr(3)).expect("infallible write");
            write!(s, " ").expect("infallible write");
            for [x, y, z] in values {
                write!(s, " {:.6} {:.6} {:.6}", x, y, z).expect("infallible write");
            }
            writeln!(s).expect("infallible write");
            writeln!(s, " </DataArray>").expect("infallible write");
        }
        AttributeArray::TextureCoords { values, dim } => {
            writeln!(s, "{}", hdr(*dim)).expect("infallible write");
            write!(s, " ").expect("infallible write");
            for x in values {
                write!(s, " {:.6}", x).expect("infallible write");
            }
            writeln!(s).expect("infallible write");
            writeln!(s, " </DataArray>").expect("infallible write");
        }
    }
}
