//! Shared XML attribute writer for VTK XML formats (VTI, VTP, VTU).
//!
//! Emits a single `<DataArray>` element for an [`AttributeArray`] in ASCII-inline format.

use crate::domain::vtk_data_object::AttributeArray;
use std::fmt::Write;

/// Emit a single attribute `<DataArray>` element into `s`.
pub(crate) fn write_attr_xml(s: &mut String, name: &str, attr: &AttributeArray) {
    let dq = char::from(34u8); // "
    let gt = char::from(62u8); // >

    let hdr = |ncomp: usize| -> String {
        let mut h = String::from(" <DataArray type=");
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
            write!(s, " ").unwrap();
            for x in values {
                write!(s, " {:.6}", x).unwrap();
            }
            writeln!(s).unwrap();
            writeln!(s, " </DataArray>").unwrap();
        }
        AttributeArray::Vectors { values } => {
            writeln!(s, "{}", hdr(3)).unwrap();
            write!(s, " ").unwrap();
            for [x, y, z] in values {
                write!(s, " {:.6} {:.6} {:.6}", x, y, z).unwrap();
            }
            writeln!(s).unwrap();
            writeln!(s, " </DataArray>").unwrap();
        }
        AttributeArray::Normals { values } => {
            writeln!(s, "{}", hdr(3)).unwrap();
            write!(s, " ").unwrap();
            for [x, y, z] in values {
                write!(s, " {:.6} {:.6} {:.6}", x, y, z).unwrap();
            }
            writeln!(s).unwrap();
            writeln!(s, " </DataArray>").unwrap();
        }
        AttributeArray::TextureCoords { values, dim } => {
            writeln!(s, "{}", hdr(*dim)).unwrap();
            write!(s, " ").unwrap();
            for x in values {
                write!(s, " {:.6}", x).unwrap();
            }
            writeln!(s).unwrap();
            writeln!(s, " </DataArray>").unwrap();
        }
    }
}
