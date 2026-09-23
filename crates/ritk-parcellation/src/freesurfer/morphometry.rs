//! FreeSurfer per-vertex scalar files (`lh.curv`, `lh.thickness`, `lh.sulc`).
//!
//! FreeSurfer calls these "curv" files whatever they hold; nibabel calls them
//! morphometry files, the name used here because most of them are not
//! curvature. Each carries one `f32` per vertex of the surface it was computed
//! on.
//!
//! # Format
//!
//! The new format, per FreeSurfer's `matlab/read_curv.m`
//! (<https://github.com/freesurfer/freesurfer/blob/dev/matlab/read_curv.m>) and
//! nibabel's `freesurfer/io.py` `read_morph_data`/`write_morph_data`
//! (<https://github.com/nipy/nibabel/blob/master/nibabel/freesurfer/io.py>),
//! big-endian throughout:
//!
//! ```text
//! magic             3 bytes, 0xFFFFFF
//! vertex count      i32
//! face count        i32
//! values per vertex i32, always 1
//! values            f32 × vertex count
//! ```
//!
//! The pre-2000 old format (three-byte counts and `i16` values scaled by 100)
//! has no magic of its own — its first three bytes are the vertex count — so
//! accepting it would make every wrong file parse as something. It is rejected
//! as an [`FreeSurferError::InvalidMagic`].

use std::io::{Read, Write};

use super::big_endian::{
    read_be, read_count, read_u24, reserve_for, write_be, write_count, write_u24,
};
use super::surface::MAX_ELEMENTS;
use super::{FreeSurferError, FreeSurferFormat};

const FORMAT: FreeSurferFormat = FreeSurferFormat::Morphometry;

/// Three-byte magic of the new format (`NEW_VERSION_MAGIC_NUMBER`, 16777215).
const MAGIC: u32 = 0x00FF_FFFF;

/// One scalar per surface vertex, with the face count of that surface.
#[derive(Debug, Clone, PartialEq)]
pub struct Morphometry {
    values: Box<[f32]>,
    face_count: usize,
}

impl Morphometry {
    /// Values for a surface of `face_count` faces.
    #[must_use]
    pub const fn new(values: Box<[f32]>, face_count: usize) -> Self {
        Self { values, face_count }
    }

    /// Read a new-format file.
    ///
    /// Values are returned as stored, non-finite ones included: the format
    /// places no constraint on them, and a `NaN` thickness marks a vertex
    /// FreeSurfer could not measure rather than a corrupt file.
    ///
    /// # Errors
    ///
    /// [`FreeSurferError::InvalidMagic`] for anything but the new-format
    /// magic; [`FreeSurferError::InvalidCount`] for an unreasonable vertex or
    /// face count; [`FreeSurferError::Unsupported`] for more than one value per
    /// vertex; [`FreeSurferError::Io`] for a file shorter than its count.
    ///
    /// # Examples
    ///
    /// ```
    /// use ritk_parcellation::freesurfer::Morphometry;
    ///
    /// let thickness = Morphometry::new(vec![2.5, 3.0].into_boxed_slice(), 0);
    /// let mut bytes = Vec::new();
    /// thickness.write(&mut bytes)?;
    /// assert_eq!(Morphometry::read(bytes.as_slice())?, thickness);
    /// # Ok::<(), ritk_parcellation::freesurfer::FreeSurferError>(())
    /// ```
    pub fn read(mut reader: impl Read) -> Result<Self, FreeSurferError> {
        let magic = read_u24(&mut reader)?;
        if magic != MAGIC {
            return Err(FreeSurferError::InvalidMagic {
                format: FORMAT,
                expected: MAGIC,
                got: magic,
            });
        }
        let vertex_count = read_count(&mut reader, FORMAT, "vertex count", MAX_ELEMENTS)?;
        let face_count = read_count(&mut reader, FORMAT, "face count", MAX_ELEMENTS)?;
        let per_vertex = read_be::<i32>(&mut reader)?;
        if per_vertex != 1 {
            return Err(FreeSurferError::Unsupported {
                format: FORMAT,
                field: "values per vertex",
                got: i64::from(per_vertex),
            });
        }
        let mut values = Vec::with_capacity(reserve_for(vertex_count));
        for _ in 0..vertex_count {
            values.push(read_be::<f32>(&mut reader)?);
        }
        Ok(Self {
            values: values.into_boxed_slice(),
            face_count,
        })
    }

    /// Write the new format.
    ///
    /// # Errors
    ///
    /// [`FreeSurferError::Io`] on write failure;
    /// [`FreeSurferError::InvalidCount`] when a count exceeds `i32`.
    pub fn write(&self, mut writer: impl Write) -> Result<(), FreeSurferError> {
        let writer = &mut writer;
        write_u24(writer, MAGIC)?;
        write_count(writer, FORMAT, "vertex count", self.values.len())?;
        write_count(writer, FORMAT, "face count", self.face_count)?;
        write_be(writer, 1_i32)?;
        for value in &self.values {
            write_be(writer, *value)?;
        }
        Ok(())
    }

    /// One value per vertex.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Face count of the surface the values belong to.
    #[must_use]
    pub const fn face_count(&self) -> usize {
        self.face_count
    }
}

#[cfg(test)]
mod tests;
