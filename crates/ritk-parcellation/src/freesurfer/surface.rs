//! FreeSurfer triangular surface geometry (`lh.white`, `lh.pial`, …).
//!
//! A surface annotation labels *vertices*. Turning one into a volumetric
//! parcellation therefore needs the geometry those vertices belong to, which is
//! what this reads.
//!
//! # Format
//!
//! The binary triangle file, per nibabel's `freesurfer/io.py`
//! `read_geometry`/`write_geometry`
//! (<https://github.com/nipy/nibabel/blob/master/nibabel/freesurfer/io.py>), is
//! big-endian throughout:
//!
//! ```text
//! magic          3 bytes, 0xFFFFFE
//! comment        text, terminated by two newlines
//! vertex count   i32
//! face count     i32
//! vertices       f32 × 3 × vertex count
//! faces          i32 × 3 × face count
//! ```
//!
//! The three-byte magic is why the count fields are not simply at a fixed
//! offset, and the comment is free text whose length is not recorded — it ends
//! at the first double newline, which is the only way to find the counts.
//!
//! # Coordinate frame — read this before using the vertices
//!
//! FreeSurfer surfaces are stored in *surface RAS* (also called tkrRAS), not in
//! the scanner frame the volumes carry. The two differ by a translation: surface
//! RAS puts the origin at the centre of the conformed `256³` volume, whereas
//! scanner RAS puts it where the scanner did. The offset is the `c_ras` field of
//! the volume the surface was reconstructed from, and it is typically tens of
//! millimetres — enough to place a cortical ribbon well outside the brain
//! without ever failing.
//!
//! This reader returns the coordinates as stored and does not guess the offset,
//! because the file does not contain it. A caller rasterising into a volume must
//! supply vertices already in that volume's frame; [`Surface::translated`] is
//! there to apply the offset once it is known.

use std::io::{Read, Write};

use super::big_endian::{
    read_be, read_count, read_u24, reserve_for, write_be, write_count, write_u24,
};
use super::{FreeSurferError, FreeSurferFormat};

const FORMAT: FreeSurferFormat = FreeSurferFormat::Surface;

/// Magic identifying a big-endian triangular surface file (16777214).
const TRIANGLE_MAGIC: u32 = 0x00FF_FFFE;

/// Largest vertex or face count that can be a real surface.
///
/// A hemisphere reconstructed at the usual resolution has of order 150,000
/// vertices; ten million is far beyond any real surface and well short of what
/// a corrupt length field would demand, so it separates the two without
/// rejecting anything genuine. The per-vertex formats share the bound.
pub(super) const MAX_ELEMENTS: usize = 10_000_000;

/// A triangular surface mesh.
#[derive(Debug, Clone, PartialEq)]
pub struct Surface {
    /// Vertex coordinates, in whatever frame the file carried.
    vertices: Box<[[f64; 3]]>,
    /// Triangles, as vertex indices.
    faces: Box<[[u32; 3]]>,
}

impl Surface {
    /// Read a FreeSurfer binary triangular surface.
    ///
    /// Coordinates are stored as `f32` and widened to `f64` exactly. Anything
    /// after the faces (the optional volume-geometry trailer) is not read.
    ///
    /// # Errors
    ///
    /// [`FreeSurferError::InvalidMagic`] for a file that is not a triangle
    /// surface (quad surfaces included); [`FreeSurferError::InvalidCount`] for
    /// an unreasonable vertex or face count; [`FreeSurferError::Malformed`] for
    /// a face referencing a vertex that does not exist or a non-finite
    /// coordinate; [`FreeSurferError::Io`] for premature end of file.
    pub fn read(mut reader: impl Read) -> Result<Self, FreeSurferError> {
        let magic = read_u24(&mut reader)?;
        if magic != TRIANGLE_MAGIC {
            return Err(FreeSurferError::InvalidMagic {
                format: FORMAT,
                expected: TRIANGLE_MAGIC,
                got: magic,
            });
        }

        skip_comment(&mut reader)?;

        let vertices_len = read_count(&mut reader, FORMAT, "vertex count", MAX_ELEMENTS)?;
        let faces_len = read_count(&mut reader, FORMAT, "face count", MAX_ELEMENTS)?;

        let mut vertices = Vec::with_capacity(reserve_for(vertices_len));
        for index in 0..vertices_len {
            let point = [
                f64::from(read_be::<f32>(&mut reader)?),
                f64::from(read_be::<f32>(&mut reader)?),
                f64::from(read_be::<f32>(&mut reader)?),
            ];
            if point.iter().any(|value| !value.is_finite()) {
                return Err(FreeSurferError::malformed(
                    FORMAT,
                    "vertex",
                    index,
                    "coordinate is not finite",
                ));
            }
            vertices.push(point);
        }

        let mut faces = Vec::with_capacity(reserve_for(faces_len));
        for index in 0..faces_len {
            let mut triangle = [0_u32; 3];
            for slot in &mut triangle {
                let value = read_be::<i32>(&mut reader)?;
                *slot = u32::try_from(value)
                    .ok()
                    .filter(|vertex| (*vertex as usize) < vertices_len)
                    .ok_or_else(|| {
                        FreeSurferError::malformed(
                            FORMAT,
                            "face",
                            index,
                            format!("references vertex {value} of {vertices_len}"),
                        )
                    })?;
            }
            faces.push(triangle);
        }

        Ok(Self {
            vertices: vertices.into_boxed_slice(),
            faces: faces.into_boxed_slice(),
        })
    }

    /// Write the binary triangle format.
    ///
    /// Coordinates are narrowed to the format's `f32`. `comment` is the
    /// creation line FreeSurfer writes as `created by <user> on <date>`.
    ///
    /// # Errors
    ///
    /// [`FreeSurferError::Malformed`] when `comment` contains a blank line,
    /// which would end the header early; [`FreeSurferError::InvalidCount`]
    /// when a count exceeds `i32`; [`FreeSurferError::Io`] on write failure.
    pub fn write(&self, mut writer: impl Write, comment: &str) -> Result<(), FreeSurferError> {
        if comment.contains("\n\n") || comment.ends_with('\n') {
            return Err(FreeSurferError::malformed(
                FORMAT,
                "comment",
                0,
                "contains a blank line, which terminates the header",
            ));
        }
        let writer = &mut writer;
        write_u24(writer, TRIANGLE_MAGIC)?;
        writer.write_all(comment.as_bytes())?;
        writer.write_all(b"\n\n")?;
        write_count(writer, FORMAT, "vertex count", self.vertices.len())?;
        write_count(writer, FORMAT, "face count", self.faces.len())?;
        for point in &self.vertices {
            for coordinate in point {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "the triangle format stores f32 coordinates"
                )]
                write_be(writer, *coordinate as f32)?;
            }
        }
        for face in &self.faces {
            for vertex in face {
                write_count(writer, FORMAT, "face vertex", *vertex as usize)?;
            }
        }
        Ok(())
    }

    /// Assemble a surface from coordinates and triangles.
    ///
    /// # Errors
    ///
    /// [`FreeSurferError::Malformed`] when a face references a vertex that does
    /// not exist, or a coordinate is not finite.
    pub fn new(vertices: Vec<[f64; 3]>, faces: Vec<[u32; 3]>) -> Result<Self, FreeSurferError> {
        if let Some(index) = vertices
            .iter()
            .position(|point| point.iter().any(|value| !value.is_finite()))
        {
            return Err(FreeSurferError::malformed(
                FORMAT,
                "vertex",
                index,
                "coordinate is not finite",
            ));
        }
        if let Some((index, face)) = faces
            .iter()
            .enumerate()
            .find(|(_, face)| face.iter().any(|v| vertices.len() <= *v as usize))
        {
            return Err(FreeSurferError::malformed(
                FORMAT,
                "face",
                index,
                format!("{face:?} references a vertex of {}", vertices.len()),
            ));
        }
        Ok(Self {
            vertices: vertices.into_boxed_slice(),
            faces: faces.into_boxed_slice(),
        })
    }

    /// Vertex coordinates.
    #[must_use]
    pub const fn vertices(&self) -> &[[f64; 3]] {
        &self.vertices
    }

    /// Triangles, as vertex indices.
    #[must_use]
    pub const fn faces(&self) -> &[[u32; 3]] {
        &self.faces
    }

    /// Number of vertices.
    #[must_use]
    pub const fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// A copy with every vertex shifted by `offset`.
    ///
    /// The way to carry a surface from its stored frame into a volume's: pass
    /// the volume's `c_ras`, and the surface RAS coordinates become scanner RAS.
    /// See the module documentation for why this is not applied automatically.
    #[must_use]
    pub fn translated(&self, offset: [f64; 3]) -> Self {
        Self {
            vertices: self
                .vertices
                .iter()
                .map(|point| {
                    [
                        point[0] + offset[0],
                        point[1] + offset[1],
                        point[2] + offset[2],
                    ]
                })
                .collect(),
            faces: self.faces.clone(),
        }
    }
}

/// Consume the free-text comment, which ends at the first double newline.
fn skip_comment(reader: &mut impl Read) -> Result<(), FreeSurferError> {
    let mut previous = 0_u8;
    loop {
        let mut byte = [0_u8; 1];
        reader.read_exact(&mut byte)?;
        if byte[0] == b'\n' && previous == b'\n' {
            return Ok(());
        }
        previous = byte[0];
    }
}

#[cfg(test)]
mod tests;
