//! FreeSurfer triangular surface geometry (`lh.white`, `lh.pial`, …).
//!
//! A surface annotation labels *vertices*. Turning one into a volumetric
//! parcellation therefore needs the geometry those vertices belong to, which is
//! what this reads.
//!
//! # Format
//!
//! The binary triangle file is big-endian throughout:
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

use std::io::Read;

use super::FreeSurferSurfaceError;

/// Magic identifying a big-endian triangular surface file.
const TRIANGLE_MAGIC: u32 = 0x00FF_FFFE;

/// Largest vertex or face count that can be a real surface.
///
/// A hemisphere reconstructed at the usual resolution has of order 150,000
/// vertices; ten million is far beyond any real surface and well short of what
/// a corrupt length field would demand, so it separates the two without
/// rejecting anything genuine.
const MAX_ELEMENTS: i32 = 10_000_000;

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
    /// # Errors
    ///
    /// [`FreeSurferSurfaceError`] for wrong magic, an unreasonable vertex or
    /// face count, a face referencing a vertex that does not exist, a
    /// non-finite coordinate, or premature end of file.
    pub fn read(mut reader: impl Read) -> Result<Self, FreeSurferSurfaceError> {
        let mut magic = [0_u8; 3];
        reader.read_exact(&mut magic)?;
        let magic = u32::from(magic[0]) << 16 | u32::from(magic[1]) << 8 | u32::from(magic[2]);
        if magic != TRIANGLE_MAGIC {
            #[expect(
                clippy::cast_possible_wrap,
                reason = "reported for diagnosis; the three-byte magic cannot reach i32::MAX"
            )]
            let got = magic as i32;
            return Err(FreeSurferSurfaceError::InvalidMagic { got });
        }

        skip_comment(&mut reader)?;

        let vertex_count = read_i32(&mut reader)?;
        let face_count = read_i32(&mut reader)?;
        for count in [vertex_count, face_count] {
            if !(0..=MAX_ELEMENTS).contains(&count) {
                return Err(FreeSurferSurfaceError::InvalidVertexCount { count });
            }
        }
        #[expect(
            clippy::cast_sign_loss,
            reason = "both counts are range-checked nonnegative immediately above"
        )]
        let (vertices_len, faces_len) = (vertex_count as usize, face_count as usize);

        let mut vertices = Vec::with_capacity(vertices_len);
        for index in 0..vertices_len {
            let point = [
                f64::from(read_f32(&mut reader)?),
                f64::from(read_f32(&mut reader)?),
                f64::from(read_f32(&mut reader)?),
            ];
            if point.iter().any(|value| !value.is_finite()) {
                return Err(FreeSurferSurfaceError::MalformedLabelTable {
                    index,
                    reason: "vertex coordinate is not finite".to_owned(),
                });
            }
            vertices.push(point);
        }

        let mut faces = Vec::with_capacity(faces_len);
        for index in 0..faces_len {
            let mut triangle = [0_u32; 3];
            for slot in &mut triangle {
                let value = read_i32(&mut reader)?;
                #[expect(
                    clippy::cast_sign_loss,
                    reason = "checked against the vertex count immediately below"
                )]
                let vertex = value as u32;
                if value < 0 || vertices_len <= vertex as usize {
                    return Err(FreeSurferSurfaceError::MalformedLabelTable {
                        index,
                        reason: format!("face references vertex {value} of {vertices_len}"),
                    });
                }
                *slot = vertex;
            }
            faces.push(triangle);
        }

        Ok(Self {
            vertices: vertices.into_boxed_slice(),
            faces: faces.into_boxed_slice(),
        })
    }

    /// Assemble a surface from coordinates and triangles.
    ///
    /// # Errors
    ///
    /// [`FreeSurferSurfaceError::MalformedLabelTable`] when a face references a
    /// vertex that does not exist, or a coordinate is not finite.
    pub fn new(
        vertices: Vec<[f64; 3]>,
        faces: Vec<[u32; 3]>,
    ) -> Result<Self, FreeSurferSurfaceError> {
        if let Some((index, _)) = vertices
            .iter()
            .enumerate()
            .find(|(_, point)| point.iter().any(|value| !value.is_finite()))
        {
            return Err(FreeSurferSurfaceError::MalformedLabelTable {
                index,
                reason: "vertex coordinate is not finite".to_owned(),
            });
        }
        if let Some((index, face)) = faces
            .iter()
            .enumerate()
            .find(|(_, face)| face.iter().any(|v| vertices.len() <= *v as usize))
        {
            return Err(FreeSurferSurfaceError::MalformedLabelTable {
                index,
                reason: format!("face {face:?} references a vertex of {}", vertices.len()),
            });
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
fn skip_comment(reader: &mut impl Read) -> Result<(), FreeSurferSurfaceError> {
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

fn read_i32(reader: &mut impl Read) -> Result<i32, FreeSurferSurfaceError> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(i32::from_be_bytes(bytes))
}

fn read_f32(reader: &mut impl Read) -> Result<f32, FreeSurferSurfaceError> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(f32::from_be_bytes(bytes))
}

#[cfg(test)]
mod tests;
