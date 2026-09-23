//! A surface file's coordinate and topology arrays as one mesh
//! (GIFTI 1.0, sections 13.1, 13.5, 13.7).

use crate::{ArrayData, DataArray, GiftiError, GiftiImage, Intent};

/// A triangle mesh read from a GIFTI surface file.
#[derive(Debug, Clone, PartialEq)]
pub struct GiftiSurface {
    /// Vertex coordinates in millimetres, in the space the file's
    /// `CoordinateSystemTransformMatrix` names.
    pub vertices: Vec<[f32; 3]>,
    /// Triangles as vertex indices, counter-clockwise seen from outside
    /// (section 2.3.4.9).
    pub triangles: Vec<[u32; 3]>,
}

impl GiftiImage {
    /// The mesh formed by the first `NIFTI_INTENT_POINTSET` array and the
    /// first `NIFTI_INTENT_TRIANGLE` array.
    ///
    /// # Errors
    ///
    /// [`GiftiError::Structure`] when either array is missing, is not shaped
    /// `[n, 3]`, has the wrong data type (`FLOAT32` coordinates, `INT32`
    /// triangles), or a triangle names a vertex that does not exist.
    pub fn surface(&self) -> Result<GiftiSurface, GiftiError> {
        let points = self.first_of(&Intent::PointSet)?;
        let ArrayData::Float32(coordinates) = triplets(points, "NIFTI_INTENT_POINTSET")? else {
            return Err(GiftiError::structure(
                "DataArray",
                "NIFTI_INTENT_POINTSET must be NIFTI_TYPE_FLOAT32",
            ));
        };
        let topology = self.first_of(&Intent::Triangle)?;
        let ArrayData::Int32(indices) = triplets(topology, "NIFTI_INTENT_TRIANGLE")? else {
            return Err(GiftiError::structure(
                "DataArray",
                "NIFTI_INTENT_TRIANGLE must be NIFTI_TYPE_INT32",
            ));
        };
        let vertices: Vec<[f32; 3]> = coordinates
            .chunks_exact(3)
            .map(|chunk| [chunk[0], chunk[1], chunk[2]])
            .collect();
        let triangles = indices
            .chunks_exact(3)
            .enumerate()
            .map(|(triangle, chunk)| {
                let mut face = [0_u32; 3];
                for (slot, index) in face.iter_mut().zip(chunk) {
                    *slot = u32::try_from(*index)
                        .ok()
                        .filter(|vertex| (*vertex as usize) < vertices.len())
                        .ok_or_else(|| {
                            GiftiError::structure(
                                "DataArray",
                                format!(
                                    "triangle {triangle} names vertex {index} of {}",
                                    vertices.len()
                                ),
                            )
                        })?;
                }
                Ok(face)
            })
            .collect::<Result<Vec<_>, GiftiError>>()?;
        Ok(GiftiSurface {
            vertices,
            triangles,
        })
    }

    fn first_of(&self, intent: &Intent) -> Result<&DataArray, GiftiError> {
        self.arrays
            .iter()
            .find(|array| array.intent() == intent)
            .ok_or_else(|| {
                GiftiError::structure("GIFTI", format!("no {} data array", intent.name()))
            })
    }
}

/// The row-major values of an array shaped `[n, 3]`.
fn triplets(array: &DataArray, intent: &str) -> Result<ArrayData, GiftiError> {
    match array.dims() {
        [_, 3] => Ok(array.row_major()),
        dims => Err(GiftiError::structure(
            "DataArray",
            format!("{intent} has shape {dims:?}, expected [n, 3]"),
        )),
    }
}
