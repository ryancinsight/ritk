//! Triangle mesh geometry type produced by surface-extraction filters.
//!
//! # Invariants
//! - Every index in `triangles[i][j]` is strictly less than `vertices.len()`.
//! - `vertices` coordinates are in physical millimetre space [x, y, z].

/// Triangle mesh: oriented triangle soup in physical mm space.
///
/// Coordinates follow the standard medical-imaging convention:
/// `[x, y, z]` where x = column axis, y = row axis, z = slice axis,
/// each scaled by the corresponding voxel spacing.
///
/// Marching cubes produces an oriented surface: outward normals face
/// toward the higher-intensity (foreground) side of the isovalue.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Mesh {
    /// Vertex positions in physical mm space, `[x, y, z]`.
    pub vertices: Vec<[f32; 3]>,
    /// Triangle face indices: each `[u32; 3]` is one counter-clockwise triangle.
    pub triangles: Vec<[u32; 3]>,
}

impl Mesh {
    /// Construct an empty mesh.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of triangles.
    #[inline]
    pub fn n_triangles(&self) -> usize {
        self.triangles.len()
    }

    /// Number of vertices (with possible duplicates for unwelded meshes).
    #[inline]
    pub fn n_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Validate the index-bound invariant.
    ///
    /// Returns `Err` with a description of the first violated invariant.
    pub fn validate(&self) -> Result<(), String> {
        let nv = self.vertices.len() as u32;
        for (i, tri) in self.triangles.iter().enumerate() {
            for &idx in tri {
                if idx >= nv {
                    return Err(format!(
                        "triangle {i}: index {idx} out of range (n_vertices = {nv})"
                    ));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_mesh_is_valid() {
        assert_eq!(Mesh::new().validate(), Ok(()));
    }

    #[test]
    fn mesh_with_valid_triangle_passes() {
        let m = Mesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            triangles: vec![[0, 1, 2]],
        };
        assert_eq!(m.validate(), Ok(()));
        assert_eq!(m.n_triangles(), 1);
        assert_eq!(m.n_vertices(), 3);
    }

    #[test]
    fn mesh_with_oob_index_fails_validation() {
        let m = Mesh {
            vertices: vec![[0.0, 0.0, 0.0]],
            triangles: vec![[0, 1, 0]],
        };
        assert!(m.validate().is_err());
    }
}
