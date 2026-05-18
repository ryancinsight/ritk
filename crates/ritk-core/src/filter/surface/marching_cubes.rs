//! Marching Cubes isosurface extraction.
//!
//! # Algorithm
//! Lorensen, W.E. & Cline, H.E. (1987). Marching Cubes: A High Resolution 3D Surface
//! Construction Algorithm. SIGGRAPH '87 Proceedings, pp. 163–169.
//!
//! For each axis-aligned voxel cube, the 256-entry edge table maps the binary
//! above/below-isovalue pattern of the 8 corners to the 12-bit mask of cut edges.
//! The triangle table maps the same index to up to 5 triangles (15 edge-index slots,
//! −1 terminated). Vertex positions are computed by linear interpolation along each
//! cut edge between the two corner physical positions.
//!
//! # Coordinate convention
//! Vertices are emitted in physical mm space `[x, y, z]` where:
//! - `x = origin[0] + ix * spacing[0]`
//! - `y = origin[1] + iy * spacing[1]`
//! - `z = origin[2] + iz * spacing[2]`
//!
//! and `(ix, iy, iz)` is the interpolated fractional voxel position on the cut edge.
//!
//! # Mesh orientation
//! Triangle winding follows the Lorensen convention: outward normals face away from
//! the interior (above-isovalue) region.
//!
//! # Output
//! Triangle vertices are fed into [`gaia::MeshBuilder`] which performs spatial-hash
//! vertex welding (1e-4 mm tolerance) and returns a deduplicated [`gaia::IndexedMesh`].
//! Emission is streamed directly into the builder (no temporary global soup buffer),
//! reducing peak memory to O(1) additional storage per active cube.

use nalgebra::Point3;

use super::mesh::{Mesh, MeshBuilder};

/// Marching Cubes isosurface extractor.
///
/// # Invariants
/// - `isovalue` must be finite.
/// - `spacing` components must all be strictly positive.
#[derive(Debug, Clone)]
pub struct MarchingCubesFilter {
    /// Isosurface threshold value.
    ///
    /// Voxels with value > isovalue are considered "inside" (above the surface).
    pub isovalue: f32,
    /// Physical origin `[ox, oy, oz]` of the first voxel in mm.
    pub origin: [f64; 3],
    /// Voxel spacing `[sx, sy, sz]` in mm.
    pub spacing: [f64; 3],
}

impl Default for MarchingCubesFilter {
    fn default() -> Self {
        Self {
            isovalue: 0.5,
            origin: [0.0; 3],
            spacing: [1.0; 3],
        }
    }
}

impl MarchingCubesFilter {
    /// Construct with `isovalue = 0.5`, unit spacing, zero origin.
    ///
    /// This default is appropriate for binary label maps where foreground = 1 and
    /// background = 0.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the isovalue. Must be finite.
    pub fn with_isovalue(mut self, isovalue: f32) -> Self {
        self.isovalue = isovalue;
        self
    }

    /// Set the physical origin of voxel (0, 0, 0) in mm.
    pub fn with_origin(mut self, origin: [f64; 3]) -> Self {
        self.origin = origin;
        self
    }

    /// Set voxel spacing in mm. All components must be > 0.
    ///
    /// # Panics
    /// Panics in debug mode if any spacing component is non-positive.
    pub fn with_spacing(mut self, spacing: [f64; 3]) -> Self {
        debug_assert!(
            spacing.iter().all(|&s| s > 0.0),
            "all spacing components must be strictly positive"
        );
        self.spacing = spacing;
        self
    }

    /// Extract the isosurface from a 3D scalar volume.
    ///
    /// # Parameters
    /// - `data`: flat row-major buffer of length `shape[0]*shape[1]*shape[2]`
    ///   where index `(iz, iy, ix)` maps to `iz*shape[1]*shape[2] + iy*shape[2] + ix`.
    /// - `shape`: `[nz, ny, nx]`.
    ///
    /// # Returns
    /// A welded, deduplicated [`Mesh`] (`gaia::IndexedMesh<f64>`) in physical mm
    /// coordinates. Returns an empty mesh if `shape` is degenerate (any dimension < 2)
    /// or if no cube contains a surface crossing.
    ///
    /// Triangle vertices from marching cubes are streamed directly into
    /// [`MeshBuilder`] via per-triangle `vertex()`/`triangle()` insertion.
    /// `MeshBuilder` performs spatial-hash vertex welding (1e-4 mm tolerance),
    /// yielding a shared-vertex representation suitable for watertight checking and CSG.
    pub fn extract(&self, data: &[f32], shape: [usize; 3]) -> Mesh {
        let [nz, ny, nx] = shape;
        if nz < 2 || ny < 2 || nx < 2 {
            return MeshBuilder::new().build();
        }
        debug_assert_eq!(
            data.len(),
            nz * ny * nx,
            "data length must equal shape[0]*shape[1]*shape[2]"
        );

        let stride_z = ny * nx;
        let stride_y = nx;
        let iso = self.isovalue;

        // Vertex offset for each of the 8 cube corners: (dz, dy, dx)
        const CORNERS: [[usize; 3]; 8] = [
            [0, 0, 0], // 0
            [0, 0, 1], // 1
            [0, 1, 1], // 2
            [0, 1, 0], // 3
            [1, 0, 0], // 4
            [1, 0, 1], // 5
            [1, 1, 1], // 6
            [1, 1, 0], // 7
        ];

        // Edge pairs: (corner_a, corner_b) for each of the 12 cube edges.
        const EDGES: [[usize; 2]; 12] = [
            [0, 1], // edge 0
            [1, 2], // edge 1
            [2, 3], // edge 2
            [3, 0], // edge 3
            [4, 5], // edge 4
            [5, 6], // edge 5
            [6, 7], // edge 6
            [7, 4], // edge 7
            [0, 4], // edge 8
            [1, 5], // edge 9
            [2, 6], // edge 10
            [3, 7], // edge 11
        ];

        // Stream triangle emission directly into gaia MeshBuilder.
        let mut builder = MeshBuilder::new();

        // Physical position of a voxel corner (iz, iy, ix) in mm.
        let phys = |iz: usize, iy: usize, ix: usize| -> Point3<f64> {
            Point3::new(
                self.origin[0] + ix as f64 * self.spacing[0],
                self.origin[1] + iy as f64 * self.spacing[1],
                self.origin[2] + iz as f64 * self.spacing[2],
            )
        };

        // Linear interpolation between physical positions pa and pb
        // at the isovalue crossing between scalar values va and vb.
        let interp = |pa: Point3<f64>, va: f32, pb: Point3<f64>, vb: f32| -> Point3<f64> {
            let dv = (vb - va) as f64;
            if dv.abs() < f64::EPSILON {
                return pa;
            }
            let t = (iso as f64 - va as f64) / dv;
            pa + (pb - pa) * t
        };

        for iz in 0..nz - 1 {
            for iy in 0..ny - 1 {
                for ix in 0..nx - 1 {
                    // Sample the 8 corner values of this cube.
                    let mut vals = [0.0f32; 8];
                    for (ci, &[dz, dy, dx]) in CORNERS.iter().enumerate() {
                        let flat = (iz + dz) * stride_z + (iy + dy) * stride_y + (ix + dx);
                        vals[ci] = data[flat];
                    }

                    // Build the 8-bit cube index: bit i = 1 iff corner i > isovalue.
                    let mut cube_idx: usize = 0;
                    for (i, &v) in vals.iter().enumerate() {
                        if v > iso {
                            cube_idx |= 1 << i;
                        }
                    }

                    let edge_mask = EDGE_TABLE[cube_idx];
                    if edge_mask == 0 {
                        continue; // Fully inside or fully outside.
                    }

                    // Compute the 3D physical positions of the 8 corners.
                    let mut cpos = [Point3::<f64>::origin(); 8];
                    for (ci, &[dz, dy, dx]) in CORNERS.iter().enumerate() {
                        cpos[ci] = phys(iz + dz, iy + dy, ix + dx);
                    }

                    // Compute the at most 12 edge-crossing vertex positions.
                    let mut edge_verts = [Point3::<f64>::origin(); 12];
                    for edge in 0..12usize {
                        if edge_mask & (1 << edge) != 0 {
                            let [a, b] = EDGES[edge];
                            edge_verts[edge] = interp(cpos[a], vals[a], cpos[b], vals[b]);
                        }
                    }

                    // Emit triangles from the triangle table.
                    let tris = &TRI_TABLE[cube_idx];
                    let mut ti = 0;
                    while ti < 15 && tris[ti] >= 0 {
                        let i0 = tris[ti] as usize;
                        let i1 = tris[ti + 1] as usize;
                        let i2 = tris[ti + 2] as usize;
                        let v0 = builder.vertex(edge_verts[i0]);
                        let v1 = builder.vertex(edge_verts[i1]);
                        let v2 = builder.vertex(edge_verts[i2]);
                        builder.triangle(v0, v1, v2);
                        ti += 3;
                    }
                }
            }
        }

        // Build welded, deduplicated IndexedMesh via gaia's MeshBuilder.
        builder.build()
    }
}

// ─── Lookup Tables ────────────────────────────────────────────────────────────
#[allow(clippy::all)]
#[path = "mc_tables.rs"]
mod mc_tables;
use mc_tables::{EDGE_TABLE, TRI_TABLE};

#[cfg(test)]
#[path = "tests_marching_cubes.rs"]
mod tests_marching_cubes;
