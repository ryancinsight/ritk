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
            [0, 1],  // edge 0
            [1, 2],  // edge 1
            [2, 3],  // edge 2
            [3, 0],  // edge 3
            [4, 5],  // edge 4
            [5, 6],  // edge 5
            [6, 7],  // edge 6
            [7, 4],  // edge 7
            [0, 4],  // edge 8
            [1, 5],  // edge 9
            [2, 6],  // edge 10
            [3, 7],  // edge 11
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
                    for i in 0..8 {
                        if vals[i] > iso {
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

// ─── Lookup Tables (Lorensen & Cline 1987, Bourke 1994 public domain) ────────

/// Edge table: 256-entry bitmask of which of the 12 cube edges are intersected.
///
/// Bit `k` of `EDGE_TABLE[idx]` is 1 iff edge `k` is cut by the isosurface
/// when the cube configuration is `idx`.
static EDGE_TABLE: [u16; 256] = [
    0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000,
];

/// Triangle table: for each of 256 cube configurations, up to 5 triangles
/// encoded as triples of edge indices, −1 terminated (padded to 16 slots).
///
/// `TRI_TABLE[idx][k]` gives the k-th edge index for the k-th triangle vertex;
/// a value of −1 terminates the triangle list.
static TRI_TABLE: [[i8; 16]; 256] = [
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 1, 9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 8, 3, 9, 8, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 3, 1, 2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 2,10, 0, 2, 9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 8, 3, 2,10, 8,10, 9, 8,-1,-1,-1,-1,-1,-1,-1],
    [ 3,11, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0,11, 2, 8,11, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 9, 0, 2, 3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1,11, 2, 1, 9,11, 9, 8,11,-1,-1,-1,-1,-1,-1,-1],
    [ 3,10, 1,11,10, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0,10, 1, 0, 8,10, 8,11,10,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 9, 0, 3,11, 9,11,10, 9,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 8,10,10, 8,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 7, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 3, 0, 7, 3, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 1, 9, 8, 4, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 1, 9, 4, 7, 1, 7, 3, 1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2,10, 8, 4, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 4, 7, 3, 0, 4, 1, 2,10,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 2,10, 9, 0, 2, 8, 4, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 2,10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4,-1,-1,-1,-1],
    [ 8, 4, 7, 3,11, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11, 4, 7,11, 2, 4, 2, 0, 4,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 0, 1, 8, 4, 7, 2, 3,11,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 7,11, 9, 4,11, 9,11, 2, 9, 2, 1,-1,-1,-1,-1],
    [ 3,10, 1, 3,11,10, 7, 8, 4,-1,-1,-1,-1,-1,-1,-1],
    [ 1,11,10, 1, 4,11, 1, 0, 4, 7,11, 4,-1,-1,-1,-1],
    [ 4, 7, 8, 9, 0,11, 9,11,10,11, 0, 3,-1,-1,-1,-1],
    [ 4, 7,11, 4,11, 9, 9,11,10,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 5, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 5, 4, 0, 8, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 5, 4, 1, 5, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 5, 4, 8, 3, 5, 3, 1, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2,10, 9, 5, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 0, 8, 1, 2,10, 4, 9, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 5, 2,10, 5, 4, 2, 4, 0, 2,-1,-1,-1,-1,-1,-1,-1],
    [ 2,10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8,-1,-1,-1,-1],
    [ 9, 5, 4, 2, 3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0,11, 2, 0, 8,11, 4, 9, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 5, 4, 0, 1, 5, 2, 3,11,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 1, 5, 2, 5, 8, 2, 8,11, 4, 8, 5,-1,-1,-1,-1],
    [10, 3,11,10, 1, 3, 9, 5, 4,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 9, 5, 0, 8, 1, 8,10, 1, 8,11,10,-1,-1,-1,-1],
    [ 5, 4, 0, 5, 0,11, 5,11,10,11, 0, 3,-1,-1,-1,-1],
    [ 5, 4, 8, 5, 8,10,10, 8,11,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 7, 8, 5, 7, 9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 3, 0, 9, 5, 3, 5, 7, 3,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 7, 8, 0, 1, 7, 1, 5, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 5, 3, 3, 5, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 7, 8, 9, 5, 7,10, 1, 2,-1,-1,-1,-1,-1,-1,-1],
    [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3,-1,-1,-1,-1],
    [ 8, 0, 2, 8, 2, 5, 8, 5, 7,10, 5, 2,-1,-1,-1,-1],
    [ 2,10, 5, 2, 5, 3, 3, 5, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 7, 9, 5, 7, 8, 9, 3,11, 2,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7,11,-1,-1,-1,-1],
    [ 2, 3,11, 0, 1, 8, 1, 7, 8, 1, 5, 7,-1,-1,-1,-1],
    [11, 2, 1,11, 1, 7, 7, 1, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 5, 8, 8, 5, 7,10, 1, 3,10, 3,11,-1,-1,-1,-1],
    [ 5, 7, 0, 5, 0, 9, 7,11, 0, 1, 0,10,11,10, 0,-1],
    [11,10, 0,11, 0, 3,10, 5, 0, 8, 0, 7, 5, 7, 0,-1],
    [11,10, 5, 7,11, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [10, 6, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 3, 5,10, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 0, 1, 5,10, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 8, 3, 1, 9, 8, 5,10, 6,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 6, 5, 2, 6, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 6, 5, 1, 2, 6, 3, 0, 8,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 6, 5, 9, 0, 6, 0, 2, 6,-1,-1,-1,-1,-1,-1,-1],
    [ 5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8,-1,-1,-1,-1],
    [ 2, 3,11,10, 6, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11, 0, 8,11, 2, 0,10, 6, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 1, 9, 2, 3,11, 5,10, 6,-1,-1,-1,-1,-1,-1,-1],
    [ 5,10, 6, 1, 9, 2, 9,11, 2, 9, 8,11,-1,-1,-1,-1],
    [ 6, 3,11, 6, 5, 3, 5, 1, 3,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8,11, 0,11, 5, 0, 5, 1, 5,11, 6,-1,-1,-1,-1],
    [ 3,11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9,-1,-1,-1,-1],
    [ 6, 5, 9, 6, 9,11,11, 9, 8,-1,-1,-1,-1,-1,-1,-1],
    [ 5,10, 6, 4, 7, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 3, 0, 4, 7, 3, 6, 5,10,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 9, 0, 5,10, 6, 8, 4, 7,-1,-1,-1,-1,-1,-1,-1],
    [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4,-1,-1,-1,-1],
    [ 6, 1, 2, 6, 5, 1, 4, 7, 8,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7,-1,-1,-1,-1],
    [ 8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6,-1,-1,-1,-1],
    [ 7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9,-1],
    [ 3,11, 2, 7, 8, 4,10, 6, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 5,10, 6, 4, 7, 2, 4, 2, 0, 2, 7,11,-1,-1,-1,-1],
    [ 0, 1, 9, 4, 7, 8, 2, 3,11, 5,10, 6,-1,-1,-1,-1],
    [ 9, 2, 1, 9,11, 2, 9, 4,11, 7,11, 4, 5,10, 6,-1],
    [ 8, 4, 7, 3,11, 5, 3, 5, 1, 5,11, 6,-1,-1,-1,-1],
    [ 5, 1,11, 5,11, 6, 1, 0,11, 7,11, 4, 0, 4,11,-1],
    [ 0, 5, 9, 0, 6, 5, 0, 3, 6,11, 6, 3, 8, 4, 7,-1],
    [ 6, 5, 9, 6, 9,11, 4, 7, 9, 7,11, 9,-1,-1,-1,-1],
    [10, 4, 9, 6, 4,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4,10, 6, 4, 9,10, 0, 8, 3,-1,-1,-1,-1,-1,-1,-1],
    [10, 0, 1,10, 6, 0, 6, 4, 0,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1,10,-1,-1,-1,-1],
    [ 1, 4, 9, 1, 2, 4, 2, 6, 4,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4,-1,-1,-1,-1],
    [ 0, 2, 4, 4, 2, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 3, 2, 8, 2, 4, 4, 2, 6,-1,-1,-1,-1,-1,-1,-1],
    [10, 4, 9,10, 6, 4,11, 2, 3,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 2, 2, 8,11, 4, 9,10, 4,10, 6,-1,-1,-1,-1],
    [ 3,11, 2, 0, 1, 6, 0, 6, 4, 6, 1,10,-1,-1,-1,-1],
    [ 6, 4, 1, 6, 1,10, 4, 8, 1, 2, 1,11, 8,11, 1,-1],
    [ 9, 6, 4, 9, 3, 6, 9, 1, 3,11, 6, 3,-1,-1,-1,-1],
    [ 8,11, 1, 8, 1, 0,11, 6, 1, 9, 1, 4, 6, 4, 1,-1],
    [ 3,11, 6, 3, 6, 0, 0, 6, 4,-1,-1,-1,-1,-1,-1,-1],
    [ 6, 4, 8,11, 6, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 7,10, 6, 7, 8,10, 8, 9,10,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 7, 3, 0,10, 7, 0, 9,10, 6, 7,10,-1,-1,-1,-1],
    [10, 6, 7, 1,10, 7, 1, 7, 8, 1, 8, 0,-1,-1,-1,-1],
    [10, 6, 7,10, 7, 1, 1, 7, 3,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7,-1,-1,-1,-1],
    [ 2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9,-1],
    [ 7, 8, 0, 7, 0, 6, 6, 0, 2,-1,-1,-1,-1,-1,-1,-1],
    [ 7, 3, 2, 6, 7, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 3,11,10, 6, 8,10, 8, 9, 8, 6, 7,-1,-1,-1,-1],
    [ 2, 0, 7, 2, 7,11, 0, 9, 7, 6, 7,10, 9,10, 7,-1],
    [ 1, 8, 0, 1, 7, 8, 1,10, 7, 6, 7,10, 2, 3,11,-1],
    [11, 2, 1,11, 1, 7,10, 6, 1, 6, 7, 1,-1,-1,-1,-1],
    [ 8, 9, 6, 8, 6, 7, 9, 1, 6,11, 6, 3, 1, 3, 6,-1],
    [ 0, 9, 1,11, 6, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 7, 8, 0, 7, 0, 6, 3,11, 0,11, 6, 0,-1,-1,-1,-1],
    [ 7,11, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 7, 6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 0, 8,11, 7, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 1, 9,11, 7, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 1, 9, 8, 3, 1,11, 7, 6,-1,-1,-1,-1,-1,-1,-1],
    [10, 1, 2, 6,11, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2,10, 3, 0, 8, 6,11, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 9, 0, 2,10, 9, 6,11, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 6,11, 7, 2,10, 3,10, 8, 3,10, 9, 8,-1,-1,-1,-1],
    [ 7, 2, 3, 6, 2, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 7, 0, 8, 7, 6, 0, 6, 2, 0,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 7, 6, 2, 3, 7, 0, 1, 9,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6,-1,-1,-1,-1],
    [10, 7, 6,10, 1, 7, 1, 3, 7,-1,-1,-1,-1,-1,-1,-1],
    [10, 7, 6, 1, 7,10, 1, 8, 7, 1, 0, 8,-1,-1,-1,-1],
    [ 0, 3, 7, 0, 7,10, 0,10, 9, 6,10, 7,-1,-1,-1,-1],
    [ 7, 6,10, 7,10, 8, 8,10, 9,-1,-1,-1,-1,-1,-1,-1],
    [ 6, 8, 4,11, 8, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 6,11, 3, 0, 6, 0, 4, 6,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 6,11, 8, 4, 6, 9, 0, 1,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 4, 6, 9, 6, 3, 9, 3, 1,11, 3, 6,-1,-1,-1,-1],
    [ 6, 8, 4, 6,11, 8, 2,10, 1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2,10, 3, 0,11, 0, 6,11, 0, 4, 6,-1,-1,-1,-1],
    [ 4,11, 8, 4, 6,11, 0, 2, 9, 2,10, 9,-1,-1,-1,-1],
    [10, 9, 3,10, 3, 2, 9, 4, 3,11, 3, 6, 4, 6, 3,-1],
    [ 8, 2, 3, 8, 4, 2, 4, 6, 2,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 4, 2, 4, 6, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8,-1,-1,-1,-1],
    [ 1, 9, 4, 1, 4, 2, 2, 4, 6,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 1, 3, 8, 6, 1, 8, 4, 6, 6,10, 1,-1,-1,-1,-1],
    [10, 1, 0,10, 0, 6, 6, 0, 4,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 6, 3, 4, 3, 8, 6,10, 3, 0, 3, 9,10, 9, 3,-1],
    [10, 9, 4, 6,10, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 9, 5, 7, 6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 3, 4, 9, 5,11, 7, 6,-1,-1,-1,-1,-1,-1,-1],
    [ 5, 0, 1, 5, 4, 0, 7, 6,11,-1,-1,-1,-1,-1,-1,-1],
    [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5,-1,-1,-1,-1],
    [ 9, 5, 4,10, 1, 2, 7, 6,11,-1,-1,-1,-1,-1,-1,-1],
    [ 6,11, 7, 1, 2,10, 0, 8, 3, 4, 9, 5,-1,-1,-1,-1],
    [ 7, 6,11, 5, 4,10, 4, 2,10, 4, 0, 2,-1,-1,-1,-1],
    [ 3, 4, 8, 3, 5, 4, 3, 2, 5,10, 5, 2,11, 7, 6,-1],
    [ 7, 2, 3, 7, 6, 2, 5, 4, 9,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7,-1,-1,-1,-1],
    [ 3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0,-1,-1,-1,-1],
    [ 6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8,-1],
    [ 9, 5, 4,10, 1, 6, 1, 7, 6, 1, 3, 7,-1,-1,-1,-1],
    [ 1, 6,10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4,-1],
    [ 4, 0,10, 4,10, 5, 0, 3,10, 6,10, 7, 3, 7,10,-1],
    [ 7, 6,10, 7,10, 8, 5, 4,10, 4, 8,10,-1,-1,-1,-1],
    [ 6, 9, 5, 6,11, 9,11, 8, 9,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 6,11, 0, 6, 3, 0, 5, 6, 0, 9, 5,-1,-1,-1,-1],
    [ 0,11, 8, 0, 5,11, 0, 1, 5, 5, 6,11,-1,-1,-1,-1],
    [ 6,11, 3, 6, 3, 5, 5, 3, 1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2,10, 9, 5,11, 9,11, 8,11, 5, 6,-1,-1,-1,-1],
    [ 0,11, 3, 0, 6,11, 0, 9, 6, 5, 6, 9, 1, 2,10,-1],
    [11, 8, 5,11, 5, 6, 8, 0, 5,10, 5, 2, 0, 2, 5,-1],
    [ 6,11, 3, 6, 3, 5, 2,10, 3,10, 5, 3,-1,-1,-1,-1],
    [ 5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2,-1,-1,-1,-1],
    [ 9, 5, 6, 9, 6, 0, 0, 6, 2,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8,-1],
    [ 1, 5, 6, 2, 1, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 3, 6, 1, 6,10, 3, 8, 6, 5, 6, 9, 8, 9, 6,-1],
    [10, 1, 0,10, 0, 6, 9, 5, 0, 5, 6, 0,-1,-1,-1,-1],
    [ 0, 3, 8, 5, 6,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [10, 5, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11, 5,10, 7, 5,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11, 5,10,11, 7, 5, 8, 3, 0,-1,-1,-1,-1,-1,-1,-1],
    [ 5,11, 7, 5,10,11, 1, 9, 0,-1,-1,-1,-1,-1,-1,-1],
    [10, 7, 5,10,11, 7, 9, 8, 1, 8, 3, 1,-1,-1,-1,-1],
    [11, 1, 2,11, 7, 1, 7, 5, 1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2,11,-1,-1,-1,-1],
    [ 9, 7, 5, 9, 2, 7, 9, 0, 2, 2,11, 7,-1,-1,-1,-1],
    [ 7, 5, 2, 7, 2,11, 5, 9, 2, 3, 2, 8, 9, 8, 2,-1],
    [ 2, 5,10, 2, 3, 5, 3, 7, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 2, 0, 8, 5, 2, 8, 7, 5,10, 2, 5,-1,-1,-1,-1],
    [ 9, 0, 1, 5,10, 3, 5, 3, 7, 3,10, 2,-1,-1,-1,-1],
    [ 9, 8, 2, 9, 2, 1, 8, 7, 2,10, 2, 5, 7, 5, 2,-1],
    [ 1, 3, 5, 3, 7, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 7, 0, 7, 1, 1, 7, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 0, 3, 9, 3, 5, 5, 3, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 8, 7, 5, 9, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 5, 8, 4, 5,10, 8,10,11, 8,-1,-1,-1,-1,-1,-1,-1],
    [ 5, 0, 4, 5,11, 0, 5,10,11,11, 3, 0,-1,-1,-1,-1],
    [ 0, 1, 9, 8, 4,10, 8,10,11,10, 4, 5,-1,-1,-1,-1],
    [10,11, 4,10, 4, 5,11, 3, 4, 9, 4, 1, 3, 1, 4,-1],
    [ 2, 5, 1, 2, 8, 5, 2,11, 8, 4, 5, 8,-1,-1,-1,-1],
    [ 0, 4,11, 0,11, 3, 4, 5,11, 2,11, 1, 5, 1,11,-1],
    [ 0, 2, 5, 0, 5, 9, 2,11, 5, 4, 5, 8,11, 8, 5,-1],
    [ 9, 4, 5, 2,11, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 5,10, 3, 5, 2, 3, 4, 5, 3, 8, 4,-1,-1,-1,-1],
    [ 5,10, 2, 5, 2, 4, 4, 2, 0,-1,-1,-1,-1,-1,-1,-1],
    [ 3,10, 2, 3, 5,10, 3, 8, 5, 4, 5, 8, 0, 1, 9,-1],
    [ 5,10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2,-1,-1,-1,-1],
    [ 8, 4, 5, 8, 5, 3, 3, 5, 1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 4, 5, 1, 0, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5,-1,-1,-1,-1],
    [ 9, 4, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4,11, 7, 4, 9,11, 9,10,11,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 3, 4, 9, 7, 9,11, 7, 9,10,11,-1,-1,-1,-1],
    [ 1,10,11, 1,11, 4, 1, 4, 0, 7, 4,11,-1,-1,-1,-1],
    [ 3, 1, 4, 3, 4, 8, 1,10, 4, 7, 4,11,10,11, 4,-1],
    [ 4,11, 7, 9,11, 4, 9, 2,11, 9, 1, 2,-1,-1,-1,-1],
    [ 9, 7, 4, 9,11, 7, 9, 1,11, 2,11, 1, 0, 8, 3,-1],
    [11, 7, 4,11, 4, 2, 2, 4, 0,-1,-1,-1,-1,-1,-1,-1],
    [11, 7, 4,11, 4, 2, 8, 3, 4, 3, 2, 4,-1,-1,-1,-1],
    [ 2, 9,10, 2, 7, 9, 2, 3, 7, 7, 4, 9,-1,-1,-1,-1],
    [ 9,10, 7, 9, 7, 4,10, 2, 7, 8, 7, 0, 2, 0, 7,-1],
    [ 3, 7,10, 3,10, 2, 7, 4,10, 1,10, 0, 4, 0,10,-1],
    [ 1,10, 2, 8, 7, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 9, 1, 4, 1, 7, 7, 1, 3,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1,-1,-1,-1,-1],
    [ 4, 0, 3, 7, 4, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 8, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 9,10, 8,10,11, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 0, 9, 3, 9,11,11, 9,10,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 1,10, 0,10, 8, 8,10,11,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 1,10,11, 3,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2,11, 1,11, 9, 9,11, 8,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 0, 9, 3, 9,11, 1, 2, 9, 2,11, 9,-1,-1,-1,-1],
    [ 0, 2,11, 8, 0,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 2,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 3, 8, 2, 8,10,10, 8, 9,-1,-1,-1,-1,-1,-1,-1],
    [ 9,10, 2, 0, 9, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 3, 8, 2, 8,10, 0, 1, 8, 1,10, 8,-1,-1,-1,-1],
    [ 1,10, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 3, 8, 9, 1, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 9, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 3, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
];

#[cfg(test)]
mod tests {
    use super::*;
    use gaia::domain::core::index::VertexId;

    /// Collect all vertex positions as `Vec<[f64;3]>` via sequential VertexId.
    fn collect_vertices(mesh: &crate::filter::surface::Mesh) -> Vec<[f64; 3]> {
        (0..mesh.vertex_count())
            .map(|i| {
                let p = mesh.vertices.position(VertexId::new(i as u32));
                [p.x, p.y, p.z]
            })
            .collect()
    }

    fn mc_default() -> MarchingCubesFilter {
        MarchingCubesFilter::new()
    }

    // ─── Edge-table spot-checks (Lorensen & Cline invariants) ────────────────

    #[test]
    fn edge_table_fully_outside_is_zero() {
        // All corners below isovalue → index 0 → no edges cut.
        assert_eq!(EDGE_TABLE[0], 0x000);
    }

    #[test]
    fn edge_table_fully_inside_is_zero() {
        // All corners above isovalue → index 255 → no edges cut.
        assert_eq!(EDGE_TABLE[255], 0x000);
    }

    #[test]
    fn edge_table_single_corner_zero_active() {
        // Only corner 0 above isovalue → index 1.
        // Edges 0 (0-1), 3 (3-0), 8 (0-4) must be cut → bits 0,3,8 → 0x109.
        assert_eq!(EDGE_TABLE[1], 0x109);
    }

    #[test]
    fn edge_table_single_corner_seven_active() {
        // Only corner 7 above isovalue → index 128.
        // Edges 6 (6-7), 7 (7-4), 11 (3-7) → bits 6,7,11 → 0x8c0.
        assert_eq!(EDGE_TABLE[128], 0x8c0);
    }

    // ─── Empty volume → no geometry ──────────────────────────────────────────

    #[test]
    fn all_zero_volume_produces_empty_mesh() {
        let data = vec![0.0f32; 3 * 3 * 3];
        let mesh = mc_default().extract(&data, [3, 3, 3]);
        assert_eq!(mesh.face_count(), 0);
        assert_eq!(mesh.vertex_count(), 0);
    }

    #[test]
    fn all_one_volume_produces_empty_mesh() {
        let data = vec![1.0f32; 3 * 3 * 3];
        let mesh = mc_default().extract(&data, [3, 3, 3]);
        assert_eq!(mesh.face_count(), 0);
    }

    #[test]
    fn volume_smaller_than_two_voxels_produces_empty_mesh() {
        let data = vec![1.0f32; 1 * 3 * 3];
        let mesh = mc_default().extract(&data, [1, 3, 3]);
        assert_eq!(mesh.face_count(), 0);
    }

    // ─── Single-corner case: exactly one triangle ─────────────────────────────

    #[test]
    fn single_corner_active_produces_one_triangle() {
        // 2×2×2 volume: only voxel (0,0,0) = 1.0, rest = 0.0.
        let mut data = vec![0.0f32; 8];
        data[0] = 1.0; // index 0 = iz=0, iy=0, ix=0
        let mesh = mc_default().extract(&data, [2, 2, 2]);
        assert_eq!(mesh.face_count(), 1, "expected exactly 1 triangle for single active corner");
        assert_eq!(mesh.vertex_count(), 3);
        // gaia::IndexedMesh structural guarantees replace the old validate() check.
    }

    // ─── Analytical vertex positions ─────────────────────────────────────────

    #[test]
    fn single_corner_analytical_vertex_positions() {
        // 2×2×2, unit spacing, zero origin.
        // Corner 0 = 1.0, all others = 0.0.
        // Cube index = 1 → edges 0, 3, 8 cut.
        // Edge 0 (corners 0-1, vals 1.0-0.0): t=0.5 → physical [0.0, 0.0, 0.5]
        // Edge 3 (corners 3-0, vals 0.0-1.0): t=0.5 → physical [0.0, 0.5, 0.0]
        // Edge 8 (corners 0-4, vals 1.0-0.0): t=0.5 → physical [0.5, 0.0, 0.0]
        //   (z-component is the spacing[2]=1.0 direction for edge 8 (iz varies))
        // Physical [x,y,z]: x=ix*sx, y=iy*sy, z=iz*sz with sx=sy=sz=1.0.
        // Edge 0: ix goes from 0→1, iy=0, iz=0  → [0.0+0.5*1.0, 0.0, 0.0] = [0.5, 0.0, 0.0]
        // Edge 3: corner 3=(iz=0,iy=1,ix=0) → corner 0=(iz=0,iy=0,ix=0)
        //   iy goes from 1→0 → t=0.5 → [0.0, 0.5, 0.0]
        // Edge 8: corner 0=(0,0,0) → corner 4=(1,0,0)
        //   iz goes from 0→1 → t=0.5 → [0.0, 0.0, 0.5]
        let mut data = vec![0.0f32; 8];
        data[0] = 1.0;
        let mesh = mc_default().extract(&data, [2, 2, 2]);
        assert_eq!(mesh.face_count(), 1);
        // Collect all vertex positions via gaia VertexId.
        let verts = collect_vertices(&mesh);
        let mut xs: Vec<f64> = verts.iter().map(|v| v[0]).collect();
        let mut ys: Vec<f64> = verts.iter().map(|v| v[1]).collect();
        let mut zs: Vec<f64> = verts.iter().map(|v| v[2]).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
        zs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // The 3 vertices are permutations of the 3 midpoints:
        // one vertex at x=0.5 (edge 0), one at y=0.5 (edge 3), one at z=0.5 (edge 8).
        let unique_nonzero_x = xs.iter().filter(|&&v| v > 0.01).count();
        let unique_nonzero_y = ys.iter().filter(|&&v| v > 0.01).count();
        let unique_nonzero_z = zs.iter().filter(|&&v| v > 0.01).count();
        assert_eq!(unique_nonzero_x, 1, "exactly one vertex has non-zero x");
        assert_eq!(unique_nonzero_y, 1, "exactly one vertex has non-zero y");
        assert_eq!(unique_nonzero_z, 1, "exactly one vertex has non-zero z");
        // All non-zero components are 0.5 (midpoint interpolation).
        for v in &verts {
            for &comp in v {
                assert!(comp == 0.0 || (comp - 0.5).abs() < 1e-5,
                    "each component is either 0.0 or 0.5, got {comp}");
            }
        }
    }

    // ─── Spacing and origin affect vertex positions ───────────────────────────

    #[test]
    fn spacing_scales_vertex_positions() {
        let mut data = vec![0.0f32; 8];
        data[0] = 1.0;
        let mesh_unit = MarchingCubesFilter::new()
            .with_spacing([1.0, 1.0, 1.0])
            .extract(&data, [2, 2, 2]);
        let mesh_double = MarchingCubesFilter::new()
            .with_spacing([2.0, 2.0, 2.0])
            .extract(&data, [2, 2, 2]);
        assert_eq!(mesh_unit.face_count(), 1);
        assert_eq!(mesh_double.face_count(), 1);
        let verts_unit = collect_vertices(&mesh_unit);
        let verts_double = collect_vertices(&mesh_double);
        // All vertex positions should be exactly doubled.
        for (u, d) in verts_unit.iter().zip(verts_double.iter()) {
            for k in 0..3 {
                assert!((d[k] - 2.0 * u[k]).abs() < 1e-5,
                    "doubled spacing → doubled position: unit={} double={}", u[k], d[k]);
            }
        }
    }

    #[test]
    fn origin_shifts_vertex_positions() {
        let mut data = vec![0.0f32; 8];
        data[0] = 1.0;
        let origin = [10.0, 20.0, 30.0];
        let mesh_orig = MarchingCubesFilter::new()
            .with_origin([0.0; 3])
            .extract(&data, [2, 2, 2]);
        let mesh_shift = MarchingCubesFilter::new()
            .with_origin(origin)
            .extract(&data, [2, 2, 2]);
        let verts_orig = collect_vertices(&mesh_orig);
        let verts_shift = collect_vertices(&mesh_shift);
        for (u, s) in verts_orig.iter().zip(verts_shift.iter()) {
            assert!((s[0] - u[0] - origin[0] as f64).abs() < 1e-4);
            assert!((s[1] - u[1] - origin[1] as f64).abs() < 1e-4);
            assert!((s[2] - u[2] - origin[2] as f64).abs() < 1e-4);
        }
    }

    // ─── Planar interface produces non-zero geometry ──────────────────────────

    #[test]
    fn planar_interface_produces_triangles() {
        // 3×4×4 volume: iz=0 all 1.0, iz=1,2 all 0.0.
        // Cubes at the iz=0/iz=1 boundary (cube z-index 0) will fire.
        let nz = 3usize;
        let ny = 4usize;
        let nx = 4usize;
        let mut data = vec![0.0f32; nz * ny * nx];
        for iy in 0..ny {
            for ix in 0..nx {
                data[0 * ny * nx + iy * nx + ix] = 1.0;
            }
        }
        let mesh = mc_default().extract(&data, [nz, ny, nx]);
        // The (ny-1)*(nx-1) = 9 cubes at z-level 0 each intersect the surface.
        // Each such cube has configuration 0b00001111 = 15 → 2 triangles → 9*2=18 triangles.
        // face_count() is invariant under vertex welding.
        assert_eq!(mesh.face_count(), 18,
            "3×3 grid of cubes at z=0/1 interface should yield 18 triangles");
        // gaia::IndexedMesh structural guarantees replace the old validate() check.
    }

    // ─── All vertex z-coordinates on planar interface ────────────────────────

    #[test]
    fn planar_interface_vertices_at_half_z() {
        // Same setup: iz=0 = 1.0, rest = 0.0.
        // All surface vertices must be at z = 0.5 (midpoint between iz=0 and iz=1).
        let nz = 3usize;
        let ny = 3usize;
        let nx = 3usize;
        let mut data = vec![0.0f32; nz * ny * nx];
        for iy in 0..ny {
            for ix in 0..nx {
                data[iy * nx + ix] = 1.0;
            }
        }
        let mesh = mc_default().extract(&data, [nz, ny, nx]);
        assert!(mesh.face_count() > 0);
        for v in collect_vertices(&mesh) {
            assert!(
                (v[2] - 0.5).abs() < 1e-5,
                "z-coordinate of surface vertex must be 0.5 (midpoint), got {}",
                v[2]
            );
        }
    }
}
