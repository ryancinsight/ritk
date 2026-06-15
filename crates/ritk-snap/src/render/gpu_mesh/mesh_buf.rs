//! Mesh buffer construction: fan-triangulation, vertex-normal smoothing, GPU upload.
//!
//! # Algorithm
//!
//! 1. **Fan-triangulate** all polygons and triangle strips from `VtkPolyData`.
//! 2. **Compute per-vertex normals** by accumulating face normals for all
//!    incident triangles and normalizing. If `point_data` contains a
//!    `AttributeArray::Normals` entry named `"Normals"`, those are used directly
//!    instead of computing from geometry.
//! 3. **Build interleaved vertex array** as `Vec<MeshVertex>` (position + normal).
//! 4. **Upload** vertex and index buffers to the GPU.
//!
//! # Invariants
//!
//! - Every returned index is strictly less than the vertex count.
//! - `GpuMeshBufs::n_indices` equals the total number of triangle vertices
//!   (always a multiple of 3).
//! - Face normals for degenerate triangles (zero area) default to `[0, 0, 1]`.

use super::params::MeshVertex;
use ritk_io::VtkPolyData;
use wgpu::util::DeviceExt as _;

/// GPU vertex and index buffers for one mesh, plus a change-detection key.
pub(super) struct GpuMeshBufs {
    pub vertex_buf: wgpu::Buffer,
    pub index_buf: wgpu::Buffer,
    pub n_indices: u32,
    /// Raw pointer value of `mesh.points` data used for change detection.
    /// If this changes, the buffers must be rebuilt.
    pub points_ptr: usize,
}

impl GpuMeshBufs {
    /// Build GPU buffers from a `VtkPolyData` mesh.
    ///
    /// Returns `None` if the mesh has no renderable geometry (no triangles after
    /// fan-triangulation).
    pub(super) fn build(device: &wgpu::Device, mesh: &VtkPolyData) -> Option<Self> {
        let (vertices, indices) = build_mesh_cpu(mesh);
        if indices.is_empty() {
            return None;
        }

        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_vertex_buf"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_index_buf"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let points_ptr = mesh.points.as_ptr() as usize;

        Some(Self {
            vertex_buf,
            index_buf,
            n_indices: indices.len() as u32,
            points_ptr,
        })
    }
}

// ── CPU mesh construction ─────────────────────────────────────────────────────

/// Fan-triangulate all polygons and triangle strips, compute vertex normals,
/// and return `(Vec<MeshVertex>, Vec<u32>)`.
fn build_mesh_cpu(mesh: &VtkPolyData) -> (Vec<MeshVertex>, Vec<u32>) {
    let n_pts = mesh.points.len();
    if n_pts == 0 {
        return (Vec::new(), Vec::new());
    }

    // Collect triangles as (i0, i1, i2) index triples.
    let mut tris: Vec<[u32; 3]> = Vec::new();

    // Fan-triangulate polygons.
    for poly in &mesh.polygons {
        if poly.len() < 3 {
            continue;
        }
        let v0 = poly[0] as usize;
        for i in 1..(poly.len() - 1) {
            let v1 = poly[i] as usize;
            let v2 = poly[i + 1] as usize;
            if v0 < n_pts && v1 < n_pts && v2 < n_pts {
                tris.push([poly[0], poly[i], poly[i + 1]]);
            }
        }
    }

    // Triangulate triangle strips (alternating winding for odd triangles).
    for strip in &mesh.triangle_strips {
        let n = strip.len();
        if n < 3 {
            continue;
        }
        for i in 0..(n - 2) {
            let (a, b, c) = if i % 2 == 0 {
                (strip[i], strip[i + 1], strip[i + 2])
            } else {
                // Odd: swap first two vertices to maintain CCW winding.
                (strip[i + 1], strip[i], strip[i + 2])
            };
            let (a, b, c) = (a as usize, b as usize, c as usize);
            if a < n_pts && b < n_pts && c < n_pts {
                tris.push([
                    strip[i],
                    if i % 2 == 0 { strip[i + 1] } else { strip[i] },
                    if i % 2 == 0 {
                        strip[i + 2]
                    } else {
                        strip[i + 1]
                    },
                ]);
                let _ = c; // suppress unused warning; validation done above
                let _ = b;
                let _ = a;
            }
        }
    }

    if tris.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // Decide vertex normal source.
    let precomputed_normals = get_precomputed_normals(mesh);

    // Accumulate face normals into per-vertex sums (if not precomputed).
    let mut normal_acc = vec![[0.0f32; 3]; n_pts];
    if precomputed_normals.is_none() {
        for tri in &tris {
            let p0 = mesh.points[tri[0] as usize];
            let p1 = mesh.points[tri[1] as usize];
            let p2 = mesh.points[tri[2] as usize];
            let fn_ = face_normal(p0, p1, p2);
            for &idx in tri {
                let a = &mut normal_acc[idx as usize];
                a[0] += fn_[0];
                a[1] += fn_[1];
                a[2] += fn_[2];
            }
        }
    }

    // Build MeshVertex array.
    let mut vertices: Vec<MeshVertex> = mesh
        .points
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            let n = match &precomputed_normals {
                Some(normals) => normals[i],
                None => normalize(normal_acc[i]),
            };
            MeshVertex {
                position: p,
                _pad0: 1.0, // W = 1.0 for position (homogeneous point)
                normal: n,
                _pad1: 0.0, // W = 0.0 for normal (direction vector)
            }
        })
        .collect();

    // Guard: if no normals were accumulated for a vertex (isolated vertex not
    // part of any triangle), assign a default up normal [0,0,1].
    if precomputed_normals.is_none() {
        for (i, acc) in normal_acc.iter().enumerate() {
            let len2 = acc[0] * acc[0] + acc[1] * acc[1] + acc[2] * acc[2];
            if len2 < 1e-24 {
                vertices[i].normal = [0.0, 0.0, 1.0];
            }
        }
    }

    // Flatten triangle index triples to a linear index buffer.
    let indices: Vec<u32> = tris.iter().flat_map(|&t| t).collect();

    (vertices, indices)
}

/// Extract precomputed vertex normals named `"Normals"` from `point_data`, if present.
fn get_precomputed_normals(mesh: &VtkPolyData) -> Option<Vec<[f32; 3]>> {
    use ritk_io::AttributeArray;
    match mesh.point_data.get("Normals")? {
        AttributeArray::Normals { values } if values.len() == mesh.points.len() => {
            Some(values.clone())
        }
        _ => None,
    }
}

// ── Vector math helpers ───────────────────────────────────────────────────────

fn face_normal(p0: [f32; 3], p1: [f32; 3], p2: [f32; 3]) -> [f32; 3] {
    let e0 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let e1 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    let n = [
        e0[1] * e1[2] - e0[2] * e1[1],
        e0[2] * e1[0] - e0[0] * e1[2],
        e0[0] * e1[1] - e0[1] * e1[0],
    ];
    normalize(n)
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-12 {
        return [0.0, 0.0, 1.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Single triangle: yields exactly 3 vertices and 3 indices.
    #[test]
    fn single_triangle_yields_three_indices() {
        let mut mesh = VtkPolyData::default();
        mesh.points = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        mesh.polygons = vec![vec![0, 1, 2]];
        let (verts, idxs) = build_mesh_cpu(&mesh);
        assert_eq!(verts.len(), 3, "vertex count");
        assert_eq!(idxs.len(), 3, "index count");
    }

    /// Face normal for XY-plane triangle is [0,0,1].
    #[test]
    fn face_normal_xy_plane() {
        let n = face_normal([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert!((n[0]).abs() < 1e-6);
        assert!((n[1]).abs() < 1e-6);
        assert!((n[2] - 1.0).abs() < 1e-6, "z = {}", n[2]);
    }

    /// Empty mesh returns empty buffers (GpuMeshBufs::build returns None).
    #[test]
    fn empty_mesh_returns_none() {
        let mesh = VtkPolyData::default();
        let (verts, idxs) = build_mesh_cpu(&mesh);
        assert!(verts.is_empty());
        assert!(idxs.is_empty());
    }

    /// Quad (4-vertex polygon) fan-triangulates to 6 indices (2 triangles).
    #[test]
    fn quad_fan_triangulates_to_two_triangles() {
        let mut mesh = VtkPolyData::default();
        mesh.points = vec![
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ];
        mesh.polygons = vec![vec![0, 1, 2, 3]];
        let (_verts, idxs) = build_mesh_cpu(&mesh);
        assert_eq!(
            idxs.len(),
            6,
            "2 triangles × 3 indices = 6, got {}",
            idxs.len()
        );
    }
}
