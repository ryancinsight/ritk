//! Tests for marching_cubes
//! Extracted to keep the 500-line structural limit.
#![allow(clippy::identity_op, clippy::erasing_op)]
use super::mc_tables::EDGE_TABLE;
use super::*;
use gaia::domain::core::index::VertexId;

/// Collect all vertex positions as `Vec<[f64;3]>` via sequential VertexId.
fn collect_vertices(mesh: &crate::surface::Mesh) -> Vec<[f64; 3]> {
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
    assert_eq!(
        mesh.face_count(),
        1,
        "expected exactly 1 triangle for single active corner"
    );
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
    xs.sort_by(|a, b| {
        a.partial_cmp(b)
            .expect("infallible: validated precondition")
    });
    ys.sort_by(|a, b| {
        a.partial_cmp(b)
            .expect("infallible: validated precondition")
    });
    zs.sort_by(|a, b| {
        a.partial_cmp(b)
            .expect("infallible: validated precondition")
    });
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
            assert!(
                comp == 0.0 || (comp - 0.5).abs() < 1e-5,
                "each component is either 0.0 or 0.5, got {comp}"
            );
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
            assert!(
                (d[k] - 2.0 * u[k]).abs() < 1e-5,
                "doubled spacing → doubled position: unit={} double={}",
                u[k],
                d[k]
            );
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
        assert!((s[0] - u[0] - origin[0]).abs() < 1e-4);
        assert!((s[1] - u[1] - origin[1]).abs() < 1e-4);
        assert!((s[2] - u[2] - origin[2]).abs() < 1e-4);
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
            data[iy * nx + ix] = 1.0;
        }
    }
    let mesh = mc_default().extract(&data, [nz, ny, nx]);
    // The (ny-1)*(nx-1) = 9 cubes at z-level 0 each intersect the surface.
    // Each such cube has configuration 0b00001111 = 15 → 2 triangles → 9*2=18 triangles.
    // face_count() is invariant under vertex welding.
    assert_eq!(
        mesh.face_count(),
        18,
        "3×3 grid of cubes at z=0/1 interface should yield 18 triangles"
    );
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
