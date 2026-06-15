use super::*;
use gaia::MeshBuilder;
use nalgebra::Point3;
use tempfile::NamedTempFile;

/// A single equilateral-ish triangle for smoke tests.
fn single_triangle() -> IndexedMesh {
    let mut b = MeshBuilder::new();
    b.add_triangle_soup(&[(
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.5, 1.0, 0.0),
    )]);
    b.build()
}

// ── STL round-trip ───────────────────────────────────────────────────────

#[test]
fn stl_binary_roundtrip_vertex_and_face_counts() {
    let mesh = single_triangle();
    let f = NamedTempFile::new().unwrap();
    write_indexed_stl_binary(f.path(), &mesh).unwrap();
    let loaded = read_stl_indexed(f.path()).unwrap();
    assert_eq!(loaded.face_count(), 1, "one triangle round-tripped");
    // Vertex count: exactly 3 unique welded vertices.
    assert_eq!(
        loaded.vertex_count(),
        3,
        "three distinct vertices after welding"
    );
}

#[test]
fn stl_ascii_roundtrip_vertex_and_face_counts() {
    let mesh = single_triangle();
    let f = NamedTempFile::new().unwrap();
    write_indexed_stl_ascii(f.path(), "test", &mesh).unwrap();
    let loaded = read_stl_indexed(f.path()).unwrap();
    assert_eq!(loaded.face_count(), 1);
    assert_eq!(loaded.vertex_count(), 3);
}

#[test]
fn stl_binary_roundtrip_coords() {
    let mesh = single_triangle();
    let f = NamedTempFile::new().unwrap();
    write_indexed_stl_binary(f.path(), &mesh).unwrap();
    let loaded = read_stl_indexed(f.path()).unwrap();
    use gaia::domain::core::index::VertexId;
    // Collect all vertex positions.
    let pts: Vec<_> = (0..loaded.vertex_count())
        .map(|i| loaded.vertices.position(VertexId::new(i as u32)))
        .collect();
    let eps = 1e-4_f64; // binary STL stores f32, so tolerance ≥ f32 epsilon
    let has_origin = pts
        .iter()
        .any(|p| p.x.abs() < eps && p.y.abs() < eps && p.z.abs() < eps);
    assert!(
        has_origin,
        "origin vertex must survive round-trip; pts={pts:?}"
    );
}

// ── OBJ round-trip ───────────────────────────────────────────────────────

#[test]
fn obj_roundtrip_vertex_and_face_counts() {
    let mesh = single_triangle();
    let f = NamedTempFile::new().unwrap();
    write_indexed_obj(f.path(), &mesh).unwrap();
    let loaded = read_obj_indexed(f.path()).unwrap();
    assert_eq!(loaded.face_count(), 1);
    assert_eq!(loaded.vertex_count(), 3);
}

// ── PLY round-trip ───────────────────────────────────────────────────────

#[test]
fn ply_roundtrip_vertex_and_face_counts() {
    let mesh = single_triangle();
    let f = NamedTempFile::new().unwrap();
    write_indexed_ply(f.path(), &mesh).unwrap();
    let loaded = read_ply_indexed(f.path()).unwrap();
    assert_eq!(loaded.face_count(), 1);
    assert_eq!(loaded.vertex_count(), 3);
}

// ── GLB write ────────────────────────────────────────────────────────────

#[test]
fn glb_write_produces_valid_glb_header() {
    let mesh = single_triangle();
    let f = NamedTempFile::new().unwrap();
    write_indexed_glb(f.path(), &mesh).unwrap();
    let bytes = std::fs::read(f.path()).unwrap();
    // GLB magic = 0x46546C67 ("glTF" in little-endian).
    assert!(
        bytes.len() >= 4,
        "GLB must be non-empty; got {} bytes",
        bytes.len()
    );
    assert_eq!(&bytes[..4], b"glTF", "GLB magic bytes must be 'glTF'");
}
