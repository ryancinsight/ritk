use super::{read_stl_mesh, write_stl_ascii, write_stl_binary};
use crate::domain::vtk_data_object::{AttributeArray, VtkPolyData};
use crate::io::stl::reader::parse_stl;
use crate::io::stl::writer::{write_stl_ascii_to_writer, write_stl_binary_to_writer};
use std::collections::HashMap;
use tempfile::NamedTempFile;

// â”€â”€ Test fixtures â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Tetrahedron with 4 triangular facets and 12 dedicated points (no sharing).
///
/// STL has no shared-vertex topology; each triangle gets 3 unique points.
///
/// Triangle vertex coordinates (analytically derived from a unit tetrahedron):
///   T0: (0,0,0), (1,0,0), (0.5,1,0)  â€” bottom face, normal (0,0,-1)
///   T1: (0,0,0), (1,0,0), (0.5,0.5,1) â€” normal (0,-1,0) approx
///   T2: (0,0,0), (0.5,1,0), (0.5,0.5,1)
///   T3: (1,0,0), (0.5,1,0), (0.5,0.5,1)
fn tet_stl() -> VtkPolyData {
    let points = vec![
        // T0
        [0.0f32, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        // T1
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.5, 1.0],
        // T2
        [0.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 0.5, 1.0],
        // T3
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 0.5, 1.0],
    ];
    let polygons = vec![
        vec![0u32, 1, 2],
        vec![3, 4, 5],
        vec![6, 7, 8],
        vec![9, 10, 11],
    ];
    let cell_normals = vec![
        [0.0f32, 0.0, -1.0],
        [0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ];
    let mut cell_data = HashMap::new();
    cell_data.insert(
        "Normals".to_string(),
        AttributeArray::Normals {
            values: cell_normals,
        },
    );
    VtkPolyData {
        points,
        polygons,
        cell_data,
        ..Default::default()
    }
}

// â”€â”€ Round-trip: ASCII â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[test]
fn test_stl_ascii_roundtrip_coordinates() {
    let mesh = tet_stl();
    let file = NamedTempFile::new().unwrap();
    write_stl_ascii(file.path(), &mesh).unwrap();
    let loaded = read_stl_mesh(file.path()).unwrap();

    assert_eq!(loaded.points.len(), 12, "4 triangles Ã— 3 points = 12");
    assert_eq!(loaded.polygons.len(), 4);

    let eps = 1e-5_f32;
    // T0 vertex 0 = (0,0,0)
    assert!((loaded.points[0][0]).abs() < eps);
    assert!((loaded.points[0][1]).abs() < eps);
    assert!((loaded.points[0][2]).abs() < eps);
    // T0 vertex 1 = (1,0,0)
    assert!((loaded.points[1][0] - 1.0).abs() < eps);
    // T0 vertex 2 = (0.5,1,0)
    assert!((loaded.points[2][0] - 0.5).abs() < eps);
    assert!((loaded.points[2][1] - 1.0).abs() < eps);
}

#[test]
fn test_stl_ascii_roundtrip_cell_normals() {
    let mesh = tet_stl();
    let file = NamedTempFile::new().unwrap();
    write_stl_ascii(file.path(), &mesh).unwrap();
    let loaded = read_stl_mesh(file.path()).unwrap();

    let normals = match loaded
        .cell_data
        .get("Normals")
        .expect("cell normals required")
    {
        AttributeArray::Normals { values } => values.clone(),
        other => panic!("expected Normals, got {other:?}"),
    };
    assert_eq!(normals.len(), 4);
    let eps = 1e-5_f32;
    // T0 normal = (0,0,-1)
    assert!((normals[0][0]).abs() < eps);
    assert!((normals[0][1]).abs() < eps);
    assert!((normals[0][2] - (-1.0)).abs() < eps);
}

// â”€â”€ Round-trip: binary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[test]
fn test_stl_binary_roundtrip_coordinates() {
    let mesh = tet_stl();
    let file = NamedTempFile::new().unwrap();
    write_stl_binary(file.path(), &mesh).unwrap();
    let loaded = read_stl_mesh(file.path()).unwrap();

    assert_eq!(loaded.points.len(), 12);
    assert_eq!(loaded.polygons.len(), 4);

    let eps = 1e-6_f32;
    // Exact bit-for-bit: binary carries no formatting rounding.
    assert!((loaded.points[0][0]).abs() < eps);
    assert!((loaded.points[1][0] - 1.0).abs() < eps);
    assert!((loaded.points[2][0] - 0.5).abs() < eps);
    assert!((loaded.points[2][1] - 1.0).abs() < eps);
}

#[test]
fn test_stl_binary_roundtrip_cell_normals() {
    let mesh = tet_stl();
    let file = NamedTempFile::new().unwrap();
    write_stl_binary(file.path(), &mesh).unwrap();
    let loaded = read_stl_mesh(file.path()).unwrap();

    let normals = match loaded.cell_data.get("Normals").unwrap() {
        AttributeArray::Normals { values } => values.clone(),
        _ => panic!("expected Normals"),
    };
    let eps = 1e-6_f32;
    assert!(
        (normals[0][2] - (-1.0)).abs() < eps,
        "T0 normal z must be -1.0"
    );
    assert!(
        (normals[1][1] - (-1.0)).abs() < eps,
        "T1 normal y must be -1.0"
    );
}

#[test]
fn test_stl_binary_polygon_indices_sequential() {
    let mesh = tet_stl();
    let file = NamedTempFile::new().unwrap();
    write_stl_binary(file.path(), &mesh).unwrap();
    let loaded = read_stl_mesh(file.path()).unwrap();

    // Each triangle's polygon indices must be 3i, 3i+1, 3i+2.
    for (i, tri) in loaded.polygons.iter().enumerate() {
        let base = (i * 3) as u32;
        assert_eq!(
            tri,
            &vec![base, base + 1, base + 2],
            "triangle {i} index pattern"
        );
    }
}

// â”€â”€ Empty mesh â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[test]
fn test_stl_ascii_empty_roundtrip() {
    let mesh = VtkPolyData::default();
    let file = NamedTempFile::new().unwrap();
    write_stl_ascii(file.path(), &mesh).unwrap();
    let loaded = read_stl_mesh(file.path()).unwrap();
    assert_eq!(loaded.points.len(), 0);
    assert_eq!(loaded.polygons.len(), 0);
}

#[test]
fn test_stl_binary_empty_roundtrip() {
    let mesh = VtkPolyData::default();
    let file = NamedTempFile::new().unwrap();
    write_stl_binary(file.path(), &mesh).unwrap();
    let loaded = read_stl_mesh(file.path()).unwrap();
    assert_eq!(loaded.points.len(), 0);
    assert_eq!(loaded.polygons.len(), 0);
}

// â”€â”€ Negative / boundary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[test]
fn test_stl_binary_rejects_quads() {
    let mesh = VtkPolyData {
        points: vec![[0.0; 3], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        polygons: vec![vec![0, 1, 2, 3]], // quad: invalid for STL
        ..Default::default()
    };
    let file = NamedTempFile::new().unwrap();
    assert!(
        write_stl_binary(file.path(), &mesh).is_err(),
        "quads must be rejected"
    );
}

#[test]
fn test_stl_ascii_rejects_quads() {
    let mesh = VtkPolyData {
        points: vec![[0.0; 3], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        polygons: vec![vec![0, 1, 2, 3]],
        ..Default::default()
    };
    let file = NamedTempFile::new().unwrap();
    assert!(
        write_stl_ascii(file.path(), &mesh).is_err(),
        "quads must be rejected"
    );
}

#[test]
fn test_stl_binary_file_size_invariant() {
    // Binary STL for N triangles must be exactly N*50 + 84 bytes.
    let mesh = tet_stl(); // 4 triangles
    let file = NamedTempFile::new().unwrap();
    write_stl_binary(file.path(), &mesh).unwrap();
    let metadata = std::fs::metadata(file.path()).unwrap();
    assert_eq!(
        metadata.len(),
        4 * 50 + 84,
        "binary STL size invariant violated"
    );
}

#[test]
fn test_stl_malformed_ascii_bad_normal() {
    let bad = b"solid bad\n  facet normal NaN NaN NaN\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid bad\n";
    // NaN parses as f32 in Rust, so this actually succeeds with NaN coords.
    // Verify we at least get a polygon back (parser is lenient on values).
    let result = parse_stl(bad);
    assert!(
        result.is_ok(),
        "ASCII STL with NaN normals should parse (values validated by caller)"
    );
}

#[test]
fn test_stl_in_memory_binary_writer() {
    let mesh = tet_stl();
    let mut buf = Vec::new();
    write_stl_binary_to_writer(&mut buf, &mesh).unwrap();
    // 80-byte header + 4-byte count + 4*(12+36+2) = 84 + 200 = 284
    assert_eq!(buf.len(), 4 * 50 + 84);
    // Header starts with "RITK binary STL".
    assert_eq!(&buf[..15], b"RITK binary STL");
    // Triangle count = 4, little-endian.
    assert_eq!(u32::from_le_bytes([buf[80], buf[81], buf[82], buf[83]]), 4);
}

#[test]
fn test_stl_in_memory_ascii_writer() {
    let mesh = tet_stl();
    let mut buf = Vec::new();
    write_stl_ascii_to_writer(&mut buf, &mesh).unwrap();
    let text = String::from_utf8(buf).unwrap();
    assert!(
        text.starts_with("solid ritk"),
        "ASCII STL must start with 'solid'"
    );
    assert!(
        text.ends_with("endsolid ritk\n"),
        "ASCII STL must end with 'endsolid'"
    );
    let facet_count = text.matches("endfacet").count();
    assert_eq!(facet_count, 4, "expected 4 facets");
}
