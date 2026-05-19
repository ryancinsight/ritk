use super::{read_ply_mesh, write_ply_ascii, write_ply_binary_le};
use crate::domain::vtk_data_object::{AttributeArray, VtkPolyData};
use crate::io::ply::reader::parse_ply;
use crate::io::ply::writer::{write_ply_ascii_to_writer, write_ply_binary_le_to_writer};
use tempfile::NamedTempFile;

// ── Test fixtures ─────────────────────────────────────────────────────────────

/// Regular tetrahedron with shared vertex topology.
///
/// V0=(0,0,0) V1=(1,0,0) V2=(0.5,1,0) V3=(0.5,0.5,1)
/// 4 triangular faces.
fn tetrahedron() -> VtkPolyData {
    VtkPolyData {
        points: vec![
            [0.0f32, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ],
        polygons: vec![
            vec![0u32, 1, 2],
            vec![0, 1, 3],
            vec![0, 2, 3],
            vec![1, 2, 3],
        ],
        ..Default::default()
    }
}

/// Tetrahedron augmented with per-vertex normals (unit axis directions).
fn tetrahedron_with_normals() -> VtkPolyData {
    let mut mesh = tetrahedron();
    mesh.point_data.insert(
        "Normals".to_string(),
        AttributeArray::Normals {
            values: vec![
                [1.0f32, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.577_350_3, 0.577_350_3, 0.577_350_3], // unit (1,1,1)/√3
            ],
        },
    );
    mesh
}

// ── Round-trip: ASCII ─────────────────────────────────────────────────────────

#[test]
fn test_ply_ascii_roundtrip_coordinates() {
    let mesh = tetrahedron();
    let file = NamedTempFile::new().unwrap();
    write_ply_ascii(file.path(), &mesh).unwrap();
    let loaded = read_ply_mesh(file.path()).unwrap();

    assert_eq!(loaded.points.len(), 4, "point count must survive round-trip");
    assert_eq!(loaded.polygons.len(), 4, "polygon count must survive round-trip");

    let eps = 1e-5_f32;
    // V0 = (0,0,0)
    assert!((loaded.points[0][0]).abs() < eps);
    assert!((loaded.points[0][1]).abs() < eps);
    assert!((loaded.points[0][2]).abs() < eps);
    // V1 = (1,0,0)
    assert!((loaded.points[1][0] - 1.0).abs() < eps);
    // V2 = (0.5,1,0)
    assert!((loaded.points[2][0] - 0.5).abs() < eps);
    assert!((loaded.points[2][1] - 1.0).abs() < eps);
    // V3 = (0.5,0.5,1)
    assert!((loaded.points[3][2] - 1.0).abs() < eps);
}

#[test]
fn test_ply_ascii_roundtrip_polygon_indices() {
    let mesh = tetrahedron();
    let file = NamedTempFile::new().unwrap();
    write_ply_ascii(file.path(), &mesh).unwrap();
    let loaded = read_ply_mesh(file.path()).unwrap();

    let expected: Vec<Vec<u32>> = vec![
        vec![0, 1, 2], vec![0, 1, 3], vec![0, 2, 3], vec![1, 2, 3],
    ];
    for (i, (e, g)) in expected.iter().zip(loaded.polygons.iter()).enumerate() {
        assert_eq!(e, g, "polygon {i} must survive round-trip");
    }
}

// ── Round-trip: binary little-endian ─────────────────────────────────────────

#[test]
fn test_ply_binary_le_roundtrip_coordinates() {
    let mesh = tetrahedron();
    let file = NamedTempFile::new().unwrap();
    write_ply_binary_le(file.path(), &mesh).unwrap();
    let loaded = read_ply_mesh(file.path()).unwrap();

    assert_eq!(loaded.points.len(), 4);
    assert_eq!(loaded.polygons.len(), 4);

    // Exact bit-for-bit: binary carries no formatting rounding.
    let eps = 1e-7_f32;
    assert!((loaded.points[0][0]).abs() < eps);
    assert!((loaded.points[1][0] - 1.0).abs() < eps);
    assert!((loaded.points[2][0] - 0.5).abs() < eps);
    assert!((loaded.points[2][1] - 1.0).abs() < eps);
    assert!((loaded.points[3][2] - 1.0).abs() < eps);
}

#[test]
fn test_ply_binary_le_roundtrip_polygon_indices() {
    let mesh = tetrahedron();
    let file = NamedTempFile::new().unwrap();
    write_ply_binary_le(file.path(), &mesh).unwrap();
    let loaded = read_ply_mesh(file.path()).unwrap();

    let expected: Vec<Vec<u32>> = vec![
        vec![0, 1, 2], vec![0, 1, 3], vec![0, 2, 3], vec![1, 2, 3],
    ];
    for (i, (e, g)) in expected.iter().zip(loaded.polygons.iter()).enumerate() {
        assert_eq!(e, g, "polygon {i} index mismatch in binary LE round-trip");
    }
}

// ── Normals round-trip ────────────────────────────────────────────────────────

#[test]
fn test_ply_ascii_normals_roundtrip() {
    let mesh = tetrahedron_with_normals();
    let file = NamedTempFile::new().unwrap();
    write_ply_ascii(file.path(), &mesh).unwrap();
    let loaded = read_ply_mesh(file.path()).unwrap();

    let got = match loaded.point_data.get("Normals").expect("Normals must be present") {
        AttributeArray::Normals { values } => values.clone(),
        other => panic!("expected Normals, got {other:?}"),
    };
    assert_eq!(got.len(), 4);
    let eps = 1e-5_f32;
    assert!((got[0][0] - 1.0).abs() < eps, "V0 normal x must be 1.0");
    assert!((got[1][1] - 1.0).abs() < eps, "V1 normal y must be 1.0");
    assert!((got[2][2] - 1.0).abs() < eps, "V2 normal z must be 1.0");
}

#[test]
fn test_ply_binary_le_normals_roundtrip() {
    let mesh = tetrahedron_with_normals();
    let file = NamedTempFile::new().unwrap();
    write_ply_binary_le(file.path(), &mesh).unwrap();
    let loaded = read_ply_mesh(file.path()).unwrap();

    let got = match loaded.point_data.get("Normals").unwrap() {
        AttributeArray::Normals { values } => values.clone(),
        _ => panic!("expected Normals"),
    };
    assert_eq!(got.len(), 4);
    let eps = 1e-7_f32;
    assert!((got[0][0] - 1.0).abs() < eps, "V0 normal x");
    assert!((got[1][1] - 1.0).abs() < eps, "V1 normal y");
    assert!((got[2][2] - 1.0).abs() < eps, "V2 normal z");
}

// ── Empty mesh ────────────────────────────────────────────────────────────────

#[test]
fn test_ply_ascii_empty_roundtrip() {
    let mesh = VtkPolyData::default();
    let file = NamedTempFile::new().unwrap();
    write_ply_ascii(file.path(), &mesh).unwrap();
    let loaded = read_ply_mesh(file.path()).unwrap();
    assert_eq!(loaded.points.len(), 0);
    assert_eq!(loaded.polygons.len(), 0);
}

#[test]
fn test_ply_binary_le_empty_roundtrip() {
    let mesh = VtkPolyData::default();
    let file = NamedTempFile::new().unwrap();
    write_ply_binary_le(file.path(), &mesh).unwrap();
    let loaded = read_ply_mesh(file.path()).unwrap();
    assert_eq!(loaded.points.len(), 0);
    assert_eq!(loaded.polygons.len(), 0);
}

// ── Negative / boundary ───────────────────────────────────────────────────────

#[test]
fn test_ply_big_endian_rejected() {
    let ply_be = b"ply\nformat binary_big_endian 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nelement face 0\nproperty list uchar int vertex_indices\nend_header\n";
    let result = parse_ply(ply_be);
    assert!(result.is_err(), "big-endian PLY must return Err");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("big-endian"),
        "error must mention 'big-endian', got: {msg}"
    );
}

#[test]
fn test_ply_missing_end_header() {
    let bad = b"ply\nformat ascii 1.0\nelement vertex 0\n";
    assert!(parse_ply(bad).is_err(), "missing end_header must return Err");
}

#[test]
fn test_ply_header_format_in_memory() {
    let mesh = tetrahedron();
    let mut buf = Vec::new();
    write_ply_ascii_to_writer(&mut buf, &mesh).unwrap();
    let text = String::from_utf8(buf).unwrap();

    assert!(text.starts_with("ply\n"), "must start with ply");
    assert!(text.contains("format ascii 1.0"), "must contain format line");
    assert!(text.contains("element vertex 4"), "must declare 4 vertices");
    assert!(text.contains("element face 4"), "must declare 4 faces");
    assert!(text.contains("end_header"), "must contain end_header");
}

#[test]
fn test_ply_binary_le_header_format_in_memory() {
    let mesh = tetrahedron();
    let mut buf = Vec::new();
    write_ply_binary_le_to_writer(&mut buf, &mesh).unwrap();
    // The header is ASCII; scan for the marker bytes.
    let header_end_pos = buf.windows(10).position(|w| w == b"end_header").unwrap();
    let header = std::str::from_utf8(&buf[..header_end_pos]).unwrap();
    assert!(header.contains("format binary_little_endian 1.0"));
    assert!(header.contains("element vertex 4"));
    assert!(header.contains("element face 4"));
}
