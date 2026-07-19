use super::{read_obj_mesh, write_obj_mesh};
use crate::domain::vtk_data_object::{AttributeArray, VtkPolyData};
use crate::io::obj::reader::parse_obj;
use crate::io::obj::writer::write_obj_to_writer;
use tempfile::NamedTempFile;

// ── Test fixtures ─────────────────────────────────────────────────────────────

/// Regular tetrahedron with shared vertex topology.
///
/// Vertices:
///   V0 = (0, 0, 0)   V1 = (1, 0, 0)
///   V2 = (0.5, 1, 0) V3 = (0.5, 0.5, 1)
///
/// Faces (outward-oriented):
///   F0 = [V0,V1,V2]  F1 = [V0,V1,V3]
///   F2 = [V0,V2,V3]  F3 = [V1,V2,V3]
fn tetrahedron() -> VtkPolyData {
    VtkPolyData {
        points: vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ],
        polygons: vec![vec![0, 1, 2], vec![0, 1, 3], vec![0, 2, 3], vec![1, 2, 3]],
        ..Default::default()
    }
}

/// Single triangle with analytically derived per-vertex normals.
fn triangle_with_normals() -> VtkPolyData {
    let mut mesh = VtkPolyData {
        points: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
        polygons: vec![vec![0, 1, 2]],
        ..Default::default()
    };
    // Unit normals pointing in distinct axis directions — analytically exact.
    mesh.point_data.insert(
        "Normals".to_string(),
        AttributeArray::Normals {
            values: vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        },
    );
    mesh
}

// ── Round-trip ────────────────────────────────────────────────────────────────

#[test]
fn test_obj_roundtrip_coordinates() {
    let mesh = tetrahedron();
    let file = NamedTempFile::new().unwrap();
    write_obj_mesh(file.path(), &mesh).unwrap();
    let loaded = read_obj_mesh(file.path()).unwrap();

    assert_eq!(
        loaded.points.len(),
        4,
        "point count must survive round-trip"
    );
    assert_eq!(
        loaded.polygons.len(),
        4,
        "polygon count must survive round-trip"
    );

    let eps = 1e-6_f32;
    for (i, (orig, got)) in mesh.points.iter().zip(loaded.points.iter()).enumerate() {
        assert!(
            (orig[0] - got[0]).abs() < eps
                && (orig[1] - got[1]).abs() < eps
                && (orig[2] - got[2]).abs() < eps,
            "point {i}: expected {orig:?}, got {got:?}"
        );
    }
    for (i, (orig, got)) in mesh.polygons.iter().zip(loaded.polygons.iter()).enumerate() {
        assert_eq!(orig, got, "polygon {i} indices must survive round-trip");
    }
}

// ── Normals ───────────────────────────────────────────────────────────────────

#[test]
fn test_obj_normals_roundtrip() {
    let mesh = triangle_with_normals();
    let file = NamedTempFile::new().unwrap();
    write_obj_mesh(file.path(), &mesh).unwrap();
    let loaded = read_obj_mesh(file.path()).unwrap();

    let got_normals = match loaded
        .point_data
        .get("Normals")
        .expect("Normals must be present")
    {
        AttributeArray::Normals { values } => values.clone(),
        other => panic!("expected AttributeArray::Normals, got {other:?}"),
    };

    assert_eq!(got_normals.len(), 3, "one normal per point");
    let expected = [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let eps = 1e-6_f32;
    for (i, (e, g)) in expected.iter().zip(got_normals.iter()).enumerate() {
        assert!(
            (e[0] - g[0]).abs() < eps && (e[1] - g[1]).abs() < eps && (e[2] - g[2]).abs() < eps,
            "normal {i}: expected {e:?}, got {g:?}"
        );
    }
}

// ── Empty mesh ────────────────────────────────────────────────────────────────

#[test]
fn test_obj_empty_roundtrip() {
    let mesh = VtkPolyData::default();
    let file = NamedTempFile::new().unwrap();
    write_obj_mesh(file.path(), &mesh).unwrap();
    let loaded = read_obj_mesh(file.path()).unwrap();
    assert_eq!(loaded.points.len(), 0);
    assert_eq!(loaded.polygons.len(), 0);
    assert!(loaded.point_data.is_empty());
}

// ── Negative / boundary ───────────────────────────────────────────────────────

#[test]
fn test_obj_malformed_vertex_too_few_coords() {
    // Two coordinates instead of three → parse_vec3 must fail.
    let src = b"v 1.0 2.0\n" as &[u8];
    let result = parse_obj(src);
    assert!(
        result.is_err(),
        "expected Err for vertex with only 2 coordinates"
    );
}

#[test]
fn test_obj_malformed_face_non_numeric_index() {
    let src = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 abc\n" as &[u8];
    let result = parse_obj(src);
    assert!(result.is_err(), "expected Err for non-numeric face index");
}

#[test]
fn test_obj_malformed_face_zero_index() {
    // OBJ indices start at 1; index 0 is invalid.
    let src = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 1 2\n" as &[u8];
    let result = parse_obj(src);
    assert!(result.is_err(), "expected Err for zero face index");
}

#[test]
fn test_obj_comments_and_unknown_directives_skipped() {
    let src = b"# a comment\nmtllib ignored.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n" as &[u8];
    let poly = parse_obj(src).expect("valid OBJ must parse successfully");
    assert_eq!(poly.points.len(), 3);
    assert_eq!(poly.polygons.len(), 1);
    assert_eq!(poly.polygons[0], vec![0, 1, 2]);
}

#[test]
fn test_obj_face_v_slash_slash_n_format() {
    // v//n syntax: texcoord slot is empty, normal slot is present.
    let src =
        b"v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nvn 0 0 1\nvn 0 0 1\nf 1//1 2//2 3//3\n" as &[u8];
    let poly = parse_obj(src).expect("v//n face format must parse");
    assert_eq!(poly.points.len(), 3);
    assert_eq!(poly.polygons[0], vec![0, 1, 2]);
    let normals = match poly.point_data.get("Normals").unwrap() {
        AttributeArray::Normals { values } => values.clone(),
        _ => panic!("expected Normals"),
    };
    let eps = 1e-6_f32;
    for n in &normals {
        assert!(
            (n[2] - 1.0).abs() < eps,
            "normal z must be 1.0, got {:?}",
            n
        );
    }
}

#[test]
fn test_obj_writer_in_memory() {
    let mesh = tetrahedron();
    let mut buf = Vec::new();
    write_obj_to_writer(&mut buf, &mesh).unwrap();
    let text = String::from_utf8(buf).unwrap();
    // Must contain the RITK header comment.
    assert!(text.contains("# Written by RITK"), "header comment missing");
    // Must contain 4 vertex lines.
    let v_count = text.lines().filter(|l| l.starts_with("v ")).count();
    assert_eq!(v_count, 4, "expected 4 'v' lines, got {v_count}");
    // Must contain 4 face lines.
    let f_count = text.lines().filter(|l| l.starts_with("f ")).count();
    assert_eq!(f_count, 4, "expected 4 'f' lines, got {f_count}");
}
