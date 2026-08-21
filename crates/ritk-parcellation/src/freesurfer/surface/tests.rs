use super::*;

/// Encode a surface in the FreeSurfer binary triangle format.
///
/// Written here rather than checked in as a fixture because the point is the
/// *format*: a reader tested against bytes some other code produced proves the
/// two agree, not that either matches the specification. These bytes are laid
/// out field by field from the format documented in the module.
fn encode(vertices: &[[f32; 3]], faces: &[[i32; 3]], comment: &str) -> Vec<u8> {
    let mut bytes = vec![0xFF, 0xFF, 0xFE];
    bytes.extend_from_slice(comment.as_bytes());
    bytes.extend_from_slice(b"\n\n");
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "test fixtures are a handful of elements"
    )]
    let counts = [vertices.len() as i32, faces.len() as i32];
    for count in counts {
        bytes.extend_from_slice(&count.to_be_bytes());
    }
    for vertex in vertices {
        for value in vertex {
            bytes.extend_from_slice(&value.to_be_bytes());
        }
    }
    for face in faces {
        for index in face {
            bytes.extend_from_slice(&index.to_be_bytes());
        }
    }
    bytes
}

fn tetrahedron() -> (Vec<[f32; 3]>, Vec<[i32; 3]>) {
    let vertices = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let faces = vec![[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
    (vertices, faces)
}

// ── The format ───────────────────────────────────────────────────────────

#[test]
fn a_surface_round_trips_through_the_binary_format() {
    let (vertices, faces) = tetrahedron();
    let bytes = encode(&vertices, &faces, "created by tests");

    let surface = Surface::read(bytes.as_slice()).expect("valid surface");

    assert_eq!(surface.vertex_count(), 4);
    assert_eq!(surface.faces().len(), 4);
    for (read, written) in surface.vertices().iter().zip(&vertices) {
        for axis in 0..3 {
            assert!((read[axis] - f64::from(written[axis])).abs() < 1.0e-9);
        }
    }
    assert_eq!(surface.faces()[3], [1, 2, 3]);
}

/// The comment is free text of unrecorded length, so the counts can only be
/// found by scanning to the double newline. A comment containing a single
/// newline must not end the scan early.
#[test]
fn a_comment_containing_a_newline_does_not_end_the_header() {
    let (vertices, faces) = tetrahedron();
    let bytes = encode(&vertices, &faces, "created by someone\non some date");

    let surface = Surface::read(bytes.as_slice()).expect("valid surface");
    assert_eq!(surface.vertex_count(), 4);
}

#[test]
fn an_empty_comment_is_accepted() {
    let (vertices, faces) = tetrahedron();
    let bytes = encode(&vertices, &faces, "");

    let surface = Surface::read(bytes.as_slice()).expect("valid surface");
    assert_eq!(surface.vertex_count(), 4);
}

/// The file is big-endian. Reading it as little-endian would give a vertex
/// count in the billions, which the range check catches — so this pins the
/// byte order rather than leaving it to chance.
#[test]
fn the_format_is_read_big_endian() {
    let (vertices, faces) = tetrahedron();
    let mut bytes = encode(&vertices, &faces, "x");
    // The vertex count sits immediately after the magic and the comment.
    let counts_at = 3 + 1 + 2;
    assert_eq!(&bytes[counts_at..counts_at + 4], &4_i32.to_be_bytes());
    // Swapping it to little-endian must be rejected, not silently misread.
    bytes[counts_at..counts_at + 4].copy_from_slice(&4_i32.to_le_bytes());
    let error = Surface::read(bytes.as_slice()).unwrap_err();
    assert!(
        matches!(error, FreeSurferSurfaceError::InvalidVertexCount { .. }),
        "got {error}"
    );
}

// ── Rejection ────────────────────────────────────────────────────────────

#[test]
fn wrong_magic_is_rejected() {
    let (vertices, faces) = tetrahedron();
    let mut bytes = encode(&vertices, &faces, "x");
    bytes[1] = 0x00;

    let error = Surface::read(bytes.as_slice()).unwrap_err();
    assert!(matches!(error, FreeSurferSurfaceError::InvalidMagic { .. }));
}

/// A face naming a vertex the surface does not have would index out of bounds
/// at every later use, so it is refused where the file is read.
#[test]
fn a_face_referencing_a_missing_vertex_is_rejected() {
    let (vertices, _) = tetrahedron();
    let bytes = encode(&vertices, &[[0, 1, 9]], "x");

    let error = Surface::read(bytes.as_slice()).unwrap_err();
    assert!(
        matches!(error, FreeSurferSurfaceError::MalformedLabelTable { .. }),
        "got {error}"
    );
}

#[test]
fn a_negative_vertex_index_is_rejected() {
    let (vertices, _) = tetrahedron();
    let bytes = encode(&vertices, &[[0, 1, -2]], "x");
    assert!(Surface::read(bytes.as_slice()).is_err());
}

#[test]
fn a_truncated_file_is_rejected() {
    let (vertices, faces) = tetrahedron();
    let bytes = encode(&vertices, &faces, "x");
    let truncated = &bytes[..bytes.len() - 8];

    let error = Surface::read(truncated).unwrap_err();
    assert!(
        matches!(error, FreeSurferSurfaceError::Io(_)),
        "got {error}"
    );
}

#[test]
fn a_non_finite_coordinate_is_rejected() {
    let bytes = encode(&[[0.0, f32::NAN, 0.0]], &[], "x");
    assert!(Surface::read(bytes.as_slice()).is_err());
}

// ── Construction and translation ─────────────────────────────────────────

#[test]
fn a_hand_built_surface_validates_its_faces() {
    assert!(Surface::new(vec![[0.0; 3]; 3], vec![[0, 1, 2]]).is_ok());
    assert!(Surface::new(vec![[0.0; 3]; 3], vec![[0, 1, 3]]).is_err());
    assert!(Surface::new(vec![[0.0, f64::INFINITY, 0.0]], Vec::new()).is_err());
}

/// The translation is how a surface leaves its stored frame for a volume's, so
/// it must move every vertex and leave the topology alone.
#[test]
fn translation_shifts_every_vertex_and_keeps_the_faces() {
    let surface = Surface::new(
        vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        vec![[0, 1, 2]],
    )
    .expect("valid surface");

    let moved = surface.translated([10.0, -20.0, 0.5]);

    assert_eq!(moved.faces(), surface.faces());
    assert_eq!(moved.vertices()[0], [11.0, -18.0, 3.5]);
    assert_eq!(moved.vertices()[2], [17.0, -12.0, 9.5]);
}
