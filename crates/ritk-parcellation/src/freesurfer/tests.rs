use super::*;

/// A minimal valid `.annot` file with 3 vertices and 2 label table entries.
fn minimal_annot_bytes() -> Vec<u8> {
    let mut buf = Vec::new();

    // Magic.
    buf.extend_from_slice(&(-2i32).to_le_bytes());

    // Vertex count.
    buf.extend_from_slice(&(3i32).to_le_bytes());

    // Label table: 2 entries.
    buf.extend_from_slice(&(2i32).to_le_bytes());

    // Entry 0: structure_idx=0, name="Unknown"
    buf.extend_from_slice(&(0i32).to_le_bytes()); // struct idx
    buf.extend_from_slice(&(7i32).to_le_bytes()); // name length
    buf.extend_from_slice(b"Unknown"); // name (no null needed, exact length)
    buf.extend_from_slice(&[0u8; 16]); // RGBA

    // Entry 1: structure_idx=1001, name="precentral-L"
    buf.extend_from_slice(&(1001i32).to_le_bytes());
    buf.extend_from_slice(&(12i32).to_le_bytes());
    buf.extend_from_slice(b"precentral-L");
    buf.extend_from_slice(&[0u8; 16]);

    // Per-vertex labels (3 vertices): [0, 1, 0] → [Unknown, precentral-L, Unknown].
    buf.extend_from_slice(&(0i32).to_le_bytes());
    buf.extend_from_slice(&(1i32).to_le_bytes());
    buf.extend_from_slice(&(0i32).to_le_bytes());

    // Color table: 1 entry (can differ from label table count).
    buf.extend_from_slice(&(1i32).to_le_bytes());
    buf.extend_from_slice(&(1001i32).to_le_bytes());
    buf.extend_from_slice(&(12i32).to_le_bytes());
    buf.extend_from_slice(b"precentral-L");
    buf.extend_from_slice(&[0u8; 16]);

    // Per-vertex colours — one i32 (4 bytes) per vertex.
    buf.extend_from_slice(&[0u8; 12]); // 3 vertices × 4 bytes (i32 LE)

    buf
}

#[test]
fn read_minimal_annot() {
    let bytes = minimal_annot_bytes();
    let annot = SurfaceAnnotation::read(bytes.as_slice()).expect("valid .annot");

    assert_eq!(annot.vertex_count, 3);
    assert_eq!(annot.label_table.len(), 2);
    assert_eq!(annot.label_table[0].1, "Unknown");
    assert_eq!(annot.label_table[1].0, 1001);
    assert_eq!(annot.label_table[1].1, "precentral-L");

    assert_eq!(annot.vertex_labels.len(), 3);
    assert_eq!(annot.vertex_labels[0], 0);
    assert_eq!(annot.vertex_labels[1], 1001);
    assert_eq!(annot.vertex_labels[2], 0);
}

#[test]
fn reject_invalid_magic() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&(42i32).to_le_bytes()); // not -2
    buf.extend_from_slice(&(3i32).to_le_bytes()); // vertex count
    buf.extend_from_slice(&(0i32).to_le_bytes()); // label table

    let err = SurfaceAnnotation::read(buf.as_slice()).expect_err("bad magic");
    assert!(matches!(
        err,
        FreeSurferSurfaceError::InvalidMagic { got: 42 }
    ));
}

#[test]
fn parse_freesurfer_lut() {
    let lut = "\
# FreeSurfer ColorLUT
# Lines starting with # are comments

0   Unknown                 0    0    0    0
1001 precentral-L           255  128  0    0
1005 postcentral-L          0    255  128  0
2001 ctx-lh-unknown          128  128  128  0
";

    let entries = read_freesurfer_lut(lut.as_bytes()).expect("valid LUT");
    assert_eq!(entries.len(), 4);
    assert_eq!(entries[0], (0, "Unknown".to_string()));
    assert_eq!(entries[1], (1001, "precentral-L".to_string()));
    assert_eq!(entries[2], (1005, "postcentral-L".to_string()));
    assert_eq!(entries[3], (2001, "ctx-lh-unknown".to_string()));
}

#[test]
fn lut_skips_non_numeric_first_tokens() {
    let lut = "\
# comment
VERSION 1.0
0   Unknown     0 0 0 0
1   Region-1    255 0 0 0
";
    let entries = read_freesurfer_lut(lut.as_bytes()).expect("valid LUT");
    assert_eq!(entries.len(), 2);
    // "VERSION" is skipped because "VERSION" is not a valid u32.
    assert_eq!(entries[0], (0, "Unknown".to_string()));
    assert_eq!(entries[1], (1, "Region-1".to_string()));
}
