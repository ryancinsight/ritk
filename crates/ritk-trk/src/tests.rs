use super::parse::{TRK_MAGIC, encode_header, parse_header};
use super::*;

/// Build a default header with a simple identity affine.
fn default_header() -> TrkHeader {
    TrkHeader::default()
}

/// Build a minimal `.trk` file in-memory.
fn write_trk_bytes(header: &TrkHeader, streamlines: &[Vec<[f32; 3]>]) -> Vec<u8> {
    let header_bytes = encode_header(header);
    let mut buf = Vec::new();
    buf.extend_from_slice(&header_bytes);

    for points in streamlines {
        let n = points.len() as i32;
        buf.extend_from_slice(&n.to_le_bytes());
        for [x, y, z] in points {
            buf.extend_from_slice(&x.to_le_bytes());
            buf.extend_from_slice(&y.to_le_bytes());
            buf.extend_from_slice(&z.to_le_bytes());
        }
    }
    buf
}

#[test]
fn parse_default_header_round_trips() {
    let original = default_header();
    let bytes = encode_header(&original);
    let parsed = parse_header(&bytes.try_into().unwrap());
    assert_eq!(parsed.dim, original.dim);
    assert_eq!(parsed.voxel_size, original.voxel_size);
    assert_eq!(parsed.origin, original.origin);
    assert_eq!(parsed.n_scalars, original.n_scalars);
    assert_eq!(parsed.n_properties, original.n_properties);
    assert_eq!(parsed.vox_to_ras, original.vox_to_ras);
    assert_eq!(parsed.voxel_order, original.voxel_order);
    assert_eq!(
        parsed.image_orientation_patient,
        original.image_orientation_patient
    );
    assert_eq!(parsed.invert_x, original.invert_x);
    assert_eq!(parsed.invert_y, original.invert_y);
    assert_eq!(parsed.invert_z, original.invert_z);
    assert_eq!(parsed.swap_xy, original.swap_xy);
    assert_eq!(parsed.swap_yz, original.swap_yz);
    assert_eq!(parsed.swap_zx, original.swap_zx);
    assert_eq!(parsed.n_count, original.n_count);
    assert_eq!(parsed.version, original.version);
    assert_eq!(parsed.hdr_size, original.hdr_size);
}

#[test]
fn read_single_straight_streamline() {
    let mut header = default_header();
    header.n_count = 1;

    let points: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    let bytes = write_trk_bytes(&header, &[points]);
    let tractogram = TrkTractogram::read(&mut bytes.as_slice()).expect("valid .trk");

    assert_eq!(tractogram.streamlines.len(), 1);
    let polyline = &tractogram.streamlines[0];
    assert_eq!(polyline.len(), 3);
    assert!((polyline.points()[0].x - 0.0).abs() < 1e-6);
    assert!((polyline.points()[2].x - 2.0).abs() < 1e-6);
    assert_eq!(tractogram.header.n_count, 1);
}

#[test]
fn read_multiple_streamlines() {
    let mut header = default_header();
    header.n_count = 3;

    let streamlines = vec![
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        vec![[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        vec![[0.0, 2.0, 0.0], [1.0, 2.0, 0.0]],
    ];
    let bytes = write_trk_bytes(&header, &streamlines);
    let tractogram = TrkTractogram::read(&mut bytes.as_slice()).expect("valid .trk");

    assert_eq!(tractogram.streamlines.len(), 3);
    for (i, polyline) in tractogram.streamlines.iter().enumerate() {
        assert_eq!(polyline.len(), 2);
        assert!((polyline.points()[0].y - i as f64).abs() < 1e-6);
    }
}

#[test]
fn affine_identity_preserves_voxel_coordinates() {
    let mut header = default_header();
    header.n_count = 1;

    let points: Vec<[f32; 3]> = vec![[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]];
    let bytes = write_trk_bytes(&header, &[points]);
    let tractogram = TrkTractogram::read(&mut bytes.as_slice()).expect("valid .trk");

    let polyline = &tractogram.streamlines[0];
    assert!((polyline.points()[0].x - 10.0).abs() < 1e-4);
    assert!((polyline.points()[0].y - 20.0).abs() < 1e-4);
    assert!((polyline.points()[0].z - 30.0).abs() < 1e-4);
    assert!((polyline.points()[1].x - 40.0).abs() < 1e-4);
    assert!((polyline.points()[1].y - 50.0).abs() < 1e-4);
    assert!((polyline.points()[1].z - 60.0).abs() < 1e-4);
}

#[test]
fn affine_applies_translation() {
    let mut header = default_header();
    header.n_count = 1;
    // Translation: shift by (5, 10, 15).
    header.vox_to_ras = [
        [1.0, 0.0, 0.0, 5.0],
        [0.0, 1.0, 0.0, 10.0],
        [0.0, 0.0, 1.0, 15.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    let points: Vec<[f32; 3]> = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let bytes = write_trk_bytes(&header, &[points]);
    let tractogram = TrkTractogram::read(&mut bytes.as_slice()).expect("valid .trk");

    let polyline = &tractogram.streamlines[0];
    assert!((polyline.points()[0].x - 6.0).abs() < 1e-4);
    assert!((polyline.points()[0].y - 12.0).abs() < 1e-4);
    assert!((polyline.points()[0].z - 18.0).abs() < 1e-4);
    assert!((polyline.points()[1].x - 9.0).abs() < 1e-4);
    assert!((polyline.points()[1].y - 15.0).abs() < 1e-4);
    assert!((polyline.points()[1].z - 21.0).abs() < 1e-4);
}

#[test]
fn write_read_round_trip_with_non_identity_affine() {
    let mut header = default_header();
    header.n_count = 1;
    // Translation + scaling affine.
    header.vox_to_ras = [
        [2.0, 0.0, 0.0, 10.0],
        [0.0, 2.0, 0.0, 20.0],
        [0.0, 0.0, 2.0, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    // Build streamline in voxel coords, write via raw helper, read via
    // the public Reader (applies affine), write via public Writer (applies
    // inverse affine), then read again.
    let voxel_points: Vec<[f32; 3]> = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let bytes = write_trk_bytes(&header, &[voxel_points]);

    let read1 = TrkTractogram::read(&mut bytes.as_slice()).expect("first read");
    assert_eq!(read1.streamlines.len(), 1);
    let poly1 = &read1.streamlines[0];
    // With affine [2x+10, 2y+20, 2z+30]:
    // [1,2,3] → [12, 24, 36], [4,5,6] → [18, 30, 42]
    assert!((poly1.points()[0].x - 12.0).abs() < 1e-4);
    assert!((poly1.points()[0].y - 24.0).abs() < 1e-4);
    assert!((poly1.points()[0].z - 36.0).abs() < 1e-4);
    assert!((poly1.points()[1].x - 18.0).abs() < 1e-4);
    assert!((poly1.points()[1].y - 30.0).abs() < 1e-4);
    assert!((poly1.points()[1].z - 42.0).abs() < 1e-4);

    // Write back and re-read — round-trip through inverse affine.
    let mut out = Vec::new();
    read1.write(&mut out).expect("write");
    let read2 = TrkTractogram::read(&mut out.as_slice()).expect("second read");
    let poly2 = &read2.streamlines[0];
    for (p1, p2) in poly1.points().iter().zip(poly2.points().iter()) {
        assert!((p1.x - p2.x).abs() < 1e-3, "x: {} vs {}", p1.x, p2.x);
        assert!((p1.y - p2.y).abs() < 1e-3, "y: {} vs {}", p1.y, p2.y);
        assert!((p1.z - p2.z).abs() < 1e-3, "z: {} vs {}", p1.z, p2.z);
    }
}

#[test]
fn read_with_per_point_scalars() {
    let mut header = default_header();
    header.n_count = 1;
    header.n_scalars = 2;

    // Write a streamline with n_scalars=2 per-point scalars.
    let header_bytes = encode_header(&header);
    let mut buf = Vec::new();
    buf.extend_from_slice(&header_bytes);

    let n_points: i32 = 2;
    buf.extend_from_slice(&n_points.to_le_bytes());
    // Point 0: pos [1,2,3], scalars [0.5, -1.0]
    buf.extend_from_slice(&1.0f32.to_le_bytes());
    buf.extend_from_slice(&2.0f32.to_le_bytes());
    buf.extend_from_slice(&3.0f32.to_le_bytes());
    buf.extend_from_slice(&0.5f32.to_le_bytes());
    buf.extend_from_slice(&(-1.0f32).to_le_bytes());
    // Point 1: pos [4,5,6], scalars [2.5, 0.0]
    buf.extend_from_slice(&4.0f32.to_le_bytes());
    buf.extend_from_slice(&5.0f32.to_le_bytes());
    buf.extend_from_slice(&6.0f32.to_le_bytes());
    buf.extend_from_slice(&2.5f32.to_le_bytes());
    buf.extend_from_slice(&0.0f32.to_le_bytes());

    let tractogram = TrkTractogram::read(&mut buf.as_slice()).expect("valid .trk with scalars");
    assert_eq!(tractogram.streamlines.len(), 1);
    assert_eq!(tractogram.scalars.len(), 1);
    let s = &tractogram.scalars[0];
    assert_eq!(s.len(), 4); // 2 points × 2 scalars
    assert!((s[0] - 0.5).abs() < 1e-6);
    assert!((s[1] - (-1.0)).abs() < 1e-6);
    assert!((s[2] - 2.5).abs() < 1e-6);
    assert!((s[3] - 0.0).abs() < 1e-6);
}

#[test]
fn write_read_round_trip() {
    let mut header = default_header();
    header.n_count = 2;
    header.dim = [128, 128, 60];
    header.voxel_size = [2.0, 2.0, 2.0];

    let streamlines = vec![
        vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 5.0, 0.0]],
        vec![[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
    ];

    let bytes = write_trk_bytes(&header, &streamlines);

    // Read back.
    let tractogram = TrkTractogram::read(&mut bytes.as_slice()).expect("valid .trk");
    assert_eq!(tractogram.streamlines.len(), 2);

    // Write out again.
    let mut out = Vec::new();
    tractogram.write(&mut out).expect("write succeeds");

    // Re-read the re-written bytes.
    let tractogram2 = TrkTractogram::read(&mut out.as_slice()).expect("re-read valid .trk");
    assert_eq!(tractogram2.streamlines.len(), 2);

    // Compare points (should be identical after double round-trip).
    for (s1, s2) in tractogram
        .streamlines
        .iter()
        .zip(tractogram2.streamlines.iter())
    {
        assert_eq!(s1.len(), s2.len());
        for (p1, p2) in s1.points().iter().zip(s2.points().iter()) {
            assert!((p1.x - p2.x).abs() < 1e-3, "x: {} vs {}", p1.x, p2.x);
            assert!((p1.y - p2.y).abs() < 1e-3, "y: {} vs {}", p1.y, p2.y);
            assert!((p1.z - p2.z).abs() < 1e-3, "z: {} vs {}", p1.z, p2.z);
        }
    }
}

#[test]
fn rejects_invalid_magic() {
    let mut bytes = vec![0u8; 1000];
    bytes[..6].copy_from_slice(b"BAD__\0");
    let err = TrkTractogram::read(&mut bytes.as_slice()).expect_err("bad magic");
    assert!(matches!(err, TrkError::InvalidMagic { .. }));
}

#[test]
fn rejects_wrong_header_size() {
    let mut bytes = vec![0u8; 1000];
    bytes[..6].copy_from_slice(TRK_MAGIC);
    // Write hdr_size = 500 at bytes 996-999.
    bytes[996..1000].copy_from_slice(&500i32.to_le_bytes());
    let err = TrkTractogram::read(&mut bytes.as_slice()).expect_err("wrong hdr_size");
    assert!(matches!(err, TrkError::InvalidHeaderSize { .. }));
}

#[test]
fn rejects_single_point_streamline() {
    let mut header = default_header();
    header.n_count = 1;

    let points: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0]]; // only 1 point
    let bytes = write_trk_bytes(&header, &[points]);
    let err = TrkTractogram::read(&mut bytes.as_slice()).expect_err("single point");
    assert!(matches!(err, TrkError::InvalidPolyline { .. }));
}

#[test]
fn rejects_negative_point_count() {
    let mut header = default_header();
    header.n_count = 1;

    let header_bytes = encode_header(&header);
    let mut buf = Vec::new();
    buf.extend_from_slice(&header_bytes);
    buf.extend_from_slice(&(-1i32).to_le_bytes()); // negative count

    let err = TrkTractogram::read(&mut buf.as_slice()).expect_err("negative count");
    assert!(matches!(err, TrkError::InvalidPointCount { count: -1, .. }));
}
