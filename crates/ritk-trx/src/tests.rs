use super::*;
use std::collections::HashMap;

use gaia::Polyline;
use leto::geometry::Point3;

/// Build a minimal header with float32 dtype.
fn header_f32(nb_streamlines: u64, nb_points: u64) -> TrxHeader {
    TrxHeader {
        nb_streamlines,
        nb_points,
        dtype: "float32".into(),
        ..Default::default()
    }
}

/// Build a minimal header with float64 dtype.
fn header_f64(nb_streamlines: u64, nb_points: u64) -> TrxHeader {
    TrxHeader {
        nb_streamlines,
        nb_points,
        dtype: "float64".into(),
        ..Default::default()
    }
}

/// Encode streamlines as raw TRX positions + offsets.
fn encode_streamlines(streamlines: &[Vec<[f64; 3]>], dtype: &str) -> (Vec<u8>, Vec<u8>) {
    let mut positions = Vec::new();
    let mut offsets = Vec::new();
    let mut cursor: u64 = 0;

    for points in streamlines {
        offsets.extend_from_slice(&cursor.to_le_bytes());
        for [x, y, z] in points {
            match dtype {
                "float32" => {
                    positions.extend_from_slice(&(*x as f32).to_le_bytes());
                    positions.extend_from_slice(&(*y as f32).to_le_bytes());
                    positions.extend_from_slice(&(*z as f32).to_le_bytes());
                }
                "float64" => {
                    positions.extend_from_slice(&x.to_le_bytes());
                    positions.extend_from_slice(&y.to_le_bytes());
                    positions.extend_from_slice(&z.to_le_bytes());
                }
                _ => unreachable!(),
            }
            cursor += 1;
        }
    }
    offsets.extend_from_slice(&cursor.to_le_bytes()); // sentinel

    (positions, offsets)
}

#[test]
fn read_single_straight_streamline() {
    let points: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    let (pos, off) = encode_streamlines(&[points], "float32");
    let header = header_f32(1, 3);

    let tractogram = TrxTractogram::from_raw(&header, &pos, &off).expect("valid TRX");
    assert_eq!(tractogram.streamlines.len(), 1);
    let polyline = &tractogram.streamlines[0];
    assert_eq!(polyline.len(), 3);
    assert!((polyline.points()[0].x - 0.0).abs() < 1e-6);
    assert!((polyline.points()[2].x - 2.0).abs() < 1e-6);
}

#[test]
fn read_multiple_streamlines() {
    let streamlines = vec![
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        vec![[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        vec![[0.0, 2.0, 0.0], [1.0, 2.0, 0.0]],
    ];
    let (pos, off) = encode_streamlines(&streamlines, "float32");
    let header = header_f32(3, 6);

    let tractogram = TrxTractogram::from_raw(&header, &pos, &off).expect("valid TRX");
    assert_eq!(tractogram.streamlines.len(), 3);
    for (i, polyline) in tractogram.streamlines.iter().enumerate() {
        assert_eq!(polyline.len(), 2);
        assert!((polyline.points()[0].y - i as f64).abs() < 1e-6);
    }
}

#[test]
fn read_float64_streamlines() {
    let points: Vec<[f64; 3]> = vec![[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]];
    let (pos, off) = encode_streamlines(&[points], "float64");
    let header = header_f64(1, 2);

    let tractogram = TrxTractogram::from_raw(&header, &pos, &off).expect("valid TRX");
    assert_eq!(tractogram.streamlines.len(), 1);
    let polyline = &tractogram.streamlines[0];
    assert!((polyline.points()[0].x - 1.5).abs() < 1e-10);
    assert!((polyline.points()[1].z - 6.5).abs() < 1e-10);
}

#[test]
fn write_read_round_trip_f32() {
    let header = TrxHeader::default(); // float32
    let points1 = vec![
        Point3::new(1.0, 2.0, 3.0),
        Point3::new(4.0, 5.0, 6.0),
        Point3::new(7.0, 8.0, 9.0),
    ];
    let points2 = vec![Point3::new(10.0, 20.0, 30.0), Point3::new(40.0, 50.0, 60.0)];

    let tractogram = TrxTractogram {
        header: header.clone(),
        streamlines: vec![
            Polyline::new(points1).unwrap(),
            Polyline::new(points2).unwrap(),
        ],
        dpv_data: HashMap::new(),
    };

    let (hdr, pos, off, _dpv) = tractogram.to_raw().expect("encode");
    assert_eq!(hdr.nb_streamlines, 2);
    assert_eq!(hdr.nb_points, 5);

    let read_back = TrxTractogram::from_raw(&hdr, &pos, &off).expect("decode");
    assert_eq!(read_back.streamlines.len(), 2);
    for (s1, s2) in tractogram
        .streamlines
        .iter()
        .zip(read_back.streamlines.iter())
    {
        assert_eq!(s1.len(), s2.len());
        for (p1, p2) in s1.points().iter().zip(s2.points().iter()) {
            assert!((p1.x - p2.x).abs() < 1e-4);
            assert!((p1.y - p2.y).abs() < 1e-4);
            assert!((p1.z - p2.z).abs() < 1e-4);
        }
    }
}

#[test]
fn write_read_round_trip_f64() {
    let header = TrxHeader {
        dtype: "float64".into(),
        ..TrxHeader::default()
    };
    let points = vec![Point3::new(1.0, 2.0, 3.0), Point3::new(4.0, 5.0, 6.0)];

    let tractogram = TrxTractogram {
        header,
        streamlines: vec![Polyline::new(points).unwrap()],
        dpv_data: HashMap::new(),
    };

    let (hdr, pos, off, _dpv) = tractogram.to_raw().expect("encode");
    let read_back = TrxTractogram::from_raw(&hdr, &pos, &off).expect("decode");
    assert_eq!(read_back.streamlines.len(), 1);
    let poly = &read_back.streamlines[0];
    assert!((poly.points()[0].x - 1.0).abs() < 1e-10);
    assert!((poly.points()[1].z - 6.0).abs() < 1e-10);
}

#[test]
fn rejects_positions_length_mismatch() {
    let header = header_f32(1, 10);
    let (pos, off) = encode_streamlines(&[vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], "float32");
    let err = TrxTractogram::from_raw(&header, &pos, &off).expect_err("length mismatch");
    assert!(matches!(err, TrxError::PositionsLengthMismatch { .. }));
}

#[test]
fn rejects_offsets_length_mismatch() {
    let points: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let (pos, off) = encode_streamlines(&[points], "float32");
    let header = header_f32(5, 2); // says 5 streamlines but offsets has 2 entries
    let err = TrxTractogram::from_raw(&header, &pos, &off).expect_err("offset mismatch");
    assert!(matches!(err, TrxError::OffsetsLengthMismatch { .. }));
}

#[test]
fn rejects_sentinel_mismatch() {
    // Build positions for 10 points (30 f32s) to match nb_points=10.
    let n = 10;
    let mut positions = Vec::new();
    for _ in 0..n {
        positions.extend_from_slice(&0.0f32.to_le_bytes());
        positions.extend_from_slice(&0.0f32.to_le_bytes());
        positions.extend_from_slice(&0.0f32.to_le_bytes());
    }
    // Offsets: [0, 5, 10] — sentinel 10 but nb_points=10, nb_streamlines=2 would be OK.
    // We claim nb_streamlines=1, so offsets has 2 entries, sentinel 10 != nb_points(=10)
    // but nb_streamlines=1 means we need 2 offsets (0 and sentinel).
    // Sentinel=10 matches nb_points=10. So this won't fail sentinel check.
    //
    // Instead: build a valid set and mutate the sentinel.
    let mut off = Vec::new();
    off.extend_from_slice(&0u64.to_le_bytes());
    off.extend_from_slice(&5u64.to_le_bytes()); // sentinel 5, but nb_points=10
    let header = header_f32(1, 10); // nb_points=10, sentinel=5 → mismatch
    let err = TrxTractogram::from_raw(&header, &positions, &off).expect_err("sentinel mismatch");
    assert!(matches!(err, TrxError::SentinelMismatch { .. }));
}

#[test]
fn rejects_unsupported_dtype() {
    let streamlines: Vec<Vec<[f64; 3]>> = vec![];
    let (pos, off) = encode_streamlines(&streamlines, "float32");
    let mut header = header_f32(0, 0);
    header.dtype = "int32".into();
    let err = TrxTractogram::from_raw(&header, &pos, &off).expect_err("bad dtype");
    assert!(matches!(err, TrxError::UnsupportedDtype(_)));
}

#[test]
fn rejects_single_point_streamline() {
    let points: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0]];
    let (pos, off) = encode_streamlines(&[points], "float32");
    let header = header_f32(1, 1);
    let err = TrxTractogram::from_raw(&header, &pos, &off).expect_err("single point");
    assert!(matches!(err, TrxError::InvalidPolyline { .. }));
}

#[test]
fn header_json_round_trips() {
    let original = TrxHeader {
        nb_streamlines: 42,
        nb_points: 1024,
        dtype: "float64".into(),
        reference: Some(TrxReference {
            affine: Some([
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ]),
            dimensions: Some([128, 128, 60]),
            voxel_sizes: Some([2.0, 2.0, 2.0]),
            path: None,
        }),
        ..Default::default()
    };

    let json = serde_json::to_string_pretty(&original).expect("serialize");
    let parsed: TrxHeader = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.nb_streamlines, 42);
    assert_eq!(parsed.nb_points, 1024);
    assert_eq!(parsed.dtype, "float64");
    assert!(parsed.reference.is_some());
}

#[test]
fn empty_streamline_set() {
    let header = header_f32(0, 0);
    let (pos, off) = encode_streamlines(&[], "float32");
    let tractogram = TrxTractogram::from_raw(&header, &pos, &off).expect("empty valid");
    assert_eq!(tractogram.streamlines.len(), 0);
    assert_eq!(tractogram.header.nb_streamlines, 0);
}

#[test]
fn skips_zero_length_streamline() {
    // A 1-point streamline plus a sentinel — zero-length result after
    // Polyline validation failure (1 point). Ensure it doesn't panic.
    let points: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0]];
    let (pos, off) = encode_streamlines(&[points], "float32");
    let header = header_f32(1, 1);
    let err = TrxTractogram::from_raw(&header, &pos, &off).expect_err("too few points");
    assert!(matches!(err, TrxError::InvalidPolyline { .. }));
}

#[test]
fn dpv_round_trips_through_write_dir_and_read_dir() {
    let mut header = TrxHeader::default();
    header.dpv.insert(
        "FA".into(),
        TrxArrayDef {
            dtype: "float32".into(),
            n_components: 1,
        },
    );

    let points = vec![Point3::new(1.0, 2.0, 3.0), Point3::new(4.0, 5.0, 6.0)];
    let tractogram = TrxTractogram {
        header,
        streamlines: vec![Polyline::new(points).unwrap()],
        dpv_data: {
            let mut m = HashMap::new();
            // 2 points × 1 component float32.
            let fa_bytes: Vec<u8> = [0.8f32, 0.7f32]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            m.insert("FA".into(), fa_bytes);
            m
        },
    };

    // Write to a temp directory.
    let dir = std::env::temp_dir().join("ritk_trx_dpv_test");
    let _ = std::fs::remove_dir_all(&dir);
    tractogram.write_dir(&dir).expect("write_dir");

    // Read back.
    let read_back = TrxTractogram::read_dir(&dir).expect("read_dir");
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(read_back.header.dpv.len(), 1);
    assert_eq!(read_back.dpv_data.len(), 1);
    assert!(read_back.dpv_data.contains_key("FA"));

    // Decode the FA bytes back to f32 values.
    let fa_raw = &read_back.dpv_data["FA"];
    let fa_vals: Vec<f32> = fa_raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(fa_vals, vec![0.8, 0.7]);

    // Streamlines also survived.
    assert_eq!(read_back.streamlines.len(), 1);
    assert_eq!(read_back.streamlines[0].len(), 2);
}
