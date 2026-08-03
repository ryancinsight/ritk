use std::collections::HashMap;
use std::io::Write;

use gaia::Polyline;
use leto::geometry::Point3;

use super::*;

/// Build a minimal `.tck` file in-memory (default Float32LE).
fn write_tck_bytes(streamlines: &[Vec<[f64; 3]>]) -> Vec<u8> {
    write_tck_bytes_with_dt(streamlines, TckDatatype::Float32LE)
}

fn write_tck_bytes_with_dt(streamlines: &[Vec<[f64; 3]>], dt: TckDatatype) -> Vec<u8> {
    let mut buf = Vec::new();

    // Header.
    writeln!(buf, "mrtrix tracks").unwrap();
    writeln!(buf, "datatype: {}", dt.as_str()).unwrap();
    writeln!(buf, "count: {}", streamlines.len()).unwrap();
    writeln!(buf, "total_count: {}", streamlines.len()).unwrap();
    writeln!(buf, "END").unwrap();

    // Streamline data.
    for points in streamlines {
        for [x, y, z] in points {
            match dt {
                TckDatatype::Float32LE => {
                    buf.extend_from_slice(&(*x as f32).to_le_bytes());
                    buf.extend_from_slice(&(*y as f32).to_le_bytes());
                    buf.extend_from_slice(&(*z as f32).to_le_bytes());
                }
                TckDatatype::Float32BE => {
                    buf.extend_from_slice(&(*x as f32).to_be_bytes());
                    buf.extend_from_slice(&(*y as f32).to_be_bytes());
                    buf.extend_from_slice(&(*z as f32).to_be_bytes());
                }
                TckDatatype::Float64LE => {
                    buf.extend_from_slice(&x.to_le_bytes());
                    buf.extend_from_slice(&y.to_le_bytes());
                    buf.extend_from_slice(&z.to_le_bytes());
                }
                TckDatatype::Float64BE => {
                    buf.extend_from_slice(&x.to_be_bytes());
                    buf.extend_from_slice(&y.to_be_bytes());
                    buf.extend_from_slice(&z.to_be_bytes());
                }
            }
        }
        // Delimiter (NaN triplet).
        match dt {
            TckDatatype::Float32LE => {
                let nan = f32::NAN;
                buf.extend_from_slice(&nan.to_le_bytes());
                buf.extend_from_slice(&nan.to_le_bytes());
                buf.extend_from_slice(&nan.to_le_bytes());
            }
            TckDatatype::Float32BE => {
                let nan = f32::NAN;
                buf.extend_from_slice(&nan.to_be_bytes());
                buf.extend_from_slice(&nan.to_be_bytes());
                buf.extend_from_slice(&nan.to_be_bytes());
            }
            TckDatatype::Float64LE => {
                let nan = f64::NAN;
                buf.extend_from_slice(&nan.to_le_bytes());
                buf.extend_from_slice(&nan.to_le_bytes());
                buf.extend_from_slice(&nan.to_le_bytes());
            }
            TckDatatype::Float64BE => {
                let nan = f64::NAN;
                buf.extend_from_slice(&nan.to_be_bytes());
                buf.extend_from_slice(&nan.to_be_bytes());
                buf.extend_from_slice(&nan.to_be_bytes());
            }
        }
    }
    // Barrier (Inf triplet).
    match dt {
        TckDatatype::Float32LE => {
            let inf = f32::INFINITY;
            buf.extend_from_slice(&inf.to_le_bytes());
            buf.extend_from_slice(&inf.to_le_bytes());
            buf.extend_from_slice(&inf.to_le_bytes());
        }
        TckDatatype::Float32BE => {
            let inf = f32::INFINITY;
            buf.extend_from_slice(&inf.to_be_bytes());
            buf.extend_from_slice(&inf.to_be_bytes());
            buf.extend_from_slice(&inf.to_be_bytes());
        }
        TckDatatype::Float64LE => {
            let inf = f64::INFINITY;
            buf.extend_from_slice(&inf.to_le_bytes());
            buf.extend_from_slice(&inf.to_le_bytes());
            buf.extend_from_slice(&inf.to_le_bytes());
        }
        TckDatatype::Float64BE => {
            let inf = f64::INFINITY;
            buf.extend_from_slice(&inf.to_be_bytes());
            buf.extend_from_slice(&inf.to_be_bytes());
            buf.extend_from_slice(&inf.to_be_bytes());
        }
    }

    buf
}

#[test]
fn read_single_straight_streamline() {
    let points: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    let bytes = write_tck_bytes(&[points]);
    let tractogram = TckTractogram::read(bytes.as_slice()).expect("valid .tck");

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
    let bytes = write_tck_bytes(&streamlines);
    let tractogram = TckTractogram::read(bytes.as_slice()).expect("valid .tck");

    assert_eq!(tractogram.streamlines.len(), 3);
    for (i, polyline) in tractogram.streamlines.iter().enumerate() {
        assert_eq!(polyline.len(), 2);
        assert!((polyline.points()[0].y - i as f64).abs() < 1e-6);
    }
}

#[test]
fn read_empty_streamline_skipped() {
    // A file with one valid streamline, then an empty one (two consecutive
    // delimiters), then another valid one.
    let streamlines = vec![
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        vec![], // empty
        vec![[0.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
    ];
    let bytes = write_tck_bytes(&streamlines);
    let tractogram = TckTractogram::read(bytes.as_slice()).expect("valid .tck");
    assert_eq!(tractogram.streamlines.len(), 2);
}

#[test]
fn parse_header_keys() {
    let streamlines: Vec<Vec<[f64; 3]>> = vec![];
    let bytes = write_tck_bytes(&streamlines);
    let tractogram = TckTractogram::read(bytes.as_slice()).expect("valid .tck");

    assert_eq!(tractogram.header.datatype, TckDatatype::Float32LE);
    assert!(tractogram.header.fields.contains_key("count"));
}

#[test]
fn read_float64le_streamlines() {
    let points: Vec<[f64; 3]> = vec![[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]];
    let bytes = write_tck_bytes_with_dt(&[points], TckDatatype::Float64LE);
    let tractogram = TckTractogram::read(bytes.as_slice()).expect("valid .tck");

    assert_eq!(tractogram.streamlines.len(), 1);
    let polyline = &tractogram.streamlines[0];
    assert!((polyline.points()[0].x - 1.5).abs() < 1e-10);
    assert!((polyline.points()[1].z - 6.5).abs() < 1e-10);
}

#[test]
fn read_float64be_streamlines() {
    let points: Vec<[f64; 3]> = vec![[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]];
    let bytes = write_tck_bytes_with_dt(&[points], TckDatatype::Float64BE);
    let tractogram = TckTractogram::read(bytes.as_slice()).expect("valid .tck");

    assert_eq!(tractogram.streamlines.len(), 1);
    let polyline = &tractogram.streamlines[0];
    assert!((polyline.points()[0].x - 10.0).abs() < 1e-10);
    assert!((polyline.points()[1].x - 11.0).abs() < 1e-10);
}

#[test]
fn read_float32be_streamlines() {
    let points: Vec<[f64; 3]> = vec![[5.0, 6.0, 7.0], [15.0, 16.0, 17.0]];
    let bytes = write_tck_bytes_with_dt(&[points], TckDatatype::Float32BE);
    let tractogram = TckTractogram::read(bytes.as_slice()).expect("valid .tck");

    assert_eq!(tractogram.streamlines.len(), 1);
    let polyline = &tractogram.streamlines[0];
    assert!((polyline.points()[0].y - 6.0).abs() < 1e-4);
    assert!((polyline.points()[1].y - 16.0).abs() < 1e-4);
}

#[test]
fn write_read_round_trip() {
    // Build a tractogram programmatically.
    let mut header = TckHeader::default();
    header.datatype = TckDatatype::Float64LE;
    header.mrtrix_version = Some("3.0.4".into());

    let mut header_fields = HashMap::new();
    header_fields.insert("mrtrix_version".into(), "3.0.4".into());
    header_fields.insert("datatype".into(), "Float64LE".into());
    header.fields = header_fields;

    let points1 = vec![
        Point3::new(1.0, 2.0, 3.0),
        Point3::new(4.0, 5.0, 6.0),
        Point3::new(7.0, 8.0, 9.0),
    ];
    let points2 = vec![Point3::new(10.0, 20.0, 30.0), Point3::new(40.0, 50.0, 60.0)];

    let tractogram = TckTractogram {
        header,
        streamlines: vec![
            Polyline::new(points1).unwrap(),
            Polyline::new(points2).unwrap(),
        ],
    };

    let mut out = Vec::new();
    tractogram.write(&mut out).expect("write");

    let read_back = TckTractogram::read(out.as_slice()).expect("read back");
    assert_eq!(read_back.streamlines.len(), 2);
    assert_eq!(read_back.header.datatype, TckDatatype::Float64LE);
    assert_eq!(read_back.header.mrtrix_version.as_deref(), Some("3.0.4"));

    for (s1, s2) in tractogram
        .streamlines
        .iter()
        .zip(read_back.streamlines.iter())
    {
        assert_eq!(s1.len(), s2.len());
        for (p1, p2) in s1.points().iter().zip(s2.points().iter()) {
            assert!((p1.x - p2.x).abs() < 1e-10);
            assert!((p1.y - p2.y).abs() < 1e-10);
            assert!((p1.z - p2.z).abs() < 1e-10);
        }
    }
}

#[test]
fn write_read_round_trip_f32() {
    let header = TckHeader::default(); // Float32LE
    let points = vec![
        Point3::new(1.0, 2.0, 3.0),
        Point3::new(4.0, 5.0, 6.0),
    ];

    let tractogram = TckTractogram {
        header,
        streamlines: vec![Polyline::new(points).unwrap()],
    };

    let mut out = Vec::new();
    tractogram.write(&mut out).expect("write");

    let read_back = TckTractogram::read(out.as_slice()).expect("read back");
    assert_eq!(read_back.streamlines.len(), 1);
    assert_eq!(read_back.header.datatype, TckDatatype::Float32LE);

    let poly = &read_back.streamlines[0];
    assert!((poly.points()[0].x - 1.0).abs() < 1e-4);
    assert!((poly.points()[1].z - 6.0).abs() < 1e-4);
}

#[test]
fn rejects_invalid_magic() {
    let mut buf = Vec::new();
    writeln!(buf, "not mrtrix tracks").unwrap();
    let err = TckTractogram::read(buf.as_slice()).expect_err("bad magic");
    assert!(matches!(err, TckError::InvalidMagic(_)));
}

#[test]
fn rejects_unknown_datatype() {
    let mut buf = Vec::new();
    writeln!(buf, "mrtrix tracks").unwrap();
    writeln!(buf, "datatype: Float128").unwrap();
    writeln!(buf, "END").unwrap();
    let err = TckTractogram::read(buf.as_slice()).expect_err("bad datatype");
    assert!(matches!(err, TckError::UnknownDatatype(_)));
}

#[test]
fn rejects_single_point_streamline() {
    let points: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0]]; // only 1 point
    let bytes = write_tck_bytes(&[points]);
    let err = TckTractogram::read(bytes.as_slice()).expect_err("single point");
    assert!(matches!(err, TckError::InvalidPolyline { .. }));
}

#[test]
fn parse_transform() {
    let mut buf = Vec::new();
    writeln!(buf, "mrtrix tracks").unwrap();
    writeln!(buf, "datatype: Float32LE").unwrap();
    writeln!(buf, "transform: 1 0 0 -100 0 1 0 -120 0 0 1 -80 0 0 0 1").unwrap();
    writeln!(buf, "END").unwrap();

    // Add a small streamline + barrier.
    let nan = f32::NAN;
    let inf = f32::INFINITY;
    let data: [f32; 12] = [
        10.0, 20.0, 30.0, // point 1
        40.0, 50.0, 60.0, // point 2
        nan, nan, nan, // delimiter
        inf, inf, inf, // barrier
    ];
    for v in &data {
        buf.extend_from_slice(&v.to_le_bytes());
    }

    let tractogram = TckTractogram::read(buf.as_slice()).expect("valid .tck");
    let t = tractogram.header.transform.expect("transform parsed");
    assert!((t[0][3] - (-100.0)).abs() < 1e-10);
    assert!((t[1][3] - (-120.0)).abs() < 1e-10);
    assert!((t[2][3] - (-80.0)).abs() < 1e-10);
    assert_eq!(tractogram.streamlines.len(), 1);
}

#[test]
fn weights_sidecar_round_trips() {
    // Two streamlines with 3 and 2 points.
    let scalars: Vec<Box<[f32]>> = vec![
        vec![0.8f32, 0.75, 0.7].into_boxed_slice(),
        vec![0.4f32, 0.3].into_boxed_slice(),
    ];

    let mut buf = Vec::new();
    write_tck_weights(&scalars, TckDatatype::Float32LE, &mut buf).expect("write weights");

    let recovered = read_tck_weights(buf.as_slice()).expect("read weights");
    assert_eq!(recovered.len(), 2);
    assert_eq!(recovered[0].len(), 3);
    assert_eq!(recovered[1].len(), 2);
    assert!((recovered[0][0] - 0.8).abs() < 1e-6);
    assert!((recovered[0][1] - 0.75).abs() < 1e-6);
    assert!((recovered[0][2] - 0.7).abs() < 1e-6);
    assert!((recovered[1][0] - 0.4).abs() < 1e-6);
    assert!((recovered[1][1] - 0.3).abs() < 1e-6);
}

#[test]
fn weights_sidecar_float64le() {
    let scalars: Vec<Box<[f32]>> = vec![vec![1.0f32, 2.0].into_boxed_slice()];

    let mut buf = Vec::new();
    write_tck_weights(&scalars, TckDatatype::Float64LE, &mut buf).expect("write weights");

    let recovered = read_tck_weights(buf.as_slice()).expect("read weights");
    assert_eq!(recovered.len(), 1);
    assert_eq!(recovered[0].len(), 2);
    assert!((recovered[0][0] - 1.0).abs() < 1e-6);
    assert!((recovered[0][1] - 2.0).abs() < 1e-6);
}

#[test]
fn no_barrier_still_reads() {
    // EOF without explicit barrier should still work.
    let mut buf = Vec::new();
    writeln!(buf, "mrtrix tracks").unwrap();
    writeln!(buf, "datatype: Float32LE").unwrap();
    writeln!(buf, "END").unwrap();

    let data: [f32; 9] = [
        1.0, 2.0, 3.0, // point 1
        4.0, 5.0, 6.0, // point 2
        f32::NAN, f32::NAN, f32::NAN, // delimiter
        // No barrier.
    ];
    for v in &data {
        buf.extend_from_slice(&v.to_le_bytes());
    }

    let tractogram = TckTractogram::read(buf.as_slice()).expect("valid .tck");
    assert_eq!(tractogram.streamlines.len(), 1);
    assert_eq!(tractogram.streamlines[0].len(), 2);
}
