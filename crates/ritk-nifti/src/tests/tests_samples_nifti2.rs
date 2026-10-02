//! NIfTI-2 files built field by field at the `nifti2.h` offsets, in both
//! byte orders, and the rescale applied to every volume of a series.

use super::tests_samples::{encode, image_of, put_scalar};
use super::*;
use consus_core::ByteOrder;
use ritk_codecs::sample::Rescale;

/// A two-volume `int16` NIfTI-2 file on a 2x2x2 grid with `y = 0.5x + 10`
/// written as the `f64` fields at 176 and 184.
fn raw_nifti2(order: ByteOrder, payload: &[u8]) -> Vec<u8> {
    let mut bytes = vec![0_u8; 544];
    put_scalar(&mut bytes[0..], 540_i32, order);
    bytes[4..12].copy_from_slice(b"n+2\0\r\n\x1a\n");
    put_scalar(&mut bytes[12..], 4_i16, order);
    put_scalar(&mut bytes[14..], 16_i16, order);
    for (index, dim) in [4_i64, 2, 2, 2, 2, 1, 1, 1].into_iter().enumerate() {
        put_scalar(&mut bytes[16 + index * 8..], dim, order);
    }
    for index in 0..8 {
        put_scalar(&mut bytes[104 + index * 8..], 1.0_f64, order);
    }
    put_scalar(&mut bytes[168..], 544_i64, order);
    put_scalar(&mut bytes[176..], 0.5_f64, order);
    put_scalar(&mut bytes[184..], 10.0_f64, order);
    bytes.extend_from_slice(payload);
    bytes
}

/// Sixteen stored samples: two volumes of eight.
const STORED: [i16; 16] = [
    -20,
    -2,
    0,
    1,
    2,
    4,
    100,
    i16::MAX,
    i16::MIN,
    -1,
    3,
    5,
    7,
    9,
    11,
    13,
];

#[test]
fn nifti2_series_rescales_every_volume_in_both_byte_orders() {
    for order in [ByteOrder::LittleEndian, ByteOrder::BigEndian] {
        let bytes = raw_nifti2(order, &encode(&STORED, order));
        let series =
            crate::read_nifti_series_from_bytes::<f64, _, _>(&bytes, &SequentialBackend, Exact)
                .expect("int16 widens to f64");
        assert_eq!(series.len(), 2, "{order:?}");
        for (volume, stored) in series.iter().zip(STORED.chunks_exact(8)) {
            let physical: Vec<f64> = stored
                .iter()
                .map(|&value| f64::from(value) * 0.5 + 10.0)
                .collect();
            assert_eq!(
                volume.data_slice().expect("contiguous"),
                physical,
                "{order:?}"
            );
        }
    }
}

#[test]
fn nifti2_series_reads_stored_samples_with_the_rescale() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("series.nii");
    std::fs::write(
        &path,
        raw_nifti2(ByteOrder::BigEndian, &encode(&STORED, ByteOrder::BigEndian)),
    )?;
    let (series, rescale) =
        crate::read_nifti_series_stored::<i16, _, _, _>(&path, &SequentialBackend, Exact)?;
    assert_eq!(rescale, Rescale::new(0.5, 10.0)?);
    let stored: Vec<i16> = series
        .iter()
        .flat_map(|volume| volume.data_slice().expect("contiguous").to_vec())
        .collect();
    assert_eq!(stored, STORED);

    let err = crate::read_nifti_series::<i16, _, _, _>(&path, &SequentialBackend, Exact)
        .expect_err("a rescale has no faithful int16 result");
    assert!(
        format!("{err:#}").contains("read_nifti_series_stored"),
        "{err:#}"
    );
    Ok(())
}

/// The typed reader hands back every cause, not only the outermost context.
#[test]
fn typed_reader_errors_keep_their_causes() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("series.nii");
    let stored = &STORED[..8];
    let mut bytes = raw_nifti2(
        ByteOrder::LittleEndian,
        &encode(stored, ByteOrder::LittleEndian),
    );
    put_scalar(&mut bytes[16..], 3_i64, ByteOrder::LittleEndian);
    put_scalar(&mut bytes[48..], 1_i64, ByteOrder::LittleEndian);
    std::fs::write(&path, bytes)?;
    let absent = crate::NiftiReader::new(SequentialBackend)
        .read::<i16, _, _>(dir.path().join("absent.nii"), Exact)
        .expect_err("the file does not exist");
    assert_eq!(absent.kind(), std::io::ErrorKind::NotFound, "{absent}");
    let err = crate::NiftiReader::new(SequentialBackend)
        .read::<u8, _, _>(&path, Exact)
        .expect_err("int16 does not widen to u8");
    assert_eq!(err.kind(), std::io::ErrorKind::Other);
    let mut causes = Vec::new();
    let mut cause: Option<&dyn std::error::Error> = Some(&err);
    while let Some(current) = cause {
        causes.push(current.to_string());
        cause = current.source();
    }
    assert!(
        causes
            .iter()
            .any(|text| text.contains("do not all have exact")),
        "{causes:?}"
    );
    Ok(())
}

#[test]
fn typed_writer_round_trips_and_keeps_the_io_kind() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("typed.nii");
    let image = image_of(STORED[8..].to_vec());
    crate::NiftiWriter::new(SequentialBackend).write(&path, &image)?;
    let loaded = crate::NiftiReader::new(SequentialBackend).read::<i16, _, _>(&path, Exact)?;
    assert_eq!(loaded.data_slice()?, &STORED[8..]);

    let missing = dir.path().join("absent").join("typed.nii");
    let err = crate::NiftiWriter::new(SequentialBackend)
        .write(&missing, &image)
        .expect_err("the parent directory does not exist");
    assert_eq!(err.kind(), std::io::ErrorKind::NotFound, "{err}");
    Ok(())
}
