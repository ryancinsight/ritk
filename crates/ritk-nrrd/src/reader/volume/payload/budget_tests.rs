//! NRRD input-budget admission tests.

use super::{parse_nrrd_raw, NrrdReadPurpose};
use crate::reader::stored::NrrdStoredReadError;
use anyhow::Result;
use flate2::write::GzEncoder;
use flate2::Compression;
use ritk_image_io::{ImageReadBudget, ImageReadBudgetError, ImageReadResource};
use std::io::Write;
use tempfile::tempdir;

fn write_nrrd(path: &std::path::Path, fields: &[&str], payload: &[u8]) -> Result<()> {
    let mut file = std::fs::File::create(path)?;
    writeln!(file, "NRRD0005")?;
    writeln!(file, "type: unsigned char")?;
    for field in fields {
        writeln!(file, "{field}")?;
    }
    writeln!(file)?;
    file.write_all(payload)?;
    Ok(())
}

fn budget(encoded: u64, decoded: u64, volumes: u64) -> ImageReadBudget {
    ImageReadBudget::new(encoded, decoded, volumes).expect("test ceilings are nonzero")
}

#[test]
fn encoded_source_limit_rejects_raw_ascii_and_gzip_before_decoding() -> Result<()> {
    let directory = tempdir()?;
    for (encoding, payload) in [
        ("raw", &b"1234"[..]),
        ("ascii", &b"1 2 3 4"[..]),
        ("gzip", &b"xxxx"[..]),
    ] {
        let path = directory.path().join(format!("{encoding}.nrrd"));
        write_nrrd(
            &path,
            &[
                "dimension: 3",
                "sizes: 4 1 1",
                &format!("encoding: {encoding}"),
            ],
            payload,
        )?;

        let error = parse_nrrd_raw(&path, budget(3, 4, 1), NrrdReadPurpose::StoredVolume)
            .expect_err("encoded input must be bounded before payload decoding");
        assert!(matches!(
            error,
            NrrdStoredReadError::ReadBudget {
                source: ImageReadBudgetError::Exceeded {
                    resource: ImageReadResource::EncodedBytes,
                    actual,
                    maximum: 3,
                }
            } if actual > 3
        ));
    }
    Ok(())
}

#[test]
fn decoded_sample_and_compute_output_limits_precede_payload_read() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("decoded.nrrd");
    write_nrrd(
        &path,
        &["dimension: 3", "sizes: 4 1 1", "encoding: raw"],
        b"1234",
    )?;

    let stored_error = parse_nrrd_raw(&path, budget(4, 3, 1), NrrdReadPurpose::StoredVolume)
        .expect_err("stored samples exceed the decoded-byte ceiling");
    assert!(matches!(
        stored_error,
        NrrdStoredReadError::ReadBudget {
            source: ImageReadBudgetError::Exceeded {
                resource: ImageReadResource::DecodedBytes,
                actual: 4,
                maximum: 3,
            }
        }
    ));

    let compute_error = parse_nrrd_raw(&path, budget(4, 15, 1), NrrdReadPurpose::ComputeF32)
        .expect_err("f32 materialization exceeds the output-byte ceiling");
    assert!(matches!(
        compute_error,
        NrrdStoredReadError::ReadBudget {
            source: ImageReadBudgetError::Exceeded {
                resource: ImageReadResource::DecodedBytes,
                actual: 16,
                maximum: 15,
            }
        }
    ));
    Ok(())
}

#[test]
fn gzip_byte_skip_counts_against_the_decoded_byte_budget() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("gzip-byte-skip.nrrd");
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(b"12345678DATA")?;
    let payload = encoder.finish()?;
    write_nrrd(
        &path,
        &[
            "dimension: 3",
            "sizes: 4 1 1",
            "encoding: gzip",
            "byte skip: 8",
        ],
        &payload,
    )?;

    let error = parse_nrrd_raw(&path, budget(128, 11, 1), NrrdReadPurpose::StoredVolume)
        .expect_err("expanded bytes skipped before the sample payload use the same bound");
    assert!(matches!(
        error,
        NrrdStoredReadError::ReadBudget {
            source: ImageReadBudgetError::Exceeded {
                resource: ImageReadResource::DecodedBytes,
                actual: 12,
                maximum: 11,
            }
        }
    ));
    Ok(())
}

#[test]
fn series_volume_limit_precedes_per_volume_allocation() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("series.nrrd");
    write_nrrd(
        &path,
        &[
            "dimension: 4",
            "sizes: 1 1 1 2",
            "kinds: domain domain domain list",
            "encoding: raw",
        ],
        b"12",
    )?;

    let error = parse_nrrd_raw(&path, budget(2, 2, 1), NrrdReadPurpose::StoredSeries)
        .expect_err("series volume count is bounded before output allocation");
    assert!(matches!(
        error,
        NrrdStoredReadError::ReadBudget {
            source: ImageReadBudgetError::Exceeded {
                resource: ImageReadResource::SeriesVolumes,
                actual: 2,
                maximum: 1,
            }
        }
    ));
    Ok(())
}

#[test]
fn detached_input_limit_uses_the_data_file_length() -> Result<()> {
    let directory = tempdir()?;
    let header = directory.path().join("detached.nhdr");
    let raw = directory.path().join("volume.raw");
    write_nrrd(
        &header,
        &[
            "dimension: 3",
            "sizes: 4 1 1",
            "encoding: raw",
            "data file: volume.raw",
        ],
        b"",
    )?;
    std::fs::write(&raw, b"1234")?;

    let error = parse_nrrd_raw(&header, budget(3, 4, 1), NrrdReadPurpose::StoredVolume)
        .expect_err("detached encoded data is bounded before payload allocation");
    assert!(matches!(
        error,
        NrrdStoredReadError::ReadBudget {
            source: ImageReadBudgetError::Exceeded {
                resource: ImageReadResource::EncodedBytes,
                actual: 4,
                maximum: 3,
            }
        }
    ));
    Ok(())
}

#[test]
fn detached_file_sets_are_rejected_before_opening_a_member() -> Result<()> {
    let directory = tempdir()?;
    for data_file in ["LIST", "slice%03d.raw"] {
        let path = directory.path().join("detached.nhdr");
        write_nrrd(
            &path,
            &[
                "dimension: 3",
                "sizes: 1 1 1",
                "encoding: raw",
                &format!("data file: {data_file}"),
            ],
            b"",
        )?;

        assert!(matches!(
            parse_nrrd_raw(
                &path,
                ImageReadBudget::DEFAULT,
                NrrdReadPurpose::StoredVolume,
            ),
            Err(NrrdStoredReadError::UnsupportedDetachedFileSet { data_file: rejected })
                if rejected == data_file
        ));
    }
    Ok(())
}
