//! Read-budget guards of the `.mif` reader.
//!
//! Each fixture declares a size in the header that the budget rejects before
//! any payload is read; reverting the matching check in `read_mif_series`
//! makes it fail.

use std::path::PathBuf;

use coeus_core::SequentialBackend;
use ritk_codecs::{ImageReadBudget, ImageReadBudgetError, ImageReadResource};
use ritk_image::Image;

use super::tests::{header_with, temp_mif_path, write_raw_mif};
use crate::{read_mif_series, MifError};

fn budget(encoded: u64, decoded: u64, volumes: u64) -> ImageReadBudget {
    ImageReadBudget::new(encoded, decoded, volumes).expect("test ceilings are nonzero")
}

fn read_with(
    path: &PathBuf,
    budget: ImageReadBudget,
) -> Result<Vec<Image<f32, SequentialBackend, 3>>, MifError> {
    read_mif_series(path, &SequentialBackend, budget)
}

/// A `dim` frame count above the volume ceiling is rejected from the header
/// alone: the fixture carries no payload, so any read would fail differently.
#[test]
fn a_frame_count_above_the_volume_budget_is_rejected_before_the_payload() {
    let path = write_raw_mif(&header_with("1 1 1 100", ". 0"), &[]);
    let result = read_with(&path, budget(1 << 20, 1 << 20, 99));
    let _ = std::fs::remove_file(&path);

    let err = result.expect_err("100 frames exceed a 99-volume budget");
    assert!(
        matches!(
            err,
            MifError::Budget {
                source: ImageReadBudgetError::Exceeded {
                    resource: ImageReadResource::SeriesVolumes,
                    actual: 100,
                    maximum: 99,
                }
            }
        ),
        "got: {err}"
    );
}

/// The payload the header implies is bounded by the encoded ceiling before it
/// is read: the fixture supplies 16 bytes for a 4 GiB declaration.
#[test]
fn a_declared_payload_above_the_encoded_budget_is_rejected() {
    let path = write_raw_mif(&header_with("1024 1024 1024", ". 0"), &[0u8; 16]);
    let result = read_with(&path, ImageReadBudget::DEFAULT);
    let _ = std::fs::remove_file(&path);

    let err = result.expect_err("a 4 GiB raw payload exceeds the 1 GiB encoded budget");
    assert!(
        matches!(
            err,
            MifError::Budget {
                source: ImageReadBudgetError::Exceeded {
                    resource: ImageReadResource::EncodedBytes,
                    actual: 4_294_967_296,
                    maximum: 1_073_741_824,
                }
            }
        ),
        "got: {err}"
    );
}

/// The `f32` output the header implies is bounded by the decoded ceiling.
#[test]
fn declared_voxel_storage_above_the_decoded_budget_is_rejected() {
    let path = write_raw_mif(&header_with("1 1 8", ". 0"), &[0u8; 32]);
    let result = read_with(&path, budget(1 << 20, 31, 1));
    let _ = std::fs::remove_file(&path);

    let err = result.expect_err("32 output bytes exceed a 31-byte budget");
    assert!(
        matches!(
            err,
            MifError::Budget {
                source: ImageReadBudgetError::Exceeded {
                    resource: ImageReadResource::DecodedBytes,
                    actual: 32,
                    maximum: 31,
                }
            }
        ),
        "got: {err}"
    );
}

/// A detached data file honours the offset check like inline data.
#[test]
fn a_detached_offset_beyond_the_data_file_is_an_overrun() {
    let path = temp_mif_path();
    let data_path = path.with_extension("dat");
    std::fs::write(&data_path, [0u8; 8]).expect("write detached data fixture");
    let data_name = data_path
        .file_name()
        .and_then(|name| name.to_str())
        .expect("fixture name is UTF-8");
    std::fs::write(&path, header_with("1 1 1", &format!("{data_name} 100")))
        .expect("write header fixture");

    let result = read_with(&path, ImageReadBudget::DEFAULT);
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&data_path);

    let err = result.expect_err("an offset of 100 is past the 8-byte data file");
    assert!(
        matches!(
            err,
            MifError::OffsetOverrun {
                offset: 100,
                available: 8,
            }
        ),
        "got: {err}"
    );
}
