//! Read-budget guards of the MetaImage reader.
//!
//! Each fixture declares a size the budget rejects before the payload is read
//! or inflated; reverting the matching check in `read_metaimage` makes it fail.

#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]
use super::reader::{write_compressed_mha_with_dims, write_minimal_mha};
use crate::MetaImageError;
use coeus_core::SequentialBackend;
use ritk_codecs::{ImageReadBudget, ImageReadBudgetError, ImageReadResource};
use tempfile::tempdir;

/// The default budget rejects the same header from its declared size alone,
/// before the zlib stream is inflated.
#[test]
fn test_compressed_hostile_dimsize_exceeds_the_decoded_budget() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("hostile_compressed_default.mha");
    write_compressed_mha_with_dims(&path, &[0u8; 16], 1024, 1024, 1024);

    let error = crate::read_metaimage(&path, &SequentialBackend, ImageReadBudget::DEFAULT)
        .expect_err("a 4 GiB declared payload exceeds the 1 GiB decoded budget");
    assert!(matches!(
        error,
        MetaImageError::Budget {
            source: ImageReadBudgetError::Exceeded {
                resource: ImageReadResource::DecodedBytes,
                actual: 4_294_967_296,
                maximum: 1_073_741_824,
            }
        }
    ));
}

/// The decoded ceiling admits exactly the payload the header declares and
/// rejects one byte less.
#[test]
fn test_decoded_budget_admits_the_exact_declared_payload() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("exact_budget.mha");
    write_minimal_mha(&path, &[1.0, 2.0, 3.0, 4.0], 2, 2, 1, [1.0; 3], [0.0; 3]);

    let exact = ImageReadBudget::new(1 << 20, 16, 1).expect("test ceilings are nonzero");
    let image = crate::read_metaimage(&path, &SequentialBackend, exact)
        .expect("a 16-byte budget admits four f32 voxels");
    assert_eq!(
        image.data_slice().expect("contiguous host data"),
        &[1.0, 2.0, 3.0, 4.0]
    );

    let one_short = ImageReadBudget::new(1 << 20, 15, 1).expect("test ceilings are nonzero");
    let error = crate::read_metaimage(&path, &SequentialBackend, one_short)
        .expect_err("16 declared bytes exceed a 15-byte budget");
    assert!(matches!(
        error,
        MetaImageError::Budget {
            source: ImageReadBudgetError::Exceeded {
                resource: ImageReadResource::DecodedBytes,
                actual: 16,
                maximum: 15,
            }
        }
    ));
}

/// The encoded payload is read through a bounded reader, so a file past the
/// encoded ceiling is rejected without being buffered whole.
#[test]
fn test_payload_above_the_encoded_budget_is_rejected() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("encoded_budget.mha");
    write_minimal_mha(&path, &[1.0, 2.0, 3.0, 4.0], 2, 2, 1, [1.0; 3], [0.0; 3]);

    let budget = ImageReadBudget::new(8, 1 << 20, 1).expect("test ceilings are nonzero");
    let error = crate::read_metaimage(&path, &SequentialBackend, budget)
        .expect_err("a 16-byte payload exceeds an 8-byte encoded budget");
    assert!(matches!(
        error,
        MetaImageError::Budget {
            source: ImageReadBudgetError::Exceeded {
                resource: ImageReadResource::EncodedBytes,
                actual: 9,
                maximum: 8,
            }
        }
    ));
}

fn write_mha_with_element(path: &std::path::Path, element_type: &str, payload: &[u8]) {
    let header = format!(
        "ObjectType = Image\nNDims = 3\nBinaryData = True\nBinaryDataByteOrderMSB = False\n\
         CompressedData = False\nTransformMatrix = 1 0 0 0 1 0 0 0 1\nOffset = 0 0 0\n\
         CenterOfRotation = 0 0 0\nElementSpacing = 1 1 1\nDimSize = 2 2 1\n\
         ElementType = {element_type}\nElementDataFile = LOCAL\n"
    );
    let mut bytes = header.into_bytes();
    bytes.extend_from_slice(payload);
    std::fs::write(path, bytes).unwrap();
}

/// A narrow element type stores fewer bytes than the `f32` output occupies, so
/// the output size is checked on its own.
#[test]
fn test_f32_output_above_the_decoded_budget_is_rejected_for_narrow_elements() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("narrow.mha");
    write_mha_with_element(&path, "MET_UCHAR", &[1, 2, 3, 4]);

    let budget = ImageReadBudget::new(1 << 20, 15, 1).expect("test ceilings are nonzero");
    let error = crate::read_metaimage(&path, &SequentialBackend, budget)
        .expect_err("16 output bytes exceed a 15-byte budget that admits 4 stored bytes");
    assert!(matches!(
        error,
        MetaImageError::Budget {
            source: ImageReadBudgetError::Exceeded {
                resource: ImageReadResource::DecodedBytes,
                actual: 16,
                maximum: 15,
            }
        }
    ));
}

/// A wide element type stores more bytes than the `f32` output occupies, so
/// the stored payload size is checked on its own.
#[test]
fn test_wide_element_payload_above_the_decoded_budget_is_rejected() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("wide.mha");
    write_mha_with_element(&path, "MET_DOUBLE", &[0u8; 32]);

    let budget = ImageReadBudget::new(1 << 20, 16, 1).expect("test ceilings are nonzero");
    let error = crate::read_metaimage(&path, &SequentialBackend, budget)
        .expect_err("32 stored bytes exceed a 16-byte budget that admits the 16-byte output");
    assert!(matches!(
        error,
        MetaImageError::Budget {
            source: ImageReadBudgetError::Exceeded {
                resource: ImageReadResource::DecodedBytes,
                actual: 32,
                maximum: 16,
            }
        }
    ));
}
