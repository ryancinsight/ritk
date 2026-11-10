//! One fixture per header and payload failure of the MetaImage reader.

#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]
use crate::MetaImageError;
use coeus_core::SequentialBackend;
use std::path::Path;
use tempfile::tempdir;

const BASE_HEADER: &str = "ObjectType = Image\nNDims = 3\nTransformMatrix = 1 0 0 0 1 0 0 0 1\n\
     Offset = 0 0 0\nElementSpacing = 1 1 1\n";

fn read(path: &Path) -> MetaImageError {
    crate::read_metaimage(path, &SequentialBackend).expect_err("the fixture is malformed")
}

fn write_header(dir: &Path, name: &str, header: &str, tail: &[u8]) -> std::path::PathBuf {
    let path = dir.join(name);
    let mut bytes = header.as_bytes().to_vec();
    bytes.extend_from_slice(tail);
    std::fs::write(&path, bytes).unwrap();
    path
}

#[test]
fn a_non_utf8_header_is_a_header_read_failure() {
    let dir = tempdir().unwrap();
    let path = write_header(dir.path(), "binary.mha", "", &[0xFF, 0xFE, b'\n']);
    let error = read(&path);
    assert!(
        matches!(&error, MetaImageError::HeaderRead { source }
            if source.kind() == std::io::ErrorKind::InvalidData),
        "got: {error}"
    );
}

#[test]
fn a_header_without_element_data_file_is_rejected() {
    let dir = tempdir().unwrap();
    let path = write_header(dir.path(), "no_edf.mha", BASE_HEADER, b"");
    assert!(matches!(
        read(&path),
        MetaImageError::MissingElementDataFile { .. }
    ));
}

#[test]
fn a_header_without_offset_or_position_is_rejected() {
    let dir = tempdir().unwrap();
    let header = "NDims = 3\nDimSize = 1 1 1\nElementSpacing = 1 1 1\n\
                  ElementType = MET_FLOAT\nElementDataFile = LOCAL\n";
    let path = write_header(dir.path(), "no_offset.mha", header, &[0; 4]);
    let error = read(&path);
    assert!(
        matches!(error, MetaImageError::MissingOffset),
        "got: {error}"
    );
    assert_eq!(
        error.to_string(),
        "missing 'Offset' (or 'Position') in MetaImage header"
    );
}

#[test]
fn a_non_integer_ndims_is_rejected() {
    let dir = tempdir().unwrap();
    let header = "NDims = x\nElementDataFile = LOCAL\n";
    let path = write_header(dir.path(), "ndims_text.mha", header, b"");
    assert!(matches!(read(&path), MetaImageError::InvalidNDims { .. }));
}

#[test]
fn four_dimensions_are_unsupported() {
    let dir = tempdir().unwrap();
    let header = "NDims = 4\nElementDataFile = LOCAL\n";
    let path = write_header(dir.path(), "ndims_four.mha", header, b"");
    let error = read(&path);
    assert!(
        matches!(error, MetaImageError::UnsupportedNDims { found: 4 }),
        "got: {error}"
    );
}

#[test]
fn a_non_numeric_dim_size_names_the_field() {
    let dir = tempdir().unwrap();
    let header =
        format!("{BASE_HEADER}DimSize = a b c\nElementType = MET_FLOAT\nElementDataFile = LOCAL\n");
    let path = write_header(dir.path(), "dim_text.mha", &header, b"");
    let error = read(&path);
    assert!(
        matches!(
            error,
            MetaImageError::InvalidField {
                field: "DimSize",
                ..
            }
        ),
        "got: {error}"
    );
}

#[test]
fn a_byte_count_beyond_usize_is_rejected() {
    let dir = tempdir().unwrap();
    let header = format!(
        "{BASE_HEADER}DimSize = 4611686018427387904 1 1\nElementType = MET_DOUBLE\n\
         ElementDataFile = LOCAL\n"
    );
    let path = write_header(dir.path(), "bytes_overflow.mha", &header, b"");
    let error = read(&path);
    assert!(
        matches!(
            error,
            MetaImageError::ByteCountOverflow {
                voxels: 4_611_686_018_427_387_904,
                elem_size: 8,
            }
        ),
        "got: {error}"
    );
}

#[test]
fn a_missing_detached_payload_file_is_a_payload_read_failure() {
    let dir = tempdir().unwrap();
    let header = format!(
        "{BASE_HEADER}DimSize = 1 1 1\nElementType = MET_FLOAT\nElementDataFile = absent.raw\n"
    );
    let path = write_header(dir.path(), "detached.mhd", &header, b"");
    let error = read(&path);
    assert!(
        matches!(&error, MetaImageError::PayloadRead { path, source }
            if path.ends_with("absent.raw") && source.kind() == std::io::ErrorKind::NotFound),
        "got: {error}"
    );
}

#[test]
fn a_compressed_payload_that_is_not_zlib_fails_to_inflate() {
    let dir = tempdir().unwrap();
    let header = format!(
        "{BASE_HEADER}CompressedData = True\nDimSize = 1 1 1\nElementType = MET_FLOAT\n\
         ElementDataFile = LOCAL\n"
    );
    let path = write_header(dir.path(), "not_zlib.mha", &header, &[0xFF; 16]);
    assert!(matches!(read(&path), MetaImageError::Inflate { .. }));
}
