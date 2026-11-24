use super::*;
use crate::test_support::{build_mgh_bytes, IDENTITY_DIR};
use crate::{MRI_FLOAT, MRI_INT, MRI_SHORT, MRI_UCHAR, VERSION};

#[test]
fn copy_preserves_all_supported_sample_lanes_and_includes_footer_in_budget() {
    for (code, payload) in [
        (MRI_UCHAR, vec![0xab]),
        (MRI_SHORT, (-257_i16).to_be_bytes().to_vec()),
        (MRI_INT, 16_777_217_i32.to_be_bytes().to_vec()),
        (MRI_FLOAT, 0x7fc0_1234_u32.to_be_bytes().to_vec()),
    ] {
        let mut input = build_mgh_bytes(
            VERSION,
            [1; 3],
            1,
            code,
            [1.0; 3],
            IDENTITY_DIR,
            [0.0; 3],
            &payload,
        );
        input.extend_from_slice(&[0x12, 0x34]);
        let length = u64::try_from(input.len()).expect("fixture length");
        let mut copied = Vec::new();
        copy_document(input.as_slice(), &mut copied, length).expect("exact fit");
        assert_eq!(copied, input);
        assert!(
            matches!(copy_document(input.as_slice(), &mut io::sink(), length - 1), Err(MghDocumentError::ExpandedLimit { maximum }) if maximum == length - 1)
        );
    }
}

#[test]
fn in_place_repack_preserves_later_members_and_rejects_damaged_trailers() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("frames.mgz");
    let raw = build_mgh_bytes(
        VERSION,
        [2, 1, 1],
        1,
        MRI_UCHAR,
        [1.0; 3],
        IDENTITY_DIR,
        [0.0; 3],
        &[17, 29],
    );
    let mut encoded = Vec::new();
    for member in [&raw[..HEADER_SIZE + 1], &raw[HEADER_SIZE + 1..]] {
        let mut gzip = GzEncoder::new(&mut encoded, Compression::fast());
        gzip.write_all(member)?;
        gzip.finish()?;
    }
    std::fs::write(&path, &encoded)?;
    transcode_mgh_document(&path, &path)?;
    let mut expanded = Vec::new();
    MultiGzDecoder::new(std::fs::File::open(&path)?).read_to_end(&mut expanded)?;
    assert_eq!(expanded, raw);
    *encoded.last_mut().expect("second member trailer") ^= 1;
    std::fs::write(&path, &encoded)?;
    assert!(matches!(
        transcode_mgh_document(&path, &path),
        Err(MghDocumentError::Io(_))
    ));
    assert_eq!(std::fs::read(path)?, encoded);
    Ok(())
}
