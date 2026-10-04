use super::*;

#[test]
fn every_truncation_of_a_valid_nrrd_errors_or_reads_exactly() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("full.nrrd");
    let data: Vec<f32> = (0..2 * 3 * 4).map(sample_value).collect();
    write_inline_nrrd(&path, &data, 2, 3, 4, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]);
    let complete = std::fs::read(&path)?;

    let backend = SequentialBackend;
    let truncated = dir.path().join("cut.nrrd");
    for cut in 0..complete.len() {
        std::fs::write(&truncated, &complete[..cut])?;
        // A prefix is short by at least one byte, so it cannot carry both the
        // header and the payload the header promises.
        let result: Result<_> = crate::read_nrrd::<SequentialBackend, _>(&truncated, &backend);
        // Which rule a prefix breaks depends on where it was cut -- magic, a
        // missing or unparsable header field, a short payload -- so no single
        // cause holds across the sweep. The property under test is that no cut
        // reads: a prefix must never yield an image, which is what a silent
        // truncation would look like.
        if let Ok(image) = result {
            panic!(
                "cut {cut} of {} bytes read a {:?} image from a truncated file",
                complete.len(),
                image.shape()
            );
        }
    }

    let image = crate::read_nrrd::<SequentialBackend, _>(&path, &backend)
        .expect("the intact file must read");
    assert_eq!(image.shape(), [4, 3, 2]);
    Ok(())
}

/// Single-byte corruption yields an error or a self-consistent image.
///
/// The header is text, so a substituted byte can turn a digit into another
/// digit — producing a header that is well-formed but describes a different
/// volume than the payload holds. That must be caught by the payload length
/// check rather than producing a buffer inconsistent with its own shape.
#[test]
fn single_byte_corruption_of_a_nrrd_stays_self_consistent() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("full.nrrd");
    let data: Vec<f32> = (0..2 * 3 * 4).map(sample_value).collect();
    write_inline_nrrd(&path, &data, 2, 3, 4, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]);
    let complete = std::fs::read(&path)?;

    let backend = SequentialBackend;
    let corrupt_path = dir.path().join("corrupt.nrrd");
    for offset in 0..complete.len() {
        for byte in [b'0', b'9', 0x00, 0xFF] {
            let mut corrupt = complete.clone();
            corrupt[offset] = byte;
            std::fs::write(&corrupt_path, &corrupt)?;
            let Ok(image) = crate::read_nrrd::<SequentialBackend, _>(&corrupt_path, &backend)
            else {
                continue;
            };
            let shape = image.shape();
            assert_eq!(
                image.data().shape(),
                shape,
                "byte {byte:#04X} at offset {offset} produced a tensor disagreeing with its shape"
            );
        }
    }
    Ok(())
}
