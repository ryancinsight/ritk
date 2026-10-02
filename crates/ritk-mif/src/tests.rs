//! Crate-level test utilities shared by `tests_reader.rs`, `tests_samples.rs`,
//! and `tests_writer.rs`; the header tests live in `tests_header.rs`.

/// The bytes of an inline `.mif` laid out the way MRtrix writes one.
///
/// `header` holds every header line before `file:`. MRtrix's
/// `MRtrix::create` (`core/formats/mrtrix.cpp`) writes `file: ` and then sets
/// the offset to the stream position plus 18, rounded up to a multiple of 4;
/// the bytes between the `END` line and the offset are zero. This helper
/// repeats that rule independently of the crate's writer.
pub(crate) fn mrtrix_inline_file(header: &str, payload: &[u8]) -> Vec<u8> {
    let position_after_key = header.len() + "file: ".len();
    let offset = (position_after_key + 18).next_multiple_of(4);
    let mut bytes = format!("{header}file: . {offset}\nEND\n").into_bytes();
    assert!(
        bytes.len() <= offset,
        "the offset must lie after the END line"
    );
    bytes.resize(offset, 0);
    bytes.extend_from_slice(payload);
    bytes
}
