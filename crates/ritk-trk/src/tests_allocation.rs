//! Allocation-bound guards for the TrackVis reader and writer.
//!
//! The SEC-446-05 hostile-input tests: a length field the format's name
//! tables cannot describe must be rejected before any reservation, and the
//! measurement is the process-wide peak heap mark (see the allocator in
//! [`super::tests`]).

use super::parse::encode_header;
use super::tests::{default_header, peak_bytes_during};
use super::*;

/// A header claiming a huge per-point scalar count must not allocate for it.
///
/// `n_scalars` sits at offset 36 of the 1000-byte header and is fully
/// caller-controlled. It sets the per-point stride (`3 + n_scalars`) that sizes
/// the decode buffer, so a 1 KiB file could demand
/// `100_000 * (3 + 32_767) * 4` bytes — roughly 13 GB — before a single
/// coordinate was read. The same value indexes the header's fixed
/// `scalar_name[10][20]` table, so a count the format cannot name is malformed
/// and is rejected by value before any reservation.
///
/// The count is the largest the i16 field admits and the point count is the
/// largest the range check admits, so the fixture reproduces the full
/// `100_000 * (3 + 32_767) * 4`-byte demand. A smaller value inside the format's
/// range would make this test pass against the very defect it exists to catch.
/// Peak allocation is measured rather than inferred, because a large reservation
/// can simply succeed — asserting only on the returned error leaves the defect
/// invisible when it does.
#[test]
fn a_huge_scalar_count_does_not_drive_allocation() {
    const PEAK_LIMIT: usize = 64 * 1024 * 1024;
    /// The largest point count [`TrkTractogram::read`] accepts.
    const MAX_POINTS: i32 = 100_000;

    let mut header = default_header();
    header.n_count = 1;
    header.n_scalars = i16::MAX;
    let mut buf = Vec::new();
    buf.extend_from_slice(&encode_header(&header));
    buf.extend_from_slice(&MAX_POINTS.to_le_bytes());

    let (result, peak) = peak_bytes_during(|| TrkTractogram::read(&mut buf.as_slice()));

    let err = result.expect_err("a count the name table cannot describe is malformed");
    assert!(
        matches!(err, TrkError::InvalidScalarCount { count } if count == i16::MAX),
        "expected the scalar count itself to be reported, got {err:?}"
    );
    assert!(
        peak < PEAK_LIMIT,
        "reading a {}-byte file peaked at {peak} live bytes, above the {PEAK_LIMIT}-byte limit; \
         the header's scalar count is driving allocation",
        buf.len()
    );
}

/// A header claiming a huge per-streamline property count must not allocate.
///
/// Mirrors the scalar case for `n_properties` at offset 238 and the fixed
/// `property_name[10][20]` table it indexes.
#[test]
fn a_huge_property_count_does_not_drive_allocation() {
    const PEAK_LIMIT: usize = 64 * 1024 * 1024;
    /// The largest point count [`TrkTractogram::read`] accepts.
    const MAX_POINTS: i32 = 100_000;

    let mut header = default_header();
    header.n_count = 1;
    header.n_properties = i16::MAX;
    let mut buf = Vec::new();
    buf.extend_from_slice(&encode_header(&header));
    buf.extend_from_slice(&MAX_POINTS.to_le_bytes());

    let (result, peak) = peak_bytes_during(|| TrkTractogram::read(&mut buf.as_slice()));

    let err = result.expect_err("a count the name table cannot describe is malformed");
    assert!(
        matches!(err, TrkError::InvalidPropertyCount { count } if count == i16::MAX),
        "expected the property count itself to be reported, got {err:?}"
    );
    assert!(
        peak < PEAK_LIMIT,
        "reading a {}-byte file peaked at {peak} live bytes, above the {PEAK_LIMIT}-byte limit; \
         the header's property count is driving allocation",
        buf.len()
    );
}

/// The largest counts the format can name are accepted and round-trip.
///
/// Pins the boundary so a future tightening cannot silently reject files at the
/// format's documented limit of ten named scalars and ten named properties.
#[test]
fn field_counts_at_the_format_limit_round_trip() {
    const FIELD_COUNT: usize = 10;

    let mut header = default_header();
    header.n_count = 1;
    header.n_scalars = FIELD_COUNT as i16;
    header.n_properties = FIELD_COUNT as i16;

    let n_points = 3usize;
    let mut buf = Vec::new();
    buf.extend_from_slice(&encode_header(&header));
    buf.extend_from_slice(&(n_points as i32).to_le_bytes());
    for p in 0..n_points {
        for value in [p as f32, 1.0, 2.0] {
            buf.extend_from_slice(&value.to_le_bytes());
        }
        for s in 0..FIELD_COUNT {
            buf.extend_from_slice(&(s as f32).to_le_bytes());
        }
    }
    for s in 0..FIELD_COUNT {
        buf.extend_from_slice(&(100.0 + s as f32).to_le_bytes());
    }

    let tractogram = TrkTractogram::read(&mut buf.as_slice()).expect("counts at the format limit");
    assert_eq!(tractogram.streamlines.len(), 1);
    assert_eq!(tractogram.scalars[0].len(), n_points * FIELD_COUNT);
    assert_eq!(tractogram.properties[0].len(), FIELD_COUNT);
    assert!((tractogram.properties[0][FIELD_COUNT - 1] - 109.0).abs() < 1e-6);

    let mut out = Vec::new();
    tractogram
        .write(&mut out)
        .expect("limit counts are writable");
    let reread = TrkTractogram::read(&mut out.as_slice()).expect("re-read the written file");
    assert_eq!(reread.scalars[0].len(), n_points * FIELD_COUNT);
    assert_eq!(reread.properties[0].len(), FIELD_COUNT);
}

/// The writer refuses counts the format cannot name rather than emitting them.
///
/// A file whose declared stride exceeds its name table is one the reader above
/// would reject, so writing it would produce a file this crate cannot read back.
#[test]
fn the_writer_rejects_out_of_range_field_counts() {
    let mut header = default_header();
    header.n_scalars = i16::MAX;
    let tractogram = TrkTractogram {
        header,
        streamlines: Vec::new(),
        scalars: Vec::new(),
        properties: Vec::new(),
    };

    let mut out = Vec::new();
    let err = tractogram
        .write(&mut out)
        .expect_err("an unnameable count is malformed");
    assert!(
        matches!(err, TrkError::InvalidScalarCount { count } if count == i16::MAX),
        "expected the scalar count itself to be reported, got {err:?}"
    );
    assert!(
        out.is_empty(),
        "the writer must not emit a header for a file it rejects"
    );
}
