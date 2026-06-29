//! Bounded allocation primitives for parsing untrusted input.
//!
//! File-format readers derive buffer sizes from header count/size fields
//! (`POINTS n`, `element vertex N`, NIfTI `dim`, MGH `width × height × depth`,
//! MINC dataset shape, …). Those fields are corruption- and attacker-controlled:
//! a header may name a multi-gigabyte payload whose bytes are not actually
//! present. Reserving the full claimed size up front turns such a header into an
//! out-of-memory abort on a tiny hostile file.
//!
//! These helpers cap *speculative* allocation — memory reserved before the bytes
//! are confirmed present — to [`MAX_EAGER_BYTES`] per chunk, so a lying length
//! fails fast with a truncation error after a bounded allocation while
//! legitimate large payloads still read in full. This is the single
//! authoritative home for the pattern (consumed by the VTK, MGH, MetaImage, and
//! MINC readers); do not reintroduce per-crate copies.

use std::io::Read;

/// Upper bound, in bytes, on speculative buffer allocation for a length field
/// that has not yet been validated against the actual input.
///
/// Chosen so a single legitimate read of a typical volume slice completes in one
/// allocation, while a hostile multi-gigabyte length only ever reserves this
/// much before the truncated input is detected.
pub const MAX_EAGER_BYTES: usize = 16 * 1024 * 1024; // 16 MiB

/// Capacity to reserve for a collection of `count` elements of `elem_size` bytes
/// when `count` comes from an unvalidated length field.
///
/// Caps the reservation at [`MAX_EAGER_BYTES`] so a hostile header cannot force a
/// huge speculative allocation; the collection still grows to its true length as
/// elements are pushed from validated input.
///
/// # Examples
/// ```
/// use ritk_core::io_bounds::{bounded_capacity, MAX_EAGER_BYTES};
/// // A small, legitimate count is reserved exactly.
/// assert_eq!(bounded_capacity(10, 4), 10);
/// // A hostile count is capped to the byte budget.
/// assert_eq!(bounded_capacity(usize::MAX, 4), MAX_EAGER_BYTES / 4);
/// ```
#[must_use]
pub fn bounded_capacity(count: usize, elem_size: usize) -> usize {
    count.min(MAX_EAGER_BYTES / elem_size.max(1))
}

/// Read `byte_count` bytes into a fresh `Vec<u8>`, bounding speculative
/// allocation to [`MAX_EAGER_BYTES`] per chunk.
///
/// `fill(offset, sub)` must populate `sub` with the `sub.len()` bytes located at
/// `offset` within the logical payload, or return an error (e.g. on truncation).
/// The buffer grows by at most one chunk between `fill` calls, so a `byte_count`
/// exceeding the available input yields the `fill` error after a bounded
/// allocation rather than a huge up-front reservation. This is the source-
/// agnostic primitive; [`read_exact_bounded`] is the [`Read`] specialization.
///
/// # Errors
/// Propagates any error returned by `fill`.
pub fn read_bounded_with<E, F>(byte_count: usize, mut fill: F) -> Result<Vec<u8>, E>
where
    F: FnMut(u64, &mut [u8]) -> Result<(), E>,
{
    let mut buf: Vec<u8> = Vec::new();
    let mut filled: usize = 0;
    while filled < byte_count {
        let chunk = (byte_count - filled).min(MAX_EAGER_BYTES);
        let start = buf.len();
        buf.resize(start + chunk, 0);
        fill(filled as u64, &mut buf[start..])?;
        filled += chunk;
    }
    Ok(buf)
}

/// Read exactly `byte_count` bytes from `reader`, bounding speculative
/// allocation to [`MAX_EAGER_BYTES`] per chunk.
///
/// Equivalent in result to [`Read::read_exact`] over a `vec![0u8; byte_count]`,
/// but the buffer grows only as bytes are confirmed present. A `byte_count`
/// larger than the remaining input yields [`std::io::ErrorKind::UnexpectedEof`]
/// after allocating at most one extra chunk. Callers add domain context to the
/// returned error.
///
/// # Errors
/// Returns the underlying [`std::io::Error`] (typically `UnexpectedEof` on
/// truncation).
pub fn read_exact_bounded<R: Read + ?Sized>(
    reader: &mut R,
    byte_count: usize,
) -> std::io::Result<Vec<u8>> {
    read_bounded_with(byte_count, |_offset, sub| reader.read_exact(sub))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn bounded_capacity_reserves_small_counts_exactly() {
        assert_eq!(bounded_capacity(0, 4), 0);
        assert_eq!(bounded_capacity(100, 12), 100);
    }

    #[test]
    fn bounded_capacity_caps_hostile_counts() {
        assert_eq!(bounded_capacity(usize::MAX, 4), MAX_EAGER_BYTES / 4);
        // elem_size 0 must not divide-by-zero.
        assert_eq!(bounded_capacity(usize::MAX, 0), MAX_EAGER_BYTES);
    }

    #[test]
    fn read_exact_bounded_reads_full_payload() {
        let mut cur = Cursor::new(vec![1u8, 2, 3, 4, 5]);
        let out = read_exact_bounded(&mut cur, 5).expect("read");
        assert_eq!(out, vec![1u8, 2, 3, 4, 5]);
    }

    #[test]
    fn read_exact_bounded_zero_count_is_empty() {
        let mut cur = Cursor::new(Vec::<u8>::new());
        assert!(read_exact_bounded(&mut cur, 0).expect("read").is_empty());
    }

    #[test]
    fn read_exact_bounded_errors_on_truncation_without_oom() {
        // Claims ~4 GiB but supplies 8 bytes: must reserve at most one chunk and
        // return UnexpectedEof rather than aborting on a 4 GiB reservation.
        let mut cur = Cursor::new(vec![0u8; 8]);
        let err = read_exact_bounded(&mut cur, 4_000_000_000).expect_err("must error");
        assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn read_bounded_with_propagates_fill_error() {
        let res: Result<Vec<u8>, &str> = read_bounded_with(10, |_off, _sub| Err("boom"));
        assert_eq!(res, Err("boom"));
    }

    #[test]
    fn read_bounded_with_passes_increasing_offsets() {
        // byte_count below one chunk → single fill at offset 0.
        let mut seen = Vec::new();
        let out: Result<Vec<u8>, ()> = read_bounded_with(4, |off, sub| {
            seen.push(off);
            sub.fill(7);
            Ok(())
        });
        assert_eq!(out.unwrap(), vec![7u8; 4]);
        assert_eq!(seen, vec![0]);
    }
}
