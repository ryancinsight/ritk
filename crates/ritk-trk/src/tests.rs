use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use super::parse::{TRK_MAGIC, encode_header, parse_header};
use super::*;

/// Live heap bytes across the test binary.
static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
/// High-water mark of [`LIVE_BYTES`], reset by [`peak_bytes_during`].
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

/// Allocator that records the high-water mark of live bytes.
///
/// A reader that reserves from an untrusted length field is a memory defect,
/// and memory is the thing to measure: inferring it from elapsed time would be
/// a wall-clock assertion, and `Vec::capacity` is not observable through the
/// public read API. Counters are relaxed atomics — no test depends on ordering
/// between threads, only on the magnitude of the peak.
struct PeakTrackingAllocator;

// SAFETY: every method forwards to `System` with the caller's layout unchanged,
// so the allocator contract is whatever `System` already guarantees. The
// counters are side effects on atomics and affect no returned pointer.
unsafe impl GlobalAlloc for PeakTrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: `layout` is forwarded exactly as received.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let live = LIVE_BYTES.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK_BYTES.fetch_max(live, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
        // SAFETY: `ptr` and `layout` are forwarded exactly as received.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: PeakTrackingAllocator = PeakTrackingAllocator;

/// Run `body` and return the peak live-byte high-water mark it reached.
///
/// The mark is process-wide, so a concurrently running sibling test inflates
/// it. Every other test in this crate works on kilobyte fixtures, so the
/// measurement stays usable for a threshold set orders of magnitude above that
/// noise floor.
fn peak_bytes_during<T>(body: impl FnOnce() -> T) -> (T, usize) {
    PEAK_BYTES.store(LIVE_BYTES.load(Ordering::Relaxed), Ordering::Relaxed);
    let value = body();
    (value, PEAK_BYTES.load(Ordering::Relaxed))
}

/// Build a default header with a simple identity affine.
fn default_header() -> TrkHeader {
    TrkHeader::default()
}

/// Build a minimal `.trk` file in-memory.
fn write_trk_bytes(header: &TrkHeader, streamlines: &[Vec<[f32; 3]>]) -> Vec<u8> {
    let header_bytes = encode_header(header);
    let mut buf = Vec::new();
    buf.extend_from_slice(&header_bytes);

    for points in streamlines {
        let n = points.len() as i32;
        buf.extend_from_slice(&n.to_le_bytes());
        for [x, y, z] in points {
            buf.extend_from_slice(&x.to_le_bytes());
            buf.extend_from_slice(&y.to_le_bytes());
            buf.extend_from_slice(&z.to_le_bytes());
        }
    }
    buf
}

#[test]
fn parse_default_header_round_trips() {
    let original = default_header();
    let bytes = encode_header(&original);
    let parsed = parse_header(&bytes.try_into().unwrap());
    assert_eq!(parsed.dim, original.dim);
    assert_eq!(parsed.voxel_size, original.voxel_size);
    assert_eq!(parsed.origin, original.origin);
    assert_eq!(parsed.n_scalars, original.n_scalars);
    assert_eq!(parsed.n_properties, original.n_properties);
    assert_eq!(parsed.vox_to_ras, original.vox_to_ras);
    assert_eq!(parsed.voxel_order, original.voxel_order);
    assert_eq!(
        parsed.image_orientation_patient,
        original.image_orientation_patient
    );
    assert_eq!(parsed.invert_x, original.invert_x);
    assert_eq!(parsed.invert_y, original.invert_y);
    assert_eq!(parsed.invert_z, original.invert_z);
    assert_eq!(parsed.swap_xy, original.swap_xy);
    assert_eq!(parsed.swap_yz, original.swap_yz);
    assert_eq!(parsed.swap_zx, original.swap_zx);
    assert_eq!(parsed.n_count, original.n_count);
    assert_eq!(parsed.version, original.version);
    assert_eq!(parsed.hdr_size, original.hdr_size);
}

#[test]
fn read_single_straight_streamline() {
    let mut header = default_header();
    header.n_count = 1;

    let points: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    let bytes = write_trk_bytes(&header, &[points]);
    let tractogram = TrkTractogram::read(&mut bytes.as_slice()).expect("valid .trk");

    assert_eq!(tractogram.streamlines.len(), 1);
    let polyline = &tractogram.streamlines[0];
    assert_eq!(polyline.len(), 3);
    assert!((polyline.points()[0].x - 0.0).abs() < 1e-6);
    assert!((polyline.points()[2].x - 2.0).abs() < 1e-6);
    assert_eq!(tractogram.header.n_count, 1);
}

#[test]
fn read_multiple_streamlines() {
    let mut header = default_header();
    header.n_count = 3;

    let streamlines = vec![
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        vec![[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        vec![[0.0, 2.0, 0.0], [1.0, 2.0, 0.0]],
    ];
    let bytes = write_trk_bytes(&header, &streamlines);
    let tractogram = TrkTractogram::read(&mut bytes.as_slice()).expect("valid .trk");

    assert_eq!(tractogram.streamlines.len(), 3);
    for (i, polyline) in tractogram.streamlines.iter().enumerate() {
        assert_eq!(polyline.len(), 2);
        assert!((polyline.points()[0].y - i as f64).abs() < 1e-6);
    }
}

#[test]
fn affine_identity_preserves_voxel_coordinates() {
    let mut header = default_header();
    header.n_count = 1;

    let points: Vec<[f32; 3]> = vec![[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]];
    let bytes = write_trk_bytes(&header, &[points]);
    let tractogram = TrkTractogram::read(&mut bytes.as_slice()).expect("valid .trk");

    let polyline = &tractogram.streamlines[0];
    assert!((polyline.points()[0].x - 10.0).abs() < 1e-4);
    assert!((polyline.points()[0].y - 20.0).abs() < 1e-4);
    assert!((polyline.points()[0].z - 30.0).abs() < 1e-4);
    assert!((polyline.points()[1].x - 40.0).abs() < 1e-4);
    assert!((polyline.points()[1].y - 50.0).abs() < 1e-4);
    assert!((polyline.points()[1].z - 60.0).abs() < 1e-4);
}

#[test]
fn affine_applies_translation() {
    let mut header = default_header();
    header.n_count = 1;
    // Translation: shift by (5, 10, 15).
    header.vox_to_ras = [
        [1.0, 0.0, 0.0, 5.0],
        [0.0, 1.0, 0.0, 10.0],
        [0.0, 0.0, 1.0, 15.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    let points: Vec<[f32; 3]> = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let bytes = write_trk_bytes(&header, &[points]);
    let tractogram = TrkTractogram::read(&mut bytes.as_slice()).expect("valid .trk");

    let polyline = &tractogram.streamlines[0];
    assert!((polyline.points()[0].x - 6.0).abs() < 1e-4);
    assert!((polyline.points()[0].y - 12.0).abs() < 1e-4);
    assert!((polyline.points()[0].z - 18.0).abs() < 1e-4);
    assert!((polyline.points()[1].x - 9.0).abs() < 1e-4);
    assert!((polyline.points()[1].y - 15.0).abs() < 1e-4);
    assert!((polyline.points()[1].z - 21.0).abs() < 1e-4);
}

#[test]
fn write_read_round_trip_with_non_identity_affine() {
    let mut header = default_header();
    header.n_count = 1;
    // Translation + scaling affine.
    header.vox_to_ras = [
        [2.0, 0.0, 0.0, 10.0],
        [0.0, 2.0, 0.0, 20.0],
        [0.0, 0.0, 2.0, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    // Build streamline in voxel coords, write via raw helper, read via
    // the public Reader (applies affine), write via public Writer (applies
    // inverse affine), then read again.
    let voxel_points: Vec<[f32; 3]> = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let bytes = write_trk_bytes(&header, &[voxel_points]);

    let read1 = TrkTractogram::read(&mut bytes.as_slice()).expect("first read");
    assert_eq!(read1.streamlines.len(), 1);
    let poly1 = &read1.streamlines[0];
    // With affine [2x+10, 2y+20, 2z+30]:
    // [1,2,3] → [12, 24, 36], [4,5,6] → [18, 30, 42]
    assert!((poly1.points()[0].x - 12.0).abs() < 1e-4);
    assert!((poly1.points()[0].y - 24.0).abs() < 1e-4);
    assert!((poly1.points()[0].z - 36.0).abs() < 1e-4);
    assert!((poly1.points()[1].x - 18.0).abs() < 1e-4);
    assert!((poly1.points()[1].y - 30.0).abs() < 1e-4);
    assert!((poly1.points()[1].z - 42.0).abs() < 1e-4);

    // Write back and re-read — round-trip through inverse affine.
    let mut out = Vec::new();
    read1.write(&mut out).expect("write");
    let read2 = TrkTractogram::read(&mut out.as_slice()).expect("second read");
    let poly2 = &read2.streamlines[0];
    for (p1, p2) in poly1.points().iter().zip(poly2.points().iter()) {
        assert!((p1.x - p2.x).abs() < 1e-3, "x: {} vs {}", p1.x, p2.x);
        assert!((p1.y - p2.y).abs() < 1e-3, "y: {} vs {}", p1.y, p2.y);
        assert!((p1.z - p2.z).abs() < 1e-3, "z: {} vs {}", p1.z, p2.z);
    }
}

#[test]
fn read_with_per_point_scalars() {
    let mut header = default_header();
    header.n_count = 1;
    header.n_scalars = 2;

    // Write a streamline with n_scalars=2 per-point scalars.
    let header_bytes = encode_header(&header);
    let mut buf = Vec::new();
    buf.extend_from_slice(&header_bytes);

    let n_points: i32 = 2;
    buf.extend_from_slice(&n_points.to_le_bytes());
    // Point 0: pos [1,2,3], scalars [0.5, -1.0]
    buf.extend_from_slice(&1.0f32.to_le_bytes());
    buf.extend_from_slice(&2.0f32.to_le_bytes());
    buf.extend_from_slice(&3.0f32.to_le_bytes());
    buf.extend_from_slice(&0.5f32.to_le_bytes());
    buf.extend_from_slice(&(-1.0f32).to_le_bytes());
    // Point 1: pos [4,5,6], scalars [2.5, 0.0]
    buf.extend_from_slice(&4.0f32.to_le_bytes());
    buf.extend_from_slice(&5.0f32.to_le_bytes());
    buf.extend_from_slice(&6.0f32.to_le_bytes());
    buf.extend_from_slice(&2.5f32.to_le_bytes());
    buf.extend_from_slice(&0.0f32.to_le_bytes());

    let tractogram = TrkTractogram::read(&mut buf.as_slice()).expect("valid .trk with scalars");
    assert_eq!(tractogram.streamlines.len(), 1);
    assert_eq!(tractogram.scalars.len(), 1);
    let s = &tractogram.scalars[0];
    assert_eq!(s.len(), 4); // 2 points × 2 scalars
    assert!((s[0] - 0.5).abs() < 1e-6);
    assert!((s[1] - (-1.0)).abs() < 1e-6);
    assert!((s[2] - 2.5).abs() < 1e-6);
    assert!((s[3] - 0.0).abs() < 1e-6);
}

#[test]
fn write_read_round_trip() {
    let mut header = default_header();
    header.n_count = 2;
    header.dim = [128, 128, 60];
    header.voxel_size = [2.0, 2.0, 2.0];

    let streamlines = vec![
        vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 5.0, 0.0]],
        vec![[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
    ];

    let bytes = write_trk_bytes(&header, &streamlines);

    // Read back.
    let tractogram = TrkTractogram::read(&mut bytes.as_slice()).expect("valid .trk");
    assert_eq!(tractogram.streamlines.len(), 2);

    // Write out again.
    let mut out = Vec::new();
    tractogram.write(&mut out).expect("write succeeds");

    // Re-read the re-written bytes.
    let tractogram2 = TrkTractogram::read(&mut out.as_slice()).expect("re-read valid .trk");
    assert_eq!(tractogram2.streamlines.len(), 2);

    // Compare points (should be identical after double round-trip).
    for (s1, s2) in tractogram
        .streamlines
        .iter()
        .zip(tractogram2.streamlines.iter())
    {
        assert_eq!(s1.len(), s2.len());
        for (p1, p2) in s1.points().iter().zip(s2.points().iter()) {
            assert!((p1.x - p2.x).abs() < 1e-3, "x: {} vs {}", p1.x, p2.x);
            assert!((p1.y - p2.y).abs() < 1e-3, "y: {} vs {}", p1.y, p2.y);
            assert!((p1.z - p2.z).abs() < 1e-3, "z: {} vs {}", p1.z, p2.z);
        }
    }
}

#[test]
fn rejects_invalid_magic() {
    let mut bytes = vec![0u8; 1000];
    bytes[..6].copy_from_slice(b"BAD__\0");
    let err = TrkTractogram::read(&mut bytes.as_slice()).expect_err("bad magic");
    assert!(matches!(err, TrkError::InvalidMagic { .. }));
}

#[test]
fn rejects_wrong_header_size() {
    let mut bytes = vec![0u8; 1000];
    bytes[..6].copy_from_slice(TRK_MAGIC);
    // Write hdr_size = 500 at bytes 996-999.
    bytes[996..1000].copy_from_slice(&500i32.to_le_bytes());
    let err = TrkTractogram::read(&mut bytes.as_slice()).expect_err("wrong hdr_size");
    assert!(matches!(err, TrkError::InvalidHeaderSize { .. }));
}

#[test]
fn rejects_single_point_streamline() {
    let mut header = default_header();
    header.n_count = 1;

    let points: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0]]; // only 1 point
    let bytes = write_trk_bytes(&header, &[points]);
    let err = TrkTractogram::read(&mut bytes.as_slice()).expect_err("single point");
    assert!(matches!(err, TrkError::InvalidPolyline { .. }));
}

#[test]
fn rejects_negative_point_count() {
    let mut header = default_header();
    header.n_count = 1;

    let header_bytes = encode_header(&header);
    let mut buf = Vec::new();
    buf.extend_from_slice(&header_bytes);
    buf.extend_from_slice(&(-1i32).to_le_bytes()); // negative count

    let err = TrkTractogram::read(&mut buf.as_slice()).expect_err("negative count");
    assert!(matches!(err, TrkError::InvalidPointCount { count: -1, .. }));
}

/// A header claiming a huge streamline count must not allocate for it.
///
/// `n_count` sits at offset 988 of a 1000-byte header and is fully
/// caller-controlled. Reserving from it let a 1000-byte file demand gigabytes
/// before a single record was read — the length-field-driven allocation class.
/// The count is a claim, so the read must be bounded by the bytes that
/// actually arrive.
///
/// The count used here is deliberately 100_000_000 — the largest value the
/// previous range check admitted, so a reader that reserves from the header
/// passes its own validation and then asks for roughly 4.8 GB across the three
/// vectors. Picking a count the old check rejected would have made this test
/// pass against the very defect it exists to catch.
///
/// Peak allocation is measured rather than inferred: the eager reservation
/// does not necessarily abort — a 4.8 GB request can simply succeed — so
/// asserting only on the returned error leaves the defect invisible. Elapsed
/// time is not an option either, since a wall-clock assertion is flaky by
/// construction.
///
/// Threshold: the input is about a kilobyte and decodes to one streamline, so
/// an honest read touches kilobytes, as do the sibling tests sharing this
/// binary. 64 MiB sits three orders of magnitude above that noise floor and
/// two below the 4.8 GB the defect demands.
#[test]
fn a_huge_streamline_count_does_not_drive_allocation() {
    const PEAK_LIMIT: usize = 64 * 1024 * 1024;

    let mut header = default_header();
    header.n_count = 100_000_000;
    let mut buf = Vec::new();
    buf.extend_from_slice(&encode_header(&header));
    // One well-formed streamline, then the file simply stops.
    buf.extend_from_slice(&2i32.to_le_bytes());
    for value in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0] {
        buf.extend_from_slice(&value.to_le_bytes());
    }

    let (result, peak) = peak_bytes_during(|| TrkTractogram::read(&mut buf.as_slice()));

    let err = result.expect_err("a file holding one record cannot satisfy a claim of 100 million");
    assert!(
        matches!(err, TrkError::UnexpectedEof { .. }),
        "expected the read to end at the real end of the input, got {err:?}"
    );
    assert!(
        peak < PEAK_LIMIT,
        "reading a {}-byte file peaked at {peak} live bytes, above the {PEAK_LIMIT}-byte limit; \
         the header's streamline count is driving allocation",
        buf.len()
    );
}

/// A negative streamline count is malformed and is rejected by value.
#[test]
fn a_negative_streamline_count_is_rejected() {
    let mut header = default_header();
    header.n_count = -1;
    let buf = encode_header(&header).to_vec();

    let err = TrkTractogram::read(&mut buf.as_slice()).expect_err("negative count is malformed");
    assert!(
        matches!(err, TrkError::InvalidStreamlineCount { count: -1 }),
        "expected the count itself to be reported, got {err:?}"
    );
}

/// A count larger than the records present yields exactly the records present.
///
/// Guards the growth path against silently padding the result to the claimed
/// length, which would fabricate empty streamlines.
#[test]
fn an_overstated_count_yields_only_the_records_that_decode() {
    let mut header = default_header();
    header.n_count = 2;
    let mut buf = Vec::new();
    buf.extend_from_slice(&encode_header(&header));
    for _ in 0..2 {
        buf.extend_from_slice(&2i32.to_le_bytes());
        for value in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            buf.extend_from_slice(&value.to_le_bytes());
        }
    }

    let tractogram = TrkTractogram::read(&mut buf.as_slice()).expect("two complete records");
    assert_eq!(tractogram.streamlines.len(), 2);
    assert_eq!(tractogram.streamlines[0].points().len(), 2);
}
