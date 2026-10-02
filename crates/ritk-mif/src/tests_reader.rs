//! Integration tests for the `.mif` reader and writer.
//!
//! These tests round-trip images through the writer and reader to verify
//! voxel fidelity, spatial metadata preservation, and frame handling.

use std::path::PathBuf;

use coeus_core::SequentialBackend;
use ritk_codecs::sample::Exact;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};

use ritk_core::alloc_probe::{peak_bytes_during, PeakTrackingAllocator};

use crate::tests::mrtrix_inline_file;
use crate::{read_mif, read_mif_series, write_mif, write_mif_series};

// `#[global_allocator]` is per binary, so the declaration lives here while the
// mechanism lives in `ritk_core::alloc_probe`.
#[global_allocator]
static ALLOCATOR: PeakTrackingAllocator = PeakTrackingAllocator;

// ── Test helpers ─────────────────────────────────────────────────────────

/// A simple 2×3×4 volume with unique values so we can identify voxels
/// after round‑tripping.
fn make_test_image(backend: &SequentialBackend) -> Image<f32, SequentialBackend, 3> {
    let nz = 4_usize;
    let ny = 3;
    let nx = 2;
    let mut data = Vec::with_capacity(nz * ny * nx);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                data.push((z * 100 + y * 10 + x) as f32 + 0.5);
            }
        }
    }
    Image::from_flat_on(
        data,
        [nz, ny, nx],
        Point::new([10.0, 20.0, 30.0]),
        Spacing::new([1.5, 2.0, 3.0]),
        Direction::identity(),
        backend,
    )
    .expect("synthetic image invariant")
}

/// A temp path unique across concurrently running test processes.
///
/// A timestamp alone is not enough: nextest runs test binaries in parallel,
/// clock granularity is coarse on some platforms, and two tests that land in
/// the same tick then read each other's file. The pid separates processes and
/// the counter separates calls within one.
fn unique_temp_path(stem: &str, extension: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id();
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{stem}_{pid}_{nanos:016x}_{seq}.{extension}"))
}

fn temp_mif_path() -> PathBuf {
    unique_temp_path("ritk_mif_test", "mif")
}

// ── Single-volume round-trip ────────────────────────────────────────────

#[test]
fn single_volume_round_trip_recovers_identical_voxels() {
    let backend = SequentialBackend;
    let image = make_test_image(&backend);

    let path = temp_mif_path();
    write_mif(&path, &image, &backend).expect("write .mif");

    let round_tripped: Image<f32, SequentialBackend, 3> =
        read_mif(&path, &backend, Exact).expect("read .mif");

    // Clean up.
    let _ = std::fs::remove_file(&path);

    assert_eq!(round_tripped.shape(), image.shape());
    assert_eq!(round_tripped.origin(), image.origin());

    let orig = image.data_cow_on(&backend);
    let rt = round_tripped.data_cow_on(&backend);
    for (i, (&a, &b)) in orig.iter().zip(rt.iter()).enumerate() {
        assert!((a - b).abs() < 1e-4, "voxel {i} differs: {a} vs {b}");
    }
}

#[test]
fn single_volume_round_trip_preserves_spatial_metadata() {
    let backend = SequentialBackend;

    let nz = 3;
    let ny = 5;
    let nx = 4;
    let data = vec![0.0f32; nz * ny * nx];

    let origin = Point::new([-20.0, 15.0, 5.0]);
    let spacing = Spacing::new([1.0, 2.0, 3.0]);
    let direction = Direction::identity();

    let image: Image<f32, SequentialBackend, 3> =
        Image::from_flat_on(data, [nz, ny, nx], origin, spacing, direction, &backend)
            .expect("synthetic image");

    let path = temp_mif_path();
    write_mif(&path, &image, &backend).expect("write .mif");
    let rt: Image<f32, SequentialBackend, 3> = read_mif(&path, &backend, Exact).expect("read .mif");
    let _ = std::fs::remove_file(&path);

    for axis in 0..3 {
        assert!(
            (rt.origin()[axis] - origin[axis]).abs() < 1e-3,
            "origin axis {axis} differs"
        );
    }
    for axis in 0..3 {
        assert!(
            (rt.spacing()[axis] - spacing[axis]).abs() < 1e-3,
            "spacing axis {axis} differs"
        );
    }
}

// ── Series round-trip ────────────────────────────────────────────────────

#[test]
fn series_round_trip_recovers_all_frames() {
    let backend = SequentialBackend;

    let nz = 3;
    let ny = 4;
    let nx = 2;
    let nframes = 5_usize;

    let mut volumes: Vec<Image<f32, SequentialBackend, 3>> = Vec::new();
    for frame in 0..nframes {
        let data: Vec<f32> = (0..(nz * ny * nx))
            .map(|i| (i + frame * 1000) as f32)
            .collect();
        let img = Image::from_flat_on(
            data,
            [nz, ny, nx],
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
            &backend,
        )
        .expect("synthetic image");
        volumes.push(img);
    }

    let path = temp_mif_path();
    write_mif_series(&path, &volumes, &backend).expect("write .mif series");
    let rt_volumes: Vec<Image<f32, SequentialBackend, 3>> =
        read_mif_series(&path, &backend, Exact).expect("read .mif series");
    let _ = std::fs::remove_file(&path);

    assert_eq!(rt_volumes.len(), nframes, "frame count mismatch");

    for fi in 0..nframes {
        let orig = volumes[fi].data_cow_on(&backend);
        let rt = rt_volumes[fi].data_cow_on(&backend);
        for (i, (&a, &b)) in orig.iter().zip(rt.iter()).enumerate() {
            assert!((a - b).abs() < 1e-4, "frame {fi} voxel {i}: {a} vs {b}");
        }
    }
}

// ── Error handling ───────────────────────────────────────────────────────

#[test]
fn single_volume_reader_rejects_multi_frame_file() {
    let backend = SequentialBackend;

    let nz = 2;
    let ny = 2;
    let nx = 2;
    let nframes = 3_usize;

    let mut volumes: Vec<Image<f32, SequentialBackend, 3>> = Vec::new();
    for frame in 0..nframes {
        let data: Vec<f32> = (0..(nz * ny * nx))
            .map(|i| (i + frame * 100) as f32)
            .collect();
        let img = Image::from_flat_on(
            data,
            [nz, ny, nx],
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
            &backend,
        )
        .expect("synthetic image");
        volumes.push(img);
    }

    let path = temp_mif_path();
    write_mif_series(&path, &volumes, &backend).expect("write .mif series");
    let result: Result<Image<f32, SequentialBackend, 3>, _> = read_mif(&path, &backend, Exact);
    let _ = std::fs::remove_file(&path);

    let msg = result.expect_err("the call must be rejected").to_string();
    assert!(
        msg.contains("3 frames"),
        "should mention frame count, got: {msg}"
    );
}

// ── Inline data offset ───────────────────────────────────────────────────

#[test]
fn a_file_laid_out_as_mrtrix_writes_it_reads_value_equal() {
    let backend = SequentialBackend;
    let values = [1.5_f32, -2.25, 3.0, 4.5, 5.75, -6.0, 7.0, 8.125];
    let payload: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let path = temp_mif_path();
    std::fs::write(&path, mrtrix_inline_file(&header_body("2 2 2"), &payload))
        .expect("write MRtrix-layout fixture");

    let image: Image<f32, SequentialBackend, 3> =
        read_mif(&path, &backend, Exact).expect("read .mif");
    let _ = std::fs::remove_file(&path);

    assert_eq!(image.data_slice().expect("host data"), values);
}

/// An inline offset inside the header is refused, and `file: . 0` is the
/// instance MRtrix rejects by name (`get_mrtrix_file_path`).
///
/// The offset counts from the start of the file, so a position the header
/// itself occupies cannot locate voxels. Reading it as a distance past the
/// header, as an earlier reader did, would accept both values here and decode
/// the 32-byte payload.
#[test]
fn an_inline_offset_inside_the_header_is_refused() {
    let backend = SequentialBackend;
    for offset in ["0", "10"] {
        let path = write_raw_mif(&header_with("2 2 2", &format!(". {offset}")), &[0_u8; 32]);
        let result: Result<Image<f32, SequentialBackend, 3>, _> = read_mif(&path, &backend, Exact);
        let _ = std::fs::remove_file(&path);

        let message = format!("{:#}", result.expect_err("an offset inside the header"));
        assert!(
            message.contains(&format!("offset {offset}, inside the")),
            "offset {offset} should be refused as lying inside the header, got: {message}"
        );
    }
}

/// A header without a `file:` line is refused: MRtrix requires the key
/// (`image_data.rst`, "file"), so no position for the voxels is defined.
#[test]
fn a_header_without_a_file_key_is_refused() {
    let backend = SequentialBackend;
    let path = write_raw_mif(&format!("{}END\n", header_body("2 2 2")), &[0_u8; 32]);
    let result: Result<Image<f32, SequentialBackend, 3>, _> = read_mif(&path, &backend, Exact);
    let _ = std::fs::remove_file(&path);

    let message = format!("{:#}", result.expect_err("a header without a file key"));
    assert!(message.contains("has no 'file' key"), "{message}");
}

/// A detached `file:` name without an offset reads its voxels from byte 0 of
/// that file, as MRtrix's `get_mrtrix_file_path` defaults the offset.
#[test]
fn a_detached_file_without_an_offset_reads_from_its_first_byte() {
    let backend = SequentialBackend;
    let values: [f32; 8] = [0.1, -0.0, 1.5, -2.25, 16_777_216.0, 3.0e-38, 7.0, -8.5];
    let data_path = unique_temp_path("ritk_mif_detached", "dat");
    let payload: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&data_path, &payload).expect("write detached payload");
    let name = data_path
        .file_name()
        .and_then(|name| name.to_str())
        .expect("a UTF-8 file name");
    let path = write_raw_mif(&header_with("2 2 2", name), &[]);

    let result: Result<Image<f32, SequentialBackend, 3>, _> = read_mif(&path, &backend, Exact);
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&data_path);

    let image = result.expect("a detached file with the default offset");
    let read: Vec<u32> = image
        .data_slice()
        .expect("host data")
        .iter()
        .map(|v| v.to_bits())
        .collect();
    assert_eq!(read, values.map(f32::to_bits));
}

/// A detached `file:` name is joined to the header's directory, so an
/// absolute name, or one climbing out through `..`, is refused before any file
/// is opened: `Path::join` would otherwise discard the directory or leave it.
#[test]
fn a_detached_file_outside_the_header_directory_is_refused() {
    let backend = SequentialBackend;
    let absolute = std::env::temp_dir().join("ritk_mif_outside.dat");
    let absolute = absolute.to_str().expect("a UTF-8 temp path").to_owned();
    for name in [
        "../ritk_mif_outside.dat",
        "sub/../../outside.dat",
        absolute.as_str(),
    ] {
        let path = write_raw_mif(&header_with("2 2 2", &format!("{name} 0")), &[]);
        let result: Result<Image<f32, SequentialBackend, 3>, _> = read_mif(&path, &backend, Exact);
        let _ = std::fs::remove_file(&path);

        let message = format!("{:#}", result.expect_err("a name outside the directory"));
        assert!(
            message.contains("must be a relative path without '..'"),
            "{name}: {message}"
        );
    }
}

/// An inline offset one byte past the end of the file is refused as such, and
/// an offset exactly at the end of the file, with voxels still owed, is a
/// truncated payload.
#[test]
fn an_inline_offset_past_the_end_of_the_file_is_refused() {
    const PAYLOAD_LEN: usize = 32;
    let backend = SequentialBackend;
    let body = header_body("2 2 2");
    let frame = "file: . ".len() + "\nEND\n".len();

    // The offset spells itself in the header, so find the one whose digits make
    // the file exactly one byte shorter than it.
    let one_past = (100..10_000_usize)
        .find(|offset| *offset == body.len() + frame + offset.to_string().len() + PAYLOAD_LEN + 1)
        .expect("an offset one past the end exists");
    let path = write_raw_mif(
        &header_with("2 2 2", &format!(". {one_past}")),
        &[0_u8; PAYLOAD_LEN],
    );
    let result: Result<Image<f32, SequentialBackend, 3>, _> = read_mif(&path, &backend, Exact);
    let _ = std::fs::remove_file(&path);
    let message = format!(
        "{:#}",
        result.expect_err("an offset past the end of the file")
    );
    assert!(
        message.contains(&format!(
            "offset {one_past}, past the end of the {}-byte file",
            one_past - 1
        )),
        "an offset one byte beyond the file should be refused as such, got: {message}"
    );

    let at_end = temp_mif_path();
    std::fs::write(&at_end, mrtrix_inline_file(&body, &[])).expect("write header-only fixture");
    let result: Result<Image<f32, SequentialBackend, 3>, _> = read_mif(&at_end, &backend, Exact);
    let _ = std::fs::remove_file(&at_end);
    let message = format!("{:#}", result.expect_err("no voxels follow the offset"));
    assert!(
        message.contains("payload is truncated"),
        "an offset at the end of the file should leave the payload truncated, got: {message}"
    );
}

// ── Malformed header contracts ──────────────────────────────────────────
//
// The reader takes header values that determine allocation and chunking. These
// pin the two that previously reached a `vec!` and a `chunks(0)` unguarded;
// before this crate had no truncation or corruption coverage at all.

/// Write a hand-built `.mif` and return its path.
fn write_raw_mif(body: &str, payload: &[u8]) -> PathBuf {
    let path = temp_mif_path();
    let mut bytes = body.as_bytes().to_vec();
    bytes.extend_from_slice(payload);
    std::fs::write(&path, &bytes).expect("write malformed .mif fixture");
    path
}

/// A minimal well-formed header up to, not including, its `file:` line, so
/// each test varies exactly one field.
fn header_body(dim: &str) -> String {
    format!("mrtrix image\ndim: {dim}\nvox: 1 1 1\nlayout: +0,+1,+2\ndatatype: Float32LE\n")
}

/// [`header_body`] closed by an explicit `file:` line and `END`.
fn header_with(dim: &str, file_key: &str) -> String {
    format!("{}file: {file_key}\nEND\n", header_body(dim))
}

/// An overstated `file:` offset must fail as truncation, not allocate for it.
///
/// The offset is a number on a header line, not a fact about the file. It
/// previously sized a zero-filled `Vec` directly, so `file: . 4000000000` in a
/// 300-byte file demanded 4 GB — and succeeded, which is silent exhaustion
/// rather than a crash. The offset counts from the start of the file, so the
/// fixture file is shorter than it by construction.
///
/// Threshold: the fixture is well under a kilobyte and decodes to nothing, so
/// an honest read touches kilobytes. 64 MiB sits far above that and two orders
/// of magnitude below the 4 GB the defect demands.
#[test]
fn an_overstated_file_offset_does_not_drive_allocation() {
    const PEAK_LIMIT: usize = 64 * 1024 * 1024;
    let backend = SequentialBackend;
    let path = write_raw_mif(&header_with("2 2 2", ". 4000000000"), &[0u8; 32]);

    let (result, peak) =
        peak_bytes_during(|| -> anyhow::Result<Image<f32, SequentialBackend, 3>> {
            read_mif(&path, &backend, Exact)
        });
    let _ = std::fs::remove_file(&path);

    let err = result.expect_err("a 32-byte payload cannot satisfy a 4 GB offset");
    let message = format!("{err:#}");
    assert!(
        message.contains("offset 4000000000") && message.contains("past the end"),
        "the error should name the offset that lies past the end of the file, got: {message}"
    );
    assert!(
        peak < PEAK_LIMIT,
        "reading a sub-kilobyte file peaked at {peak} live bytes, above {PEAK_LIMIT}; \
         the header offset is driving allocation"
    );
}

/// A zero extent on any axis is rejected before it reaches `chunks(0)`.
///
/// `Vec::chunks` panics on a zero chunk size, and `voxels_per_volume` is the
/// product of the spatial extents. The payload check cannot catch this: zero
/// voxels expects zero bytes, so a truncated file satisfies it.
#[test]
fn a_zero_extent_on_any_axis_is_rejected() {
    let backend = SequentialBackend;
    for (dim, axis) in [
        ("0 4 4", "x"),
        ("4 0 4", "y"),
        ("4 4 0", "z"),
        ("4 4 4 0", "frame"),
    ] {
        let path = temp_mif_path();
        std::fs::write(&path, mrtrix_inline_file(&header_body(dim), &[]))
            .expect("write zero-extent .mif fixture");
        let result: Result<Image<f32, SequentialBackend, 3>, _> = read_mif(&path, &backend, Exact);
        let _ = std::fs::remove_file(&path);

        let err = result.expect_err("a zero extent cannot describe a voxel grid");
        let message = format!("{err:#}");
        assert!(
            message.contains("extent of 0"),
            "dim {dim} should be rejected for its zero {axis} extent, got: {message}"
        );
    }
}
