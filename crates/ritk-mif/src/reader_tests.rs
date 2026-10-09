//! Integration tests for the `.mif` reader and writer.
//!
//! These tests round-trip images through the writer and reader to verify
//! voxel fidelity, spatial metadata preservation, and frame handling.

use std::path::PathBuf;

use coeus_core::SequentialBackend;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};

use ritk_core::alloc_probe::{peak_bytes_during, PeakTrackingAllocator};

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
        read_mif(&path, &backend).expect("read .mif");

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
    let rt: Image<f32, SequentialBackend, 3> = read_mif(&path, &backend).expect("read .mif");
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

/// A transform-less `.mif` reverses its `vox:` sizes into RITK axis order.
///
/// `vox:` is X, Y, Z (MRtrix3), and the writer emits it as
/// `[Δcol, Δrow, Δdepth]` because file X is the column axis. A header with no
/// `transform:` block — legal, and what most third-party writers produce —
/// must therefore land those sizes in RITK `[Δdepth, Δrow, Δcol]`, the order
/// `decompose_transform_affine` derives from a transform block. Reading them
/// index-for-index transposes the spacing, and the round-trip suite cannot see
/// it: `write_mif` always emits a `transform:`, so this branch is reached only
/// by files this crate did not write.
#[test]
fn a_transform_less_header_reverses_vox_sizes_into_ritk_axis_order() {
    let backend = SequentialBackend;
    // File X = 0.5, Y = 0.75, Z = 1.25: non-isotropic and not palindromic, so
    // a transposition cannot cancel itself out.
    let body = "mrtrix image\ndim: 4 3 2\nvox: 0.5 0.75 1.25\nlayout: +0,+1,+2\n\
                datatype: Float32LE\nfile: . 0\nEND\n";
    let payload = vec![0u8; 4 * 3 * 2 * 4];
    let path = write_raw_mif(body, &payload);

    let image: Image<f32, SequentialBackend, 3> = read_mif(&path, &backend).expect("read .mif");
    let _ = std::fs::remove_file(&path);

    assert_eq!(
        image.shape(),
        [2, 3, 4],
        "the file's 4 x 3 x 2 extent is [depth, row, col] = [2, 3, 4]"
    );

    let spacing = image.spacing();
    assert!(
        (spacing[0] - 1.25).abs() < 1e-6,
        "spacing[0] is Δdepth and must be the file Z size 1.25, got {}",
        spacing[0]
    );
    assert!(
        (spacing[1] - 0.75).abs() < 1e-6,
        "spacing[1] is Δrow and must be the file Y size 0.75, got {}",
        spacing[1]
    );
    assert!(
        (spacing[2] - 0.5).abs() < 1e-6,
        "spacing[2] is Δcol and must be the file X size 0.5, got {}",
        spacing[2]
    );
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
        read_mif_series(&path, &backend).expect("read .mif series");
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
    let result: Result<Image<f32, SequentialBackend, 3>, _> = read_mif(&path, &backend);
    let _ = std::fs::remove_file(&path);

    let msg = result.expect_err("the call must be rejected").to_string();
    assert!(
        msg.contains("3 frames"),
        "should mention frame count, got: {msg}"
    );
}

// ── Detached data file ───────────────────────────────────────────────────

#[test]
fn inline_file_dot_zero_is_read_correctly() {
    let backend = SequentialBackend;
    let image = make_test_image(&backend);

    let path = temp_mif_path();
    write_mif(&path, &image, &backend).expect("write .mif");

    let rt: Image<f32, SequentialBackend, 3> = read_mif(&path, &backend).expect("read .mif");
    let _ = std::fs::remove_file(&path);

    assert_eq!(rt.shape(), image.shape());
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

/// A minimal well-formed header, so each test varies exactly one field.
fn header_with(dim: &str, file_key: &str) -> String {
    format!(
        "mrtrix image\ndim: {dim}\nvox: 1 1 1\nlayout: +0,+1,+2\ndatatype: Float32LE\n\
         file: {file_key}\nEND\n"
    )
}

/// An overstated `file:` offset must fail as truncation, not allocate for it.
///
/// The offset is a number on a header line, not a fact about the file. It
/// previously sized a zero-filled `Vec` directly, so `file: . 4000000000` in a
/// 300-byte file demanded 4 GB — and succeeded, which is silent exhaustion
/// rather than a crash.
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
            read_mif(&path, &backend)
        });
    let _ = std::fs::remove_file(&path);

    let err = result.expect_err("a 32-byte payload cannot satisfy a 4 GB offset");
    let message = format!("{err:#}");
    assert!(
        message.contains("offset"),
        "the error should name the offset that could not be satisfied, got: {message}"
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
        let path = write_raw_mif(&header_with(dim, ". 0"), &[]);
        let result: Result<Image<f32, SequentialBackend, 3>, _> = read_mif(&path, &backend);
        let _ = std::fs::remove_file(&path);

        let err = result.expect_err("a zero extent cannot describe a voxel grid");
        let message = format!("{err:#}");
        assert!(
            message.contains("extent of 0"),
            "dim {dim} should be rejected for its zero {axis} extent, got: {message}"
        );
    }
}
