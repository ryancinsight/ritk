//! Integration tests for the `.mif` reader and writer.
//!
//! These tests round-trip images through the writer and reader to verify
//! voxel fidelity, spatial metadata preservation, and frame handling.

use std::path::PathBuf;

use coeus_core::SequentialBackend;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};

use crate::{read_mif, read_mif_series, write_mif, write_mif_series};

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

fn temp_mif_path() -> PathBuf {
    let dir = std::env::temp_dir();
    let suffix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    dir.join(format!("ritk_mif_test_{suffix:016x}.mif"))
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

    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
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
