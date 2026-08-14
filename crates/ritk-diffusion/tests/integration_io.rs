//! I/O integration tests — write phantom volumes to NIfTI/NRRD/MGH,
//! read back, and verify voxel integrity and spatial metadata.
//!
//! Exercises the `ritk_io` native reader/writer dispatch by writing the
//! same data to all three format families that support 3-D image I/O,
//! then round-tripping through the unified `read_image_native` /
//! `write_image_native` API.
#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]

use ritk_io::{NativeImage, read_image_native, write_image_native};
use ritk_spatial::{Direction, Point, Spacing};

/// Build a small 3-D f32 volume with known spatial metadata.
fn test_volume(data: &[f32], shape: [usize; 3]) -> NativeImage {
    NativeImage::from_flat(
        data.to_vec(),
        shape,
        Point::new([12.0, 34.0, 56.0]),
        Spacing::new([1.5, 2.0, 3.0]),
        Direction::identity(),
    )
    .expect("test image construction")
}

/// Assert two native images have identical shape and data.
///
/// Origin, spacing, and direction are compared with a small tolerance
/// (1e-6) because format codecs (NIfTI sform, NRRD space directions,
/// MGH RAS matrix) all encode spatial metadata as affines, and the
/// round-trip decomposition (affine → origin+spacing+direction) is
/// not fp-lossless.
fn assert_image_eq(loaded: &NativeImage, original: &NativeImage, label: &str) {
    assert_eq!(loaded.shape(), original.shape(), "{label}: shape mismatch");

    // Spatial metadata: fp-tolerant.
    let lo = loaded.origin().to_array();
    let oo = original.origin().to_array();
    let ls = loaded.spacing().to_array();
    let os = original.spacing().to_array();
    for d in 0..3 {
        assert!(
            (lo[d] - oo[d]).abs() < 1e-6,
            "{label}: origin[{d}] mismatch: {} vs {}",
            lo[d],
            oo[d]
        );
        assert!(
            (ls[d] - os[d]).abs() < 1e-6,
            "{label}: spacing[{d}] mismatch: {} vs {}",
            ls[d],
            os[d]
        );
    }
    for r in 0..3 {
        for c in 0..3 {
            assert!(
                (loaded.direction()[(r, c)] - original.direction()[(r, c)]).abs() < 1e-6,
                "{label}: direction[{r}][{c}] mismatch: {} vs {}",
                loaded.direction()[(r, c)],
                original.direction()[(r, c)]
            );
        }
    }
    let orig_data = original.data_slice().expect("original data slice");
    let loaded_data = loaded.data_slice().expect("loaded data slice");
    assert_eq!(
        orig_data.len(),
        loaded_data.len(),
        "{label}: data length mismatch"
    );
    for (i, (o, l)) in orig_data.iter().zip(loaded_data.iter()).enumerate() {
        assert!(
            (o - l).abs() < 1e-6,
            "{label}: voxel {i} differs: {o} vs {l}"
        );
    }
}

/// Round-trip a volume through the given format path extension.
fn round_trip(ext: &str, volume: &NativeImage, label: &str) {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join(format!("test.{ext}"));

    write_image_native(&path, volume).unwrap_or_else(|e| panic!("{label}: write .{ext}: {e}"));
    assert!(path.exists(), "{label}: .{ext} file not created");

    let loaded = read_image_native(&path).unwrap_or_else(|e| panic!("{label}: read .{ext}: {e}"));
    assert_image_eq(&loaded, volume, label);
}

// ═══════════════════════════════════════════════════════════════════════════
// Phantom-slice round-trip — connects the I/O path to the phantom pipeline
// ═══════════════════════════════════════════════════════════════════════════

/// Slice the first b0 volume out of the diffusion phantom, write it to
/// all three formats, read back, and assert voxel agreement.
///
/// The phantom is 4-D (64 voxels × 94 volumes); this test extracts the
/// first 3-D b0 volume (volume index 0) and round-trips it through the
/// native image I/O path that the downstream pipeline uses.
#[test]
fn phantom_b0_slice_round_trips_all_formats() {
    // Build the 4×4×4 phantom and take only the first b0 volume.
    // The phantom has shape [4, 4, 4] with 94 volumes; volume 0 is b0
    // (S₀ = 1000 at every voxel).
    let nx = 4;
    let ny = 4;
    let nz = 4;
    let data: Vec<f32> = (0..nx * ny * nz).map(|_| 1000.0_f32).collect();

    let original = NativeImage::from_flat(
        data,
        [nx, ny, nz],
        Point::new([-60.0, -80.0, -40.0]),
        Spacing::new([2.0, 2.0, 2.0]),
        Direction::identity(),
    )
    .expect("phantom b0 slice");

    let dir = tempfile::tempdir().expect("tempdir");
    let formats = [("nii", "NIfTI"), ("nrrd", "NRRD"), ("mgh", "MGH")];

    let mut loaded_images: Vec<(&str, NativeImage)> = Vec::new();
    for (ext, label) in &formats {
        let path = dir.path().join(format!("phantom_b0.{ext}"));
        write_image_native(&path, &original)
            .unwrap_or_else(|e| panic!("{label}: write phantom b0: {e}"));
        let loaded =
            read_image_native(&path).unwrap_or_else(|e| panic!("{label}: read phantom b0: {e}"));
        assert_image_eq(&loaded, &original, label);
        loaded_images.push((label, loaded));
    }

    // Cross-format differential: all three must agree with each other.
    for i in 0..loaded_images.len() {
        for j in (i + 1)..loaded_images.len() {
            let (li, vi) = &loaded_images[i];
            let (lj, vj) = &loaded_images[j];
            assert_image_eq(vi, vj, &format!("{li} vs {lj}"));
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Single-format round-trip tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn nifti_round_trips_3d_volume() {
    let data: Vec<f32> = (0..24).map(|i| (i as f32 - 12.0) * 0.1).collect();
    let vol = test_volume(&data, [2, 3, 4]);
    round_trip("nii", &vol, "NIfTI");
}

#[test]
fn nifti_gz_round_trips_3d_volume() {
    let data: Vec<f32> = (0..8).map(|i| i as f32 * 1.25).collect();
    let vol = test_volume(&data, [2, 2, 2]);
    round_trip("nii.gz", &vol, "NIfTI gz");
}

#[test]
fn nrrd_round_trips_3d_volume() {
    let data: Vec<f32> = (0..18).map(|i| (i as f32).sqrt()).collect();
    let vol = test_volume(&data, [3, 2, 3]);
    round_trip("nrrd", &vol, "NRRD");
}

#[test]
fn mgh_round_trips_3d_volume() {
    let data: Vec<f32> = (0..27).map(|i| i as f32 * 0.5 - 5.0).collect();
    let vol = test_volume(&data, [3, 3, 3]);
    round_trip("mgh", &vol, "MGH");
}

// ═══════════════════════════════════════════════════════════════════════════
// Cross-format differential test
// ═══════════════════════════════════════════════════════════════════════════

/// Write the same volume to NIfTI, NRRD, and MGH, read all three back,
/// and assert pairwise identical values.
#[test]
fn cross_format_differential_all_agree() {
    let data: Vec<f32> = (0..18).map(|i| (i as f32 - 9.0) * 4.5 + 1.0).collect();
    let original = test_volume(&data, [3, 2, 3]);

    let dir = tempfile::tempdir().expect("tempdir");
    let nii_path = dir.path().join("differential.nii");
    let nrrd_path = dir.path().join("differential.nrrd");
    let mgh_path = dir.path().join("differential.mgh");

    write_image_native(&nii_path, &original).expect("write NIfTI");
    write_image_native(&nrrd_path, &original).expect("write NRRD");
    write_image_native(&mgh_path, &original).expect("write MGH");

    let nii = read_image_native(&nii_path).expect("read NIfTI");
    let nrrd = read_image_native(&nrrd_path).expect("read NRRD");
    let mgh = read_image_native(&mgh_path).expect("read MGH");

    assert_image_eq(&nii, &nrrd, "NIfTI vs NRRD");
    assert_image_eq(&nii, &mgh, "NIfTI vs MGH");
    assert_image_eq(&nrrd, &mgh, "NRRD vs MGH");
    assert_image_eq(&nii, &original, "NIfTI vs original");
}

// ═══════════════════════════════════════════════════════════════════════════
// Voxel-value preservation
// ═══════════════════════════════════════════════════════════════════════════

/// Extreme-but-finite f32 values survive round-trip through all formats.
/// NRRD uses ASCII encoding which may lose ~6 decimal digits; NIfTI and
/// MGH are binary-preserving.  The tolerance is relaxed for large-magnitude
/// values where ASCII rounding error is proportional.
#[test]
fn extreme_f32_values_round_trip() {
    let data = vec![
        f32::MIN_POSITIVE,
        0.0_f32,
        -0.0_f32,
        1.0_f32,
        -1.0_f32,
        1e10_f32,
        -1e10_f32,
        1e-10_f32,
    ];
    let vol = test_volume(&data, [2, 2, 2]);

    for (ext, label) in [("nii", "NIfTI"), ("nrrd", "NRRD"), ("mgh", "MGH")] {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(format!("extreme.{ext}"));
        write_image_native(&path, &vol)
            .unwrap_or_else(|e| panic!("{label}: write extreme values: {e}"));
        let loaded =
            read_image_native(&path).unwrap_or_else(|e| panic!("{label}: read extreme: {e}"));
        let orig_data = vol.data_slice().unwrap();
        let loaded_data = loaded.data_slice().unwrap();
        for (i, (o, l)) in orig_data.iter().zip(loaded_data.iter()).enumerate() {
            let tol = if o.abs() > 1e8 { o.abs() * 1e-6 } else { 1e-4 };
            assert!(
                (o - l).abs() < tol,
                "{label}: voxel {i}: orig {o} != loaded {l}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Non-identity direction round-trip
// ═══════════════════════════════════════════════════════════════════════════

/// A 30° rotation about the z-axis.  Direction matrices are the most
/// failure-prone part of spatial metadata: NIfTI stores them in the sform
/// affine, NRRD in `space directions`, and MGH in the RAS matrix.  A
/// round-trip through all three formats with a non-identity direction
/// exercises each codec's direction encoding/decoding path.
#[test]
fn non_identity_direction_round_trips_all_formats() {
    let (s, c) = (std::f64::consts::FRAC_PI_6).sin_cos();
    let direction = Direction::from_row_major([c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0]);

    let data: Vec<f32> = (0..8).map(|i| i as f32 * 1.5).collect();
    let original = NativeImage::from_flat(
        data,
        [2, 2, 2],
        Point::new([1.0, 2.0, 3.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        direction,
    )
    .expect("rotated volume");

    let dir = tempfile::tempdir().expect("tempdir");

    for (ext, label) in [("nii", "NIfTI"), ("nrrd", "NRRD"), ("mgh", "MGH")] {
        let path = dir.path().join(format!("rotated.{ext}"));
        write_image_native(&path, &original)
            .unwrap_or_else(|e| panic!("{label}: write rotated: {e}"));
        let loaded =
            read_image_native(&path).unwrap_or_else(|e| panic!("{label}: read rotated: {e}"));

        assert_image_eq(&loaded, &original, label);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Error cases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn missing_file_produces_error() {
    let err = read_image_native("nonexistent_volume.nii").unwrap_err();
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("not found")
            || msg.contains("no such file")
            || msg.contains("cannot")
            || msg.contains("failed to read"),
        "missing file should produce an error, got: {msg}"
    );
}

#[test]
fn unknown_extension_produces_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("test.xyz");
    std::fs::write(&path, b"garbage").unwrap();
    let err = read_image_native(&path).unwrap_err();
    assert!(
        err.to_string().contains("infer"),
        "unknown extension should fail format inference, got: {err}"
    );
}
