//! Differential test for [`vector_confidence_connected`] against SimpleITK.
//!
//! Uses a deterministic structured 2-channel input (reproducible in Python) with
//! a clear channel-0 blob; the expected region mask is captured verbatim from
//! `sitk.VectorConfidenceConnected` (default geometry, multiplier 3.0).

use super::vector_confidence_connected;

/// Build the deterministic 10×10 two-channel scene: structured background plus a
/// 5×5 channel-0 blob over rows/cols 3..8.
fn scene() -> (Vec<Vec<f64>>, [usize; 3]) {
    let (h, w) = (10usize, 10usize);
    let mut ch0 = vec![0.0_f64; h * w];
    let mut ch1 = vec![0.0_f64; h * w];
    for y in 0..h {
        for x in 0..w {
            ch0[y * w + x] = 0.1 * (((x + y) % 3) as f64 - 1.0);
            ch1[y * w + x] = 0.1 * (((x * y) % 5) as f64 - 2.0);
        }
    }
    for y in 3..8 {
        for x in 3..8 {
            ch0[y * w + x] += 3.0;
        }
    }
    (vec![ch0, ch1], [1, h, w])
}

#[test]
fn matches_sitk_blob_region() {
    let (channels, dims) = scene();
    let seeds = [[0usize, 5, 5], [0, 4, 6]];
    let out = vector_confidence_connected(&channels, dims, &seeds, 3.0, 4, 1, 1.0)
        .expect("valid vector scene must segment");

    // sitk reference: the 5×5 blob (rows/cols 3..8) is the region.
    let mut expect = vec![0.0_f32; 100];
    for y in 3..8 {
        for x in 3..8 {
            expect[y * 10 + x] = 1.0;
        }
    }
    assert_eq!(out, expect, "region differs from sitk");
}

#[test]
fn no_seeds_yields_empty() {
    let (channels, dims) = scene();
    let out = vector_confidence_connected(&channels, dims, &[], 2.5, 4, 1, 1.0)
        .expect("valid vector scene with no seeds must return an empty mask");
    assert!(out.iter().all(|&v| v == 0.0));
}

#[test]
fn out_of_bounds_seed_ignored() {
    let (channels, dims) = scene();
    // Only the in-bounds seed contributes; region is still the blob.
    let out =
        vector_confidence_connected(&channels, dims, &[[0, 5, 5], [0, 99, 99]], 3.0, 4, 1, 1.0)
            .expect("valid vector scene with an out-of-bounds seed must segment");
    assert_eq!(out[5 * 10 + 5], 1.0);
    assert_eq!(out[0], 0.0);
}

#[test]
fn channel_length_mismatch_returns_error() {
    let channels = vec![vec![0.0_f64; 4], vec![1.0_f64; 3]];
    let err = vector_confidence_connected(&channels, [1, 2, 2], &[[0, 0, 0]], 2.5, 1, 0, 1.0)
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("channel 1 length 3") && msg.contains("voxel count 4"),
        "error must identify mismatched channel length: {}",
        msg
    );
}

#[test]
fn overflowing_dims_return_error() {
    let channels = vec![Vec::<f64>::new()];
    let err = vector_confidence_connected(
        &channels,
        [usize::MAX, usize::MAX, usize::MAX],
        &[[0, 0, 0]],
        2.5,
        1,
        0,
        1.0,
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("voxel count overflows"),
        "error must identify voxel-count overflow: {}",
        msg
    );
}
