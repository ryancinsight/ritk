use super::*;
use coeus_core::SequentialBackend;
use ritk_image::test_support::make_image;

type TestBackend = SequentialBackend;

// ── Positive tests ────────────────────────────────────────────────────────

#[test]
fn histogram_3d_uniform_distribution() {
    // 8 voxels, range [0, 8), 8 bins → exactly one per bin
    let img: Image<f32, TestBackend, 3> =
        make_image(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [2, 2, 2]);
    let h = histogram(&img, 0.0, 8.0, 8);
    assert_eq!(h.counts, vec![1, 1, 1, 1, 1, 1, 1, 1]);
    assert_eq!(h.total(), 8);
}

#[test]
fn histogram_3d_last_bin_inclusive_of_max() {
    // Two voxels at v=7.0, range [0,7], 7 bins
    // Bin edges: [0,1), [1,2), [2,3), [3,4), [4,5), [5,6), [6,7]
    let img: Image<f32, TestBackend, 3> =
        make_image(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [2, 2, 2]);
    let h = histogram(&img, 0.0, 7.0, 7);
    assert_eq!(h.counts, vec![1, 1, 1, 1, 1, 1, 2]);
}

#[test]
fn histogram_3d_single_bin_collects_all_in_range() {
    // 1 bin: [0, 10] → all 8 in-range voxels go into bin 0
    let img: Image<f32, TestBackend, 3> =
        make_image(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [2, 2, 2]);
    let h = histogram(&img, 0.0, 10.0, 1);
    assert_eq!(h.counts, vec![8]);
}

#[test]
fn histogram_3d_values_outside_range_excluded() {
    // Range [0, 5] excludes 5.0+ (last bin inclusive of 5.0 only).
    // v=5 → in (last) bin; v=6,7 → excluded.
    let img: Image<f32, TestBackend, 3> =
        make_image(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [2, 2, 2]);
    let h = histogram(&img, 0.0, 5.0, 5);
    // Bin 0 [0,1): {0.0} → 1
    // Bin 1 [1,2): {1.0} → 1
    // Bin 2 [2,3): {2.0} → 1
    // Bin 3 [3,4): {3.0} → 1
    // Bin 4 [4,5]: {4.0, 5.0} → 2
    // 6.0, 7.0 excluded
    assert_eq!(h.counts, vec![1, 1, 1, 1, 2]);
    assert_eq!(h.total(), 6);
}

#[test]
fn histogram_1d_constant_lands_in_first_bin() {
    // All values = 5.0; range [0, 10], 5 bins. v=5.0 → bin 2.
    let img: Image<f32, TestBackend, 1> = make_image(vec![5.0; 10], [10]);
    let h = histogram(&img, 0.0, 10.0, 5);
    assert_eq!(h.counts, vec![0, 0, 10, 0, 0]);
}

#[test]
fn histogram_3d_constant_at_min_lands_in_bin_zero() {
    // v == min → bin 0
    let img: Image<f32, TestBackend, 3> = make_image(vec![3.0; 8], [2, 2, 2]);
    let h = histogram(&img, 3.0, 5.0, 4);
    // Δw = 0.5, bin 0 = [3.0, 3.5) ... but v=3.0 hits bin 0
    assert_eq!(h.counts[0], 8);
    assert_eq!(h.total(), 8);
}

#[test]
fn histogram_3d_constant_at_max_lands_in_last_bin() {
    // v == max → last bin (inclusive convention)
    let img: Image<f32, TestBackend, 3> = make_image(vec![7.0; 8], [2, 2, 2]);
    let h = histogram(&img, 0.0, 7.0, 7);
    assert_eq!(h.counts[6], 8);
    assert_eq!(h.total(), 8);
}

#[test]
fn histogram_3d_negative_range() {
    // Range [-10, 0], 5 bins
    let img: Image<f32, TestBackend, 3> = make_image(
        vec![-10.0, -7.5, -5.0, -2.5, 0.0, -9.0, -1.0, -100.0],
        [2, 2, 2],
    );
    // -100.0 is excluded (below min)
    let h = histogram(&img, -10.0, 0.0, 5);
    // Bin 0 [-10,-8): {-10.0, -9.0} → 2
    // Bin 1 [-8, -6): {-7.5} → 1
    // Bin 2 [-6, -4): {-5.0} → 1
    // Bin 3 [-4, -2): {-2.5} → 1
    // Bin 4 [-2,  0]: {-1.0, 0.0} → 2
    assert_eq!(h.counts, vec![2, 1, 1, 1, 2]);
    assert_eq!(h.total(), 7);
}

// ── Properties ────────────────────────────────────────────────────────────

#[test]
fn histogram_bin_width_is_correct() {
    let img: Image<f32, TestBackend, 3> = make_image(vec![0.0; 1], [1, 1, 1]);
    let h = histogram(&img, 0.0, 10.0, 4);
    assert!((h.bin_width() - 2.5).abs() < 1e-6);
}

#[test]
fn histogram_total_equals_in_range_voxel_count() {
    let img: Image<f32, TestBackend, 3> =
        make_image(vec![0.0, 1.0, 2.0, 100.0, -100.0, 5.0, 6.0, 7.0], [2, 2, 2]);
    // Range [0, 10]: 100.0 and -100.0 excluded → 6 in-range
    let h = histogram(&img, 0.0, 10.0, 10);
    assert_eq!(h.total(), 6);
}

#[test]
fn histogram_values_outside_range_yield_zero_counts() {
    // All voxels = 20.0, range [0, 10] → all excluded.
    let img: Image<f32, TestBackend, 3> = make_image(vec![20.0_f32; 8], [2, 2, 2]);
    let h = histogram(&img, 0.0, 10.0, 5);
    assert_eq!(h.counts, vec![0, 0, 0, 0, 0]);
    assert_eq!(h.total(), 0);
}

// ── Negative / boundary ───────────────────────────────────────────────────

#[test]
#[should_panic(expected = "bins must be ≥ 1")]
fn histogram_zero_bins_panics() {
    let img: Image<f32, TestBackend, 3> = make_image(vec![1.0; 1], [1, 1, 1]);
    let _ = histogram(&img, 0.0, 10.0, 0);
}

#[test]
#[should_panic(expected = "min must be strictly less than max")]
fn histogram_min_equal_max_panics() {
    let img: Image<f32, TestBackend, 3> = make_image(vec![1.0; 1], [1, 1, 1]);
    let _ = histogram(&img, 5.0, 5.0, 4);
}

#[test]
#[should_panic(expected = "min must be strictly less than max")]
fn histogram_min_greater_than_max_panics() {
    let img: Image<f32, TestBackend, 3> = make_image(vec![1.0; 1], [1, 1, 1]);
    let _ = histogram(&img, 10.0, 0.0, 4);
}

// ── D-type genericity ─────────────────────────────────────────────────────

#[test]
fn histogram_works_on_1d_image() {
    let img: Image<f32, TestBackend, 1> = make_image(vec![0.5, 1.5, 2.5, 3.5, 4.5], [5]);
    // Range [0, 5], 5 bins, Δw=1
    let h = histogram(&img, 0.0, 5.0, 5);
    // Each value lands in its own bin; 4.5 is the last bin's interior.
    assert_eq!(h.counts, vec![1, 1, 1, 1, 1]);
}
