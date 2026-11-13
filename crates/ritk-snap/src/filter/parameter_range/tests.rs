use super::{
    ParameterRange, ANISOTROPIC_DIFFUSION_CONDUCTANCE, ANISOTROPIC_DIFFUSION_ITERATIONS,
    ANISOTROPIC_DIFFUSION_TIME_STEP, CLAHE_CLIP_LIMIT, CLAHE_TILES_PER_AXIS,
    CONFIDENCE_CONNECTED_MAX_ITERATIONS, CONFIDENCE_CONNECTED_MULTIPLIER,
    CURVATURE_FLOW_ITERATIONS, CURVATURE_FLOW_TIME_STEP, GAUSSIAN_SIGMA_MM, HIST_EQ_BINS,
    MEDIAN_RADIUS, MULTI_OTSU_CLASSES, NEIGHBORHOOD_CONNECTED_RADIUS, UNSHARP_AMOUNT,
    UNSHARP_SIGMA_MM, UNSHARP_THRESHOLD,
};
use std::fmt::Debug;
use std::ops::Sub;

fn bounds<T: Copy + PartialOrd>(range: ParameterRange<T>) -> (T, T, T) {
    (range.min(), range.max(), range.step())
}

/// The bounds and step each smoothing and segmentation parameter had in the
/// filter panel's sliders (`ui/filter_panel/controls.rs` at 6d0f66501),
/// except the diffusion time step, whose panel bound 0.1667 exceeded the
/// Δt ≤ 1/6 that `FilterKind` documents.
#[test]
fn smoothing_and_segmentation_ranges_keep_the_panel_bounds() {
    assert_eq!(bounds(GAUSSIAN_SIGMA_MM), (0.1, 20.0, 0.1));
    assert_eq!(bounds(MEDIAN_RADIUS), (0, 10, 1));
    assert_eq!(bounds(CLAHE_TILES_PER_AXIS), (1, 32, 1));
    assert_eq!(bounds(CLAHE_CLIP_LIMIT), (1.0, 200.0, 1.0));
    assert_eq!(bounds(HIST_EQ_BINS), (2, 1024, 1));
    assert_eq!(bounds(UNSHARP_SIGMA_MM), (0.1, 10.0, 0.1));
    assert_eq!(bounds(UNSHARP_AMOUNT), (0.0, 5.0, 0.05));
    assert_eq!(bounds(UNSHARP_THRESHOLD), (0.0, 100.0, 0.5));
    assert_eq!(bounds(ANISOTROPIC_DIFFUSION_ITERATIONS), (1, 50, 1));
    assert_eq!(
        bounds(ANISOTROPIC_DIFFUSION_TIME_STEP),
        (0.01, 1.0 / 6.0, 0.005)
    );
    assert_eq!(bounds(ANISOTROPIC_DIFFUSION_CONDUCTANCE), (0.1, 100.0, 0.1));
    assert_eq!(bounds(MULTI_OTSU_CLASSES), (2, 8, 1));
    assert_eq!(bounds(CURVATURE_FLOW_ITERATIONS), (1, 50, 1));
    assert_eq!(bounds(CURVATURE_FLOW_TIME_STEP), (0.001, 0.166, 0.001));
    assert_eq!(bounds(CONFIDENCE_CONNECTED_MULTIPLIER), (0.5, 10.0, 0.1));
    assert_eq!(bounds(CONFIDENCE_CONNECTED_MAX_ITERATIONS), (1, 100, 1));
    assert_eq!(bounds(NEIGHBORHOOD_CONNECTED_RADIUS), (0, 5, 1));
}

fn assert_step_fits<T>(range: ParameterRange<T>)
where
    T: Copy + PartialOrd + Default + Sub<Output = T> + Debug,
{
    assert!(range.min() < range.max(), "empty range {range:?}");
    assert!(
        range.step() > T::default(),
        "non-positive step in {range:?}"
    );
    assert!(
        range.step() <= range.max() - range.min(),
        "step wider than {range:?}"
    );
}

#[test]
fn every_range_is_ordered_with_a_step_inside_it() {
    for range in [
        GAUSSIAN_SIGMA_MM,
        CLAHE_CLIP_LIMIT,
        UNSHARP_SIGMA_MM,
        UNSHARP_AMOUNT,
        UNSHARP_THRESHOLD,
        ANISOTROPIC_DIFFUSION_TIME_STEP,
        ANISOTROPIC_DIFFUSION_CONDUCTANCE,
        CURVATURE_FLOW_TIME_STEP,
        CONFIDENCE_CONNECTED_MULTIPLIER,
    ] {
        assert_step_fits(range);
    }
    for range in [
        MEDIAN_RADIUS,
        CLAHE_TILES_PER_AXIS,
        HIST_EQ_BINS,
        NEIGHBORHOOD_CONNECTED_RADIUS,
    ] {
        assert_step_fits(range);
    }
    for range in [
        ANISOTROPIC_DIFFUSION_ITERATIONS,
        MULTI_OTSU_CLASSES,
        CURVATURE_FLOW_ITERATIONS,
        CONFIDENCE_CONNECTED_MAX_ITERATIONS,
    ] {
        assert_step_fits(range);
    }
}

/// `below < min <= inside <= max < above`.
fn assert_clamps<T>(range: ParameterRange<T>, below: T, inside: T, above: T)
where
    T: Copy + PartialOrd + Debug,
{
    assert_eq!(range.clamp(below), range.min());
    assert_eq!(range.clamp(inside), inside);
    assert_eq!(range.clamp(above), range.max());
    assert_eq!(range.clamp(range.min()), range.min());
    assert_eq!(range.clamp(range.max()), range.max());
    assert!(!range.contains(below));
    assert!(range.contains(inside));
    assert!(!range.contains(above));
    assert!(range.contains(range.min()));
    assert!(range.contains(range.max()));
}

#[test]
fn clamp_moves_values_outside_the_range_to_the_nearest_bound() {
    assert_clamps(GAUSSIAN_SIGMA_MM, 0.0, 3.5, 25.0);
    assert_clamps(
        GAUSSIAN_SIGMA_MM,
        0.1_f32.next_down(),
        0.1,
        20.0_f32.next_up(),
    );
    // usize has no value below the median radius minimum of 0.
    assert_eq!(MEDIAN_RADIUS.clamp(4), 4);
    assert_eq!(MEDIAN_RADIUS.clamp(11), 10);
    assert!(!MEDIAN_RADIUS.contains(11));
    assert_clamps(CLAHE_TILES_PER_AXIS, 0, 8, 33);
    assert_clamps(HIST_EQ_BINS, 1, 256, 1025);
    assert_clamps(MULTI_OTSU_CLASSES, 1, 3, 9);
    assert_clamps(CONFIDENCE_CONNECTED_MAX_ITERATIONS, 0, 15, 101);
}

#[test]
fn clamp_maps_nan_to_the_minimum_and_contains_rejects_it() {
    for range in [
        GAUSSIAN_SIGMA_MM,
        UNSHARP_THRESHOLD,
        CURVATURE_FLOW_TIME_STEP,
    ] {
        assert_eq!(range.clamp(f32::NAN), range.min());
        assert!(!range.contains(f32::NAN));
    }
}

#[test]
fn diffusion_time_step_rejects_the_rounded_panel_bound() {
    assert!(!ANISOTROPIC_DIFFUSION_TIME_STEP.contains(0.1667));
    assert_eq!(ANISOTROPIC_DIFFUSION_TIME_STEP.clamp(0.1667), 1.0 / 6.0);
}

#[test]
fn clamp_maps_infinities_to_the_bounds() {
    assert_eq!(CLAHE_CLIP_LIMIT.clamp(f32::INFINITY), 200.0);
    assert_eq!(CLAHE_CLIP_LIMIT.clamp(f32::NEG_INFINITY), 1.0);
}
