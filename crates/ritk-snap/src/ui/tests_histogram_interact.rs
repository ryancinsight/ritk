use super::*;

// ── x_to_intensity ──────────────────────────────────────────────────────────

/// Left edge of canvas maps to hist_min.
#[test]
fn x_to_intensity_left_edge_returns_hist_min() {
    let v = x_to_intensity(10.0, -1000.0, 3000.0, 10.0, 210.0);
    assert_eq!(v, -1000.0);
}

/// Right edge of canvas maps to hist_max.
#[test]
fn x_to_intensity_right_edge_returns_hist_max() {
    let v = x_to_intensity(210.0, -1000.0, 3000.0, 10.0, 210.0);
    assert_eq!(v, 3000.0);
}

/// Midpoint of canvas maps to midpoint of intensity range.
///
/// Analytical: t = 0.5, v = -1000 + 0.5 × 4000 = 1000.
#[test]
fn x_to_intensity_midpoint_maps_analytically() {
    let v = x_to_intensity(110.0, -1000.0, 3000.0, 10.0, 210.0);
    let expected = -1000.0_f32 + 0.5 * 4000.0_f32;
    assert!((v - expected).abs() < 1e-3, "got {v}, expected {expected}");
}

/// x below x_left clamps to hist_min.
#[test]
fn x_to_intensity_below_left_clamps_to_min() {
    let v = x_to_intensity(0.0, -1000.0, 3000.0, 10.0, 210.0);
    assert_eq!(v, -1000.0);
}

/// x above x_right clamps to hist_max.
#[test]
fn x_to_intensity_above_right_clamps_to_max() {
    let v = x_to_intensity(999.0, -1000.0, 3000.0, 10.0, 210.0);
    assert_eq!(v, 3000.0);
}

/// Degenerate canvas (x_left == x_right) returns hist_min.
#[test]
fn x_to_intensity_degenerate_canvas_returns_min() {
    let v = x_to_intensity(50.0, -500.0, 500.0, 50.0, 50.0);
    assert_eq!(v, -500.0);
}

/// Degenerate intensity span (min == max) returns hist_min.
#[test]
fn x_to_intensity_degenerate_span_returns_min() {
    let v = x_to_intensity(100.0, 500.0, 500.0, 0.0, 200.0);
    assert_eq!(v, 500.0);
}

// ── wl_from_histogram_drag ──────────────────────────────────────────────────

/// Zero drag delta leaves center and width unchanged.
#[test]
fn wl_from_drag_zero_delta_identity() {
    let (c, w) = wl_from_histogram_drag(
        0.0,
        0.0,
        HistogramCanvasGeometry {
            canvas_width: 200.0,
            canvas_height: 80.0,
            hist_min: -1000.0,
            hist_max: 3000.0,
        },
        40.0,
        400.0,
    );
    assert_eq!(c, 40.0);
    assert_eq!(w, 400.0);
}

/// Rightward drag (dx = canvas_width/2) shifts center by span/2.
///
/// Analytical: Δcenter = (100/200) × 4000 = 2000; new center = 40 + 2000 = 2040.
#[test]
fn wl_from_drag_rightward_shifts_center() {
    let (c, _) = wl_from_histogram_drag(
        100.0,
        0.0,
        HistogramCanvasGeometry {
            canvas_width: 200.0,
            canvas_height: 80.0,
            hist_min: -1000.0,
            hist_max: 3000.0,
        },
        40.0,
        400.0,
    );
    let expected = 40.0_f32 + 0.5 * 4000.0_f32;
    assert!((c - expected).abs() < 1e-3, "got {c}, expected {expected}");
}

/// Leftward drag shifts center toward lower intensity.
#[test]
fn wl_from_drag_leftward_shifts_center_negative() {
    let (c, _) = wl_from_histogram_drag(
        -200.0,
        0.0,
        HistogramCanvasGeometry {
            canvas_width: 200.0,
            canvas_height: 80.0,
            hist_min: -1000.0,
            hist_max: 3000.0,
        },
        40.0,
        400.0,
    );
    // Δcenter = (-200/200) × 4000 = -4000; new center = 40 - 4000 = -3960
    let expected = 40.0_f32 - 4000.0_f32;
    assert!((c - expected).abs() < 1e-3, "got {c}, expected {expected}");
}

/// Upward drag (dy < 0) narrows the window width.
///
/// Analytical: scale = 1 - (-40/80) = 1.5; new_width = 400 × 1.5 = 600.
#[test]
fn wl_from_drag_upward_narrows_width() {
    let (_, w) = wl_from_histogram_drag(
        0.0,
        -40.0,
        HistogramCanvasGeometry {
            canvas_width: 200.0,
            canvas_height: 80.0,
            hist_min: -1000.0,
            hist_max: 3000.0,
        },
        40.0,
        400.0,
    );
    let expected = 400.0_f32 * 1.5_f32;
    assert!((w - expected).abs() < 1e-3, "got {w}, expected {expected}");
}

/// Downward drag (dy > 0) widens the window width.
///
/// Analytical: scale = 1 - (80/80) = 0.0; new_width = max(1, 400 × 0.0) = 1.
#[test]
fn wl_from_drag_extreme_downward_clamps_to_min_width() {
    let (_, w) = wl_from_histogram_drag(
        0.0,
        80.0,
        HistogramCanvasGeometry {
            canvas_width: 200.0,
            canvas_height: 80.0,
            hist_min: -1000.0,
            hist_max: 3000.0,
        },
        40.0,
        400.0,
    );
    assert!(w >= 1.0, "width {w} below minimum");
    assert!((w - 1.0).abs() < 1e-3, "got {w}, expected 1.0");
}

/// Degenerate canvas_width returns input unchanged.
#[test]
fn wl_from_drag_degenerate_canvas_width_identity() {
    let (c, w) = wl_from_histogram_drag(
        10.0,
        5.0,
        HistogramCanvasGeometry {
            canvas_width: 0.0,
            canvas_height: 80.0,
            hist_min: -1000.0,
            hist_max: 3000.0,
        },
        40.0,
        400.0,
    );
    assert_eq!(c, 40.0);
    assert_eq!(w, 400.0);
}

/// Degenerate intensity span (hist_min == hist_max) returns input unchanged.
#[test]
fn wl_from_drag_degenerate_span_identity() {
    let (c, w) = wl_from_histogram_drag(
        50.0,
        20.0,
        HistogramCanvasGeometry {
            canvas_width: 200.0,
            canvas_height: 80.0,
            hist_min: 1000.0,
            hist_max: 1000.0,
        },
        40.0,
        400.0,
    );
    assert_eq!(c, 40.0);
    assert_eq!(w, 400.0);
}

// ── wl_center_from_click ─────────────────────────────────────────────────

/// Click at canvas left gives hist_min.
#[test]
fn wl_center_from_click_left_returns_min() {
    let c = wl_center_from_click(0.0, -500.0, 500.0, 0.0, 100.0);
    assert_eq!(c, -500.0);
}

/// Click at canvas right gives hist_max.
#[test]
fn wl_center_from_click_right_returns_max() {
    let c = wl_center_from_click(100.0, -500.0, 500.0, 0.0, 100.0);
    assert_eq!(c, 500.0);
}

/// Click at canvas midpoint gives intensity midpoint.
#[test]
fn wl_center_from_click_midpoint_analytical() {
    let c = wl_center_from_click(50.0, -500.0, 500.0, 0.0, 100.0);
    assert!((c - 0.0_f32).abs() < 1e-3, "got {c}, expected 0.0");
}
