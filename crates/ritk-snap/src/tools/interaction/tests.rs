use super::*;
use crate::render::slice_render::WindowLevel;
use crate::render::NamedColorMap;
use crate::tools::ToolKind;
use egui::Pos2;
use iris::color::{ColorMap, Normalized};

// ── compute_length ────────────────────────────────────────────────────────

/// Axis-aligned horizontal displacement with unit spacing must yield the
/// exact integer pixel distance.
///
/// Analytical: p1=[0,0], p2=[0,3], spacing=[1,1]
/// length = √( (0·1)² + (3·1)² ) = √9 = 3.0
#[test]
fn test_compute_length_axis_aligned() {
    let p1 = [0.0_f32, 0.0];
    let p2 = [0.0_f32, 3.0];
    let spacing = [1.0_f32, 1.0];
    let length = Annotation::compute_length(p1, p2, spacing);
    assert_eq!(
        length, 3.0_f32,
        "axis-aligned horizontal length of 3 pixels with unit spacing must equal 3.0 mm"
    );
}

/// Axis-aligned vertical displacement with unit spacing.
///
/// Analytical: p1=[0,0], p2=[4,0], spacing=[1,1]
/// length = √( (4·1)² + (0·1)² ) = √16 = 4.0
#[test]
fn test_compute_length_axis_aligned_vertical() {
    let p1 = [0.0_f32, 0.0];
    let p2 = [4.0_f32, 0.0];
    let spacing = [1.0_f32, 1.0];
    let length = Annotation::compute_length(p1, p2, spacing);
    assert_eq!(
        length, 4.0_f32,
        "axis-aligned vertical length of 4 pixels with unit spacing must equal 4.0 mm"
    );
}

/// Non-unit spacing scales the physical length independently per axis.
///
/// Analytical: p1=[0,0], p2=[2,0], spacing=[0.5, 1.0]
/// length = √( (2·0.5)² + (0·1.0)² ) = √1 = 1.0
#[test]
fn test_compute_length_anisotropic_spacing() {
    let p1 = [0.0_f32, 0.0];
    let p2 = [2.0_f32, 0.0];
    let spacing = [0.5_f32, 1.0];
    let length = Annotation::compute_length(p1, p2, spacing);
    let expected = 1.0_f32;
    assert!(
        (length - expected).abs() < 1e-5,
        "anisotropic length: expected {expected}, got {length}"
    );
}

// ── compute_angle ─────────────────────────────────────────────────────────

/// Three points forming a right angle (90°) at the vertex.
///
/// Analytical:
/// p1 = (0, 1), p2 = (0, 0) [vertex], p3 = (1, 0)
/// v₁ = (0,1)−(0,0) = (0,1), v₂ = (1,0)−(0,0) = (1,0)
/// dot = 0·1 + 1·0 = 0 ⟹ cos θ = 0 ⟹ θ = 90°
#[test]
fn test_compute_angle_right_angle() {
    let p1 = [0.0_f32, 1.0];
    let p2 = [0.0_f32, 0.0]; // vertex
    let p3 = [1.0_f32, 0.0];
    let angle = Annotation::compute_angle(p1, p2, p3);
    assert!(
        (angle - 90.0_f32).abs() < 0.001,
        "90° angle must be computed to within 0.001°, got {angle}°"
    );
}

/// Three collinear points (same direction) must yield 0°.
///
/// Analytical:
/// p1 = (0, 0), p2 = (0, 1) [vertex], p3 = (0, 2)
/// v₁ = (0,−1), v₂ = (0,1)
/// cos θ = −1 ⟹ θ = 180°
///
/// Note: the two rays point in exactly opposite directions, so the angle
/// at the vertex is 180°.
#[test]
fn test_compute_angle_straight_line() {
    let p1 = [0.0_f32, 0.0];
    let p2 = [0.0_f32, 1.0]; // vertex on the line
    let p3 = [0.0_f32, 2.0];
    let angle = Annotation::compute_angle(p1, p2, p3);
    assert!(
        (angle - 180.0_f32).abs() < 0.001,
        "straight-line angle must be 180°, got {angle}°"
    );
}

/// Degenerate input where p1 == p2 must return 0.0 rather than NaN or panic.
#[test]
fn test_compute_angle_degenerate_zero_length_ray() {
    let p1 = [1.0_f32, 1.0];
    let p2 = [1.0_f32, 1.0]; // p1 == p2 → zero-length ray
    let p3 = [2.0_f32, 3.0];
    let angle = Annotation::compute_angle(p1, p2, p3);
    assert_eq!(
        angle, 0.0_f32,
        "degenerate (zero-length ray) angle must return 0.0, got {angle}"
    );
}

// ── ToolState ─────────────────────────────────────────────────────────────

/// `Idle` must report `is_idle() == true` and `tool_kind() == None`.
#[test]
fn test_tool_state_idle() {
    let state = ToolState::Idle;
    assert!(state.is_idle(), "Idle must report is_idle() = true");
    assert_eq!(state.tool_kind(), None, "Idle must have tool_kind() = None");
}

/// Each non-idle variant must report `is_idle() == false` and the
/// correct `ToolKind`.
#[test]
fn test_tool_state_non_idle_variants() {
    let cases: &[(ToolState, ToolKind)] = &[
        (
            ToolState::Panning {
                start: Pos2::ZERO,
                viewport_origin: Pos2::ZERO,
            },
            ToolKind::Pan,
        ),
        (
            ToolState::Zooming {
                start: Pos2::ZERO,
                original_zoom: 1.0,
            },
            ToolKind::Zoom,
        ),
        (
            ToolState::WindowLevelDrag {
                start: Pos2::ZERO,
                original_center: 0.0,
                original_width: 1.0,
            },
            ToolKind::WindowLevel,
        ),
        (
            ToolState::MeasureLength1 { p1: Pos2::ZERO },
            ToolKind::MeasureLength,
        ),
        (
            ToolState::MeasureAngle2 {
                p1: Pos2::ZERO,
                p2: Pos2::new(1.0, 0.0),
            },
            ToolKind::MeasureAngle,
        ),
        (
            ToolState::RoiDrag {
                start: Pos2::ZERO,
                current: Pos2::new(1.0, 1.0),
                kind: RoiKind::Rect,
            },
            ToolKind::RoiRect,
        ),
        (
            ToolState::RoiDrag {
                start: Pos2::ZERO,
                current: Pos2::new(1.0, 1.0),
                kind: RoiKind::Ellipse,
            },
            ToolKind::RoiEllipse,
        ),
    ];

    for (state, expected_kind) in cases {
        assert!(!state.is_idle(), "{:?} must not be idle", state.tool_kind());
        assert_eq!(
            state.tool_kind(),
            Some(*expected_kind),
            "tool_kind() must return {:?}",
            expected_kind
        );
    }
}

// ── WindowLevel monotone (cross-module, uses slice_render::WindowLevel) ───

/// [`WindowLevel::apply`] must produce monotonically non-decreasing output
/// over 100 uniformly spaced input values in [0, 1000].
///
/// Analytical justification: the DICOM PS 3.3 §C.7.6.3.1.5 formula is
/// piece-wise linear with non-negative slope in every segment; therefore
/// the mapping is monotone non-decreasing by construction.
#[test]
fn test_window_level_apply_range() {
    let wl = WindowLevel::new(500.0, 1000.0);

    // 100 uniformly spaced values in [0.0, 1000.0].
    let values: Vec<f64> = (0..100).map(|i| i as f64 * (1000.0 / 99.0)).collect();
    let mut prev = wl.apply(values[0]);
    for &v in &values[1..] {
        let cur = wl.apply(v);
        assert!(
            cur >= prev,
            "WindowLevel::apply must be non-decreasing: apply({v}) = {cur} < prev = {prev}"
        );
        prev = cur;
    }
}

// ── NamedColorMap grayscale (cross-module, uses Iris) ────────────────────────

/// [`NamedColorMap::Grayscale`] must produce R = G = B and must be monotonically
/// non-decreasing in the R channel as `t` increases from 0 to 1.
///
/// Analytical: R(t) = round(t × 255), which is non-decreasing for t ∈ [0, 1].
#[test]
fn test_colormap_grayscale_monotone() {
    let cm = NamedColorMap::Grayscale;
    let mut prev_r = cm
        .sample(Normalized::new(0.0).expect("zero is normalized"))
        .to_rgba8()[0];
    for i in 1..=255u32 {
        let t = i as f32 / 255.0;
        let [r, g, b, _] = cm
            .sample(Normalized::new(t).expect("generated test value is normalized"))
            .to_rgba8();
        // R = G = B invariant.
        assert_eq!(r, g, "Grayscale R≠G at t={t}");
        assert_eq!(g, b, "Grayscale G≠B at t={t}");
        // Monotone non-decreasing R channel.
        assert!(
            r >= prev_r,
            "Grayscale R not non-decreasing at t={t}: prev={prev_r}, cur={r}"
        );
        prev_r = r;
    }
}

// ── compute_roi_rect_stats ────────────────────────────────────────────────

/// A 1×3 ROI over pixels [10, 20, 30] (row-major, width=3, height=1).
///
/// Analytical:
/// mean = (10 + 20 + 30) / 3 = 20.0
/// std_dev = √( ((10−20)² + (20−20)² + (30−20)²) / 3 )
///         = √( (100 + 0 + 100) / 3 )
///         = √(200/3) ≈ 8.164_966
/// min = 10.0, max = 30.0
/// area = 1 × 1.0 × 3 × 1.0 = 3.0 mm²
#[test]
fn test_compute_roi_rect_stats_analytic() {
    let pixels = vec![10.0_f32, 20.0, 30.0];
    let (mean, std_dev, min, max, area) =
        Annotation::compute_roi_rect_stats([0.0, 0.0], [0.0, 2.0], &pixels, 3, 1, [1.0, 1.0]);
    assert!((mean - 20.0).abs() < 1e-5, "mean must be 20.0, got {mean}");
    let expected_std = (200.0_f32 / 3.0).sqrt();
    assert!(
        (std_dev - expected_std).abs() < 1e-4,
        "std_dev must be {expected_std}, got {std_dev}"
    );
    assert!((min - 10.0).abs() < 1e-5, "min must be 10.0, got {min}");
    assert!((max - 30.0).abs() < 1e-5, "max must be 30.0, got {max}");
    assert!(
        (area - 3.0).abs() < 1e-5,
        "area must be 3.0 mm², got {area}"
    );
}

/// An ROI entirely outside the image must return the all-zero tuple without
/// panicking.
#[test]
fn test_compute_roi_rect_stats_out_of_bounds() {
    let pixels = vec![1.0_f32; 4]; // 2×2 image
    let (mean, std_dev, min, max, area) =
        Annotation::compute_roi_rect_stats([10.0, 10.0], [20.0, 20.0], &pixels, 2, 2, [1.0, 1.0]);
    // Clamped to last valid pixel — still produces a valid (non-NaN) result.
    assert!(
        mean.is_finite(),
        "out-of-bounds ROI must return finite mean, got {mean}"
    );
    assert!(
        std_dev.is_finite(),
        "out-of-bounds ROI must return finite std_dev, got {std_dev}"
    );
    assert!(
        area >= 0.0,
        "out-of-bounds ROI must return non-negative area, got {area}"
    );
    // The entire image is 1.0, so mean and std_dev are deterministic.
    assert!(
        (mean - 1.0).abs() < 1e-5,
        "clamped out-of-bounds ROI over constant field must have mean=1.0, got {mean}"
    );
    let _ = (min, max); // min/max are valid but their exact values depend on clamping
}

// ── compute_roi_ellipse_stats ─────────────────────────────────────────────

/// A 5×5 constant field (all pixels = 2.0) with a 5×5 bounding box must
/// produce mean = 2.0 and std_dev = 0.0 over only the pixels inside the
/// inscribed ellipse.
///
/// Analytical:
/// p1=[0,0], p2=[4,4] → center=[2,2], a=2, b=2 (circle radius 2)
/// Pixel (r,c) inside when ((r−2)/2)²+((c−2)/2)²≤1, i.e., (r−2)²+(c−2)²≤4
/// Inside pixels: center ring pattern (13 pixels in a 5×5 grid for r=2 circle)
/// All values = 2.0 → mean = 2.0, std_dev = 0.0
#[test]
fn test_compute_roi_ellipse_constant_field_mean_and_stddev() {
    let pixels: Vec<f32> = vec![2.0; 25]; // 5×5 grid
    let (center, radii, mean, std_dev, min, max, area_mm2) =
        Annotation::compute_roi_ellipse_stats([0.0, 0.0], [4.0, 4.0], &pixels, 5, 5, [1.0, 1.0]);

    assert_eq!(center, [2.0, 2.0], "center must be midpoint [2,2]");
    assert_eq!(radii, [2.0, 2.0], "radii must be half bounding box [2,2]");
    assert!(
        (mean - 2.0).abs() < 1e-5,
        "constant field mean must be 2.0, got {mean}"
    );
    assert!(
        std_dev.abs() < 1e-5,
        "constant field std_dev must be 0.0, got {std_dev}"
    );
    assert_eq!(min, 2.0, "constant field min must be 2.0");
    assert_eq!(max, 2.0, "constant field max must be 2.0");

    // area = π × 2 × 1.0 × 2 × 1.0 = 4π ≈ 12.566
    let expected_area = std::f32::consts::PI * 2.0 * 2.0;
    assert!(
        (area_mm2 - expected_area).abs() < 1e-4,
        "area must be π×2×2 = {expected_area:.4}, got {area_mm2:.4}"
    );
}

/// Degenerate ellipse (zero row semi-axis) must return all-zero statistics.
///
/// Analytical: p1=[2,0], p2=[2,4] → a=0, b=2. Since a=0, membership
/// condition has division by zero — function must return zeros.
#[test]
fn test_compute_roi_ellipse_degenerate_zero_row_radius() {
    let pixels: Vec<f32> = vec![5.0; 25];
    let (center, radii, mean, std_dev, min, max, area_mm2) =
        Annotation::compute_roi_ellipse_stats([2.0, 0.0], [2.0, 4.0], &pixels, 5, 5, [1.0, 1.0]);

    assert_eq!(center, [2.0, 2.0]);
    assert_eq!(radii[0], 0.0, "row radius must be 0 for degenerate input");
    assert_eq!(mean, 0.0, "degenerate ellipse must have mean=0.0");
    assert_eq!(std_dev, 0.0);
    assert_eq!(min, 0.0);
    assert_eq!(max, 0.0);
    assert_eq!(area_mm2, 0.0);
}

/// Pixels strictly outside the ellipse boundary must be excluded.
///
/// 3×3 pixel grid with corner pixels set to 100 and centre set to 1.0.
/// Ellipse p1=[0,0], p2=[2,2] → center=[1,1], a=1, b=1 (unit circle).
///
/// Membership: ((r−1)/1)²+((c−1)/1)²≤1
/// (0,0): (−1)²+(−1)²=2 > 1 → outside
/// (0,1): (−1)²+(0)²=1 ≤ 1 → inside
/// (0,2): (−1)²+(1)²=2 > 1 → outside
/// (1,0): (0)²+(−1)²=1 ≤ 1 → inside
/// (1,1): (0)²+(0)²=0 ≤ 1 → inside
/// (1,2): (0)²+(1)²=1 ≤ 1 → inside
/// (2,0): (1)²+(−1)²=2 > 1 → outside
/// (2,1): (1)²+(0)²=1 ≤ 1 → inside
/// (2,2): (1)²+(1)²=2 > 1 → outside
///
/// Corner pixels (value 100) at (0,0),(0,2),(2,0),(2,2) are excluded.
/// Inside pixels: (0,1),(1,0),(1,1),(1,2),(2,1) with values [0,1,2,3,4]
#[test]
fn test_compute_roi_ellipse_excludes_corners() {
    // 3×3 grid, values 0..9 row-major
    // (0,0)=0, (0,1)=1, (0,2)=2, (1,0)=3, (1,1)=4, (1,2)=5, (2,0)=6, (2,1)=7, (2,2)=8
    let pixels: Vec<f32> = (0..9).map(|v| v as f32).collect();
    let (_center, _radii, mean, std_dev, min, max, _area_mm2) =
        Annotation::compute_roi_ellipse_stats([0.0, 0.0], [2.0, 2.0], &pixels, 3, 3, [1.0, 1.0]);

    // Inside pixels: (0,1)=1, (1,0)=3, (1,1)=4, (1,2)=5, (2,1)=7
    // Analytical: mean = (1+3+4+5+7)/5 = 20/5 = 4.0
    // variance = ((1−4)²+(3−4)²+(4−4)²+(5−4)²+(7−4)²)/5
    //          = (9+1+0+1+9)/5 = 20/5 = 4.0
    // std_dev = √4.0 = 2.0
    assert!(
        (mean - 4.0).abs() < 1e-5,
        "mean of inside pixels must be 4.0, got {mean}"
    );
    assert!(
        (std_dev - 2.0).abs() < 1e-4,
        "std_dev of inside pixels must be 2.0, got {std_dev}"
    );
    assert_eq!(min, 1.0, "min inside must be 1 (pixel (0,1))");
    assert_eq!(max, 7.0, "max inside must be 7 (pixel (2,1))");
}

/// Physical area uses π × a × spacing[0] × b × spacing[1].
///
/// p1=[0,0], p2=[10,6] → a=5, b=3, spacing=[2.0, 3.0]
/// area = π × 5 × 2.0 × 3 × 3.0 = π × 90 ≈ 282.743
#[test]
fn test_compute_roi_ellipse_area_anisotropic_spacing() {
    let pixels: Vec<f32> = vec![1.0; 200]; // 20×10 grid, all pixels inside will be 1.0
    let (_center, _radii, _mean, _std_dev, _min, _max, area_mm2) =
        Annotation::compute_roi_ellipse_stats([0.0, 0.0], [10.0, 6.0], &pixels, 10, 11, [2.0, 3.0]);
    let expected = std::f32::consts::PI * 5.0 * 2.0 * 3.0 * 3.0;
    assert!(
        (area_mm2 - expected).abs() < 1e-3,
        "area with anisotropic spacing must be {expected:.3}, got {area_mm2:.3}"
    );
}

/// A single-pixel ellipse (bounding box is one pixel: p1=p2) is degenerate.
///
/// Analytical: p1=[3,3], p2=[3,3] → a=0, b=0. Degenerate: return zeros.
#[test]
fn test_compute_roi_ellipse_single_point_is_degenerate() {
    let pixels: Vec<f32> = vec![42.0; 25];
    let (_center, _radii, mean, std_dev, min, max, area_mm2) =
        Annotation::compute_roi_ellipse_stats([3.0, 3.0], [3.0, 3.0], &pixels, 5, 5, [1.0, 1.0]);
    assert_eq!(mean, 0.0, "zero-radius ellipse must have mean=0.0");
    assert_eq!(std_dev, 0.0);
    assert_eq!(min, 0.0);
    assert_eq!(max, 0.0);
    assert_eq!(area_mm2, 0.0);
}
