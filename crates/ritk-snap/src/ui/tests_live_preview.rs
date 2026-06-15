use super::*;

// ── live_length_mm ────────────────────────────────────────────────────────

#[test]
fn live_length_horizontal_unit_spacing_matches_pixel_distance() {
    // p1=[0,0] p2=[0,3], spacing=[1,1] → Δcol=3, d=3.0 mm
    let mm = live_length_mm([0.0, 0.0], [0.0, 3.0], [1.0, 1.0]);
    assert!((mm - 3.0).abs() < 1e-5, "expected 3.0 mm, got {mm}");
}

#[test]
fn live_length_vertical_unit_spacing_matches_pixel_distance() {
    // p1=[0,0] p2=[4,0], spacing=[1,1] → Δrow=4, d=4.0 mm
    let mm = live_length_mm([0.0, 0.0], [4.0, 0.0], [1.0, 1.0]);
    assert!((mm - 4.0).abs() < 1e-5, "expected 4.0 mm, got {mm}");
}

#[test]
fn live_length_anisotropic_spacing_scales_per_axis() {
    // p1=[0,0] p2=[3,4], spacing=[2.0, 0.5]
    // dr=3×2=6, dc=4×0.5=2 → √(36+4) = √40 ≈ 6.3246
    let mm = live_length_mm([0.0, 0.0], [3.0, 4.0], [2.0, 0.5]);
    let expected = (36.0_f32 + 4.0_f32).sqrt();
    assert!(
        (mm - expected).abs() < 1e-4,
        "expected {expected:.6} mm, got {mm:.6}"
    );
}

#[test]
fn live_length_zero_delta_returns_zero() {
    // Same point → distance = 0
    let mm = live_length_mm([5.0, 3.0], [5.0, 3.0], [1.5, 0.8]);
    assert!(mm.abs() < 1e-6, "expected 0.0 mm, got {mm}");
}

#[test]
fn live_length_diagonal_pythagorean_triple() {
    // p1=[0,0] p2=[3,4], spacing=[1,1] → d=√(9+16)=5.0 mm  (3-4-5)
    let mm = live_length_mm([0.0, 0.0], [3.0, 4.0], [1.0, 1.0]);
    assert!((mm - 5.0).abs() < 1e-5, "expected 5.0 mm, got {mm}");
}

// ── live_angle_deg ────────────────────────────────────────────────────────

#[test]
fn live_angle_right_angle_returns_90_degrees() {
    // vertex at origin, p1 in +row direction, p3 in +col direction
    // v1=[1,0], v3=[0,1] → dot=0 → θ=90°
    let deg = live_angle_deg([1.0, 0.0], [0.0, 0.0], [0.0, 1.0]);
    assert!((deg - 90.0).abs() < 1e-4, "expected 90.0°, got {deg}");
}

#[test]
fn live_angle_straight_line_returns_180_degrees() {
    // vertex at origin, p1=[1,0], p3=[-1,0]: opposite collinear vectors
    // v1=[1,0], v3=[-1,0] → dot=-1 → θ=180°
    let deg = live_angle_deg([1.0, 0.0], [0.0, 0.0], [-1.0, 0.0]);
    assert!((deg - 180.0).abs() < 1e-4, "expected 180.0°, got {deg}");
}

#[test]
fn live_angle_45_degrees_analytical() {
    // vertex at origin, p1=[1,0], p3=[1,1]
    // v1=[1,0], v3=[1,1], dot=1, |v1|=1, |v3|=√2
    // cos θ = 1/√2 → θ = 45°
    let deg = live_angle_deg([1.0, 0.0], [0.0, 0.0], [1.0, 1.0]);
    assert!((deg - 45.0).abs() < 1e-4, "expected 45.0°, got {deg}");
}

#[test]
fn live_angle_degenerate_p1_equals_vertex_returns_zero() {
    // p1 == vertex → zero-length ray → returns 0
    let deg = live_angle_deg([0.0, 0.0], [0.0, 0.0], [1.0, 0.0]);
    assert!(
        deg.abs() < 1e-6,
        "expected 0.0° for degenerate input, got {deg}"
    );
}

#[test]
fn live_angle_60_degrees_equilateral_analytical() {
    // Equilateral triangle: vertex at origin, p1=[1,0], p3=[0.5, 0.866025]
    // v1=[1,0], v3=[0.5,0.866025]: dot=0.5, |v1|=1, |v3|=1 → cos θ=0.5 → θ=60°
    let deg = live_angle_deg([1.0, 0.0], [0.0, 0.0], [0.5, 0.866_025_4]);
    assert!((deg - 60.0).abs() < 1e-3, "expected 60.0°, got {deg}");
}
