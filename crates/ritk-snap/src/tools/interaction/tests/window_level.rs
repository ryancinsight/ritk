use super::*;
// ── WindowLevel monotone (cross-module) ───────────────────────────────────

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
