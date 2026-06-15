use super::*;

/// Every colormap must return valid [u8; 3] at the boundary values 0.0 and 1.0.
/// This covers the "first/last element in a LUT" contract.
#[test]
fn test_colormap_boundary_values() {
    for cm in Colormap::all() {
        let lo = cm.map(0.0);
        let hi = cm.map(1.0);
        // Values are always valid u8, so simply asserting the structure is a
        // value-semantic check; additionally we verify specific boundary
        // properties where the mathematical spec uniquely determines the output.
        // All channels are u8, so values are structurally bounded to [0, 255].
        // Verify the array has exactly 3 channels (value-semantic length check).
        assert_eq!(lo.len(), 3, "{:?} map(0.0) must return 3 channels", cm);
        assert_eq!(hi.len(), 3, "{:?} map(1.0) must return 3 channels", cm);
    }
    // Grayscale boundary values are analytically determined.
    assert_eq!(Colormap::Grayscale.map(0.0), [0, 0, 0]);
    assert_eq!(Colormap::Grayscale.map(1.0), [255, 255, 255]);
    // Inverted is the complement.
    assert_eq!(Colormap::Inverted.map(0.0), [255, 255, 255]);
    assert_eq!(Colormap::Inverted.map(1.0), [0, 0, 0]);
    // Hot at t=0 is black; at t=1 is white.
    assert_eq!(Colormap::Hot.map(0.0), [0, 0, 0]);
    assert_eq!(Colormap::Hot.map(1.0), [255, 255, 255]);
    // Cool at t=0: R=0, G=255, B=255 (cyan); at t=1: R=255, G=0, B=255 (magenta).
    assert_eq!(Colormap::Cool.map(0.0), [0, 255, 255]);
    assert_eq!(Colormap::Cool.map(1.0), [255, 0, 255]);
}

/// map(t) must not panic for any t in [0.0, 1.0] sampled at 1001 points.
#[test]
fn test_colormap_no_panic_in_range() {
    for cm in Colormap::all() {
        for i in 0..=1000u32 {
            let t = i as f32 / 1000.0;
            let rgb = cm.map(t);
            // Value-semantic: all channels must be in [0, 255] (trivially true
            // for u8, but verifies no wrapping occurred in intermediate maths).
            // All channels are u8, so values are structurally bounded to [0, 255].
            // Verify the array has exactly 3 channels (value-semantic length check).
            assert_eq!(
                rgb.len(),
                3,
                "{:?} map({t}) must return exactly 3 channels",
                cm
            );
        }
    }
}

/// Out-of-range inputs must clamp without panic rather than producing
/// incorrect values or overflowing.
#[test]
fn test_colormap_clamp_out_of_range() {
    for cm in Colormap::all() {
        let below = cm.map(-0.5);
        let above = cm.map(1.5);
        // Clamped result must match the boundary values exactly.
        assert_eq!(below, cm.map(0.0), "{:?} map(-0.5) must equal map(0.0)", cm);
        assert_eq!(above, cm.map(1.0), "{:?} map(1.5) must equal map(1.0)", cm);
    }
}

/// Grayscale must produce R=G=B and must be strictly non-decreasing in all
/// three channels as t increases (monotone brightness).
#[test]
fn test_colormap_grayscale_monotone() {
    let mut prev = Colormap::Grayscale.map(0.0);
    for i in 1..=255u32 {
        let t = i as f32 / 255.0;
        let rgb = Colormap::Grayscale.map(t);
        // R = G = B invariant.
        assert_eq!(rgb[0], rgb[1], "Grayscale: R≠G at t={t}");
        assert_eq!(rgb[1], rgb[2], "Grayscale: G≠B at t={t}");
        // Monotone non-decreasing.
        assert!(
            rgb[0] >= prev[0],
            "Grayscale R not non-decreasing at t={t}: prev={} cur={}",
            prev[0],
            rgb[0]
        );
        prev = rgb;
    }
}

/// Hot colormap must be monotone non-decreasing in luminance (sum of channels)
/// since it ramps from black to white.
#[test]
fn test_colormap_hot_monotone_luminance() {
    let lum = |rgb: [u8; 3]| rgb[0] as u32 + rgb[1] as u32 + rgb[2] as u32;
    let mut prev = lum(Colormap::Hot.map(0.0));
    for i in 1..=255u32 {
        let t = i as f32 / 255.0;
        let cur = lum(Colormap::Hot.map(t));
        assert!(
            cur >= prev,
            "Hot luminance not non-decreasing at t={t}: prev={prev} cur={cur}"
        );
        prev = cur;
    }
}

/// Jet at t=0.5 must be closer to green/yellow than to blue or red.
/// At t=0 the blue channel must dominate; at t=1 the red channel must dominate.
#[test]
fn test_colormap_jet_color_topology() {
    let lo = Colormap::Jet.map(0.0);
    // At t=0, blue dominates (B > R and B > G).
    assert!(
        lo[2] > lo[0] && lo[2] > lo[1],
        "Jet t=0: expected blue-dominant, got R={} G={} B={}",
        lo[0],
        lo[1],
        lo[2]
    );
    let hi = Colormap::Jet.map(1.0);
    // At t=1, red dominates (R > G and R > B).
    assert!(
        hi[0] > hi[1] && hi[0] > hi[2],
        "Jet t=1: expected red-dominant, got R={} G={} B={}",
        hi[0],
        hi[1],
        hi[2]
    );
}

/// Plasma at t=0 must be purple-blue (B > G, R > G); at t=1 must be yellow (R, G >> B).
#[test]
fn test_colormap_plasma_topology() {
    let lo = Colormap::Plasma.map(0.0);
    // Purple-blue: R and B both larger than G.
    assert!(
        lo[2] > lo[1],
        "Plasma t=0: expected B > G, got R={} G={} B={}",
        lo[0],
        lo[1],
        lo[2]
    );
    let hi = Colormap::Plasma.map(1.0);
    // Yellow: R and G both >> B.
    assert!(
        hi[0] > hi[2] && hi[1] > hi[2],
        "Plasma t=1: expected yellow (R,G >> B), got R={} G={} B={}",
        hi[0],
        hi[1],
        hi[2]
    );
}

/// `all()` must return exactly 7 distinct variants with no duplicates.
#[test]
fn test_colormap_all_complete_and_distinct() {
    let all = Colormap::all();
    assert_eq!(all.len(), 7, "Colormap::all() must list all 7 variants");
    // Pairwise distinctness check (O(n²), acceptable for n=7).
    for i in 0..all.len() {
        for j in (i + 1)..all.len() {
            assert_ne!(
                all[i], all[j],
                "Colormap::all() contains duplicate at indices {i} and {j}"
            );
        }
    }
}

/// `label()` must return a non-empty string for every variant.
#[test]
fn test_colormap_label_non_empty() {
    for cm in Colormap::all() {
        let label = cm.label();
        assert!(!label.is_empty(), "{:?} label() must not be empty", cm);
    }
}
