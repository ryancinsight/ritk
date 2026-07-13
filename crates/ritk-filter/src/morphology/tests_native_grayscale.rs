//! Differential + analytical coverage for the Coeus-native grayscale-morphology
//! paths (erosion, dilation, opening, closing, white/black top-hat,
//! hit-or-miss).
//!
//! Each native wrapper must be value-identical to the Burn filter it mirrors —
//! both call the identical substrate-agnostic host core (shared harness in
//! `native_support::assert_native_matches_burn`) — plus analytical oracles that
//! pin the morphological algebra independent of the Burn reference.

use crate::morphology::{
    BlackTopHatFilter, GrayscaleClosingFilter, GrayscaleDilation, GrayscaleErosion,
    GrayscaleOpeningFilter, HitOrMissTransform, WhiteTopHatFilter,
};
use crate::native_support::{assert_native_matches_burn, make_native_image, native_vals};
use coeus_core::SequentialBackend;

fn scattered(dims: [usize; 3]) -> Vec<f32> {
    let n = dims[0] * dims[1] * dims[2];
    (0..n).map(|i| ((i * 37) % 17) as f32).collect()
}

// ── Erosion / dilation ────────────────────────────────────────────────────────

#[test]
fn erosion_matches_burn() {
    let dims = [4usize, 5, 6];
    assert_native_matches_burn(
        scattered(dims),
        dims,
        |img| GrayscaleErosion::new(1).apply(img).expect("burn erosion"),
        |img, backend| GrayscaleErosion::new(1).apply_native(img, backend),
    );
}

#[test]
fn dilation_matches_burn() {
    let dims = [4usize, 5, 6];
    assert_native_matches_burn(
        scattered(dims),
        dims,
        |img| GrayscaleDilation::new(1).apply(img).expect("burn dilation"),
        |img, backend| GrayscaleDilation::new(1).apply_native(img, backend),
    );
}

#[test]
fn oracle_erosion_of_constant_is_constant() {
    let img = make_native_image(vec![7.5f32; 27], [3, 3, 3]);
    let out = GrayscaleErosion::new(1)
        .apply_native(&img, &SequentialBackend)
        .expect("native erosion");
    for &v in &native_vals(&out) {
        assert_eq!(v, 7.5, "erosion of a constant field must be the constant");
    }
}

#[test]
fn oracle_erosion_le_original_le_dilation() {
    let dims = [4usize, 5, 6];
    let vals = scattered(dims);
    let img = make_native_image(vals.clone(), dims);
    let eroded = native_vals(
        &GrayscaleErosion::new(1)
            .apply_native(&img, &SequentialBackend)
            .expect("native erosion"),
    );
    let dilated = native_vals(
        &GrayscaleDilation::new(1)
            .apply_native(&img, &SequentialBackend)
            .expect("native dilation"),
    );
    for i in 0..vals.len() {
        assert!(
            eroded[i] <= vals[i] && vals[i] <= dilated[i],
            "expected erosion <= original <= dilation at {i}: {} <= {} <= {}",
            eroded[i],
            vals[i],
            dilated[i]
        );
    }
}

// ── Opening / closing ─────────────────────────────────────────────────────────

#[test]
fn opening_matches_burn() {
    let dims = [4usize, 5, 6];
    assert_native_matches_burn(
        scattered(dims),
        dims,
        |img| {
            GrayscaleOpeningFilter::new(1)
                .apply(img)
                .expect("burn opening")
        },
        |img, backend| GrayscaleOpeningFilter::new(1).apply_native(img, backend),
    );
}

#[test]
fn closing_matches_burn() {
    let dims = [4usize, 5, 6];
    assert_native_matches_burn(
        scattered(dims),
        dims,
        |img| {
            GrayscaleClosingFilter::new(1)
                .apply(img)
                .expect("burn closing")
        },
        |img, backend| GrayscaleClosingFilter::new(1).apply_native(img, backend),
    );
}

#[test]
fn oracle_opening_le_original_le_closing() {
    let dims = [4usize, 5, 6];
    let vals = scattered(dims);
    let img = make_native_image(vals.clone(), dims);
    let opened = native_vals(
        &GrayscaleOpeningFilter::new(1)
            .apply_native(&img, &SequentialBackend)
            .expect("native opening"),
    );
    let closed = native_vals(
        &GrayscaleClosingFilter::new(1)
            .apply_native(&img, &SequentialBackend)
            .expect("native closing"),
    );
    for i in 0..vals.len() {
        assert!(
            opened[i] <= vals[i] + 1e-6,
            "opening must be anti-extensive at {i}: {} <= {}",
            opened[i],
            vals[i]
        );
        assert!(
            closed[i] + 1e-6 >= vals[i],
            "closing must be extensive at {i}: {} >= {}",
            closed[i],
            vals[i]
        );
    }
}

// ── Top-hat ───────────────────────────────────────────────────────────────────

#[test]
fn white_top_hat_matches_burn() {
    let dims = [4usize, 5, 6];
    assert_native_matches_burn(
        scattered(dims),
        dims,
        |img| {
            WhiteTopHatFilter::new(1)
                .apply(img)
                .expect("burn white top-hat")
        },
        |img, backend| WhiteTopHatFilter::new(1).apply_native(img, backend),
    );
}

#[test]
fn black_top_hat_matches_burn() {
    let dims = [4usize, 5, 6];
    assert_native_matches_burn(
        scattered(dims),
        dims,
        |img| {
            BlackTopHatFilter::new(1)
                .apply(img)
                .expect("burn black top-hat")
        },
        |img, backend| BlackTopHatFilter::new(1).apply_native(img, backend),
    );
}

#[test]
fn oracle_top_hat_of_constant_is_zero() {
    let img = make_native_image(vec![4.0f32; 27], [3, 3, 3]);
    let wth = native_vals(
        &WhiteTopHatFilter::new(1)
            .apply_native(&img, &SequentialBackend)
            .expect("native white top-hat"),
    );
    let bth = native_vals(
        &BlackTopHatFilter::new(1)
            .apply_native(&img, &SequentialBackend)
            .expect("native black top-hat"),
    );
    for (&w, &b) in wth.iter().zip(bth.iter()) {
        assert_eq!(w, 0.0, "white top-hat of a constant must be 0");
        assert_eq!(b, 0.0, "black top-hat of a constant must be 0");
    }
}

#[test]
fn oracle_top_hat_non_negative() {
    let dims = [4usize, 5, 6];
    let img = make_native_image(scattered(dims), dims);
    let wth = native_vals(
        &WhiteTopHatFilter::new(1)
            .apply_native(&img, &SequentialBackend)
            .expect("native white top-hat"),
    );
    for &w in &wth {
        assert!(w >= 0.0, "white top-hat must be non-negative, got {w}");
    }
}

// ── Hit-or-miss ───────────────────────────────────────────────────────────────

#[test]
fn hit_or_miss_matches_burn() {
    // Binary volume with a solid 3x3x3 fg block embedded in background.
    let dims = [5usize, 5, 5];
    let n = dims[0] * dims[1] * dims[2];
    let vals: Vec<f32> = (0..n)
        .map(|i| {
            let z = i / 25;
            let y = (i % 25) / 5;
            let x = i % 5;
            if (1..=3).contains(&z) && (1..=3).contains(&y) && (1..=3).contains(&x) {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    assert_native_matches_burn(
        vals,
        dims,
        |img| {
            HitOrMissTransform::new(1, 1)
                .apply(img)
                .expect("burn hit-or-miss")
        },
        |img, backend| HitOrMissTransform::new(1, 1).apply_native(img, backend),
    );
}
