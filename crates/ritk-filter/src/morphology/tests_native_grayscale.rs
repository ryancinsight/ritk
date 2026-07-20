//! Differential + analytical coverage for the Coeus-native grayscale-morphology
//! paths (erosion, dilation, opening, closing, white/black top-hat,
//! hit-or-miss).
//!
//! Each native wrapper must be value-identical to the Coeus filter it mirrors —
//! both call the identical substrate-agnostic host core (shared harness in
//! `native_support::assert_coeus_matches_coeus`) — plus analytical oracles that
//! pin the morphological algebra independent of the Coeus reference.

use crate::morphology::{
    BlackTopHatFilter, GrayscaleClosingFilter, GrayscaleDilation, GrayscaleErosion,
    GrayscaleOpeningFilter, HitOrMissTransform, WhiteTopHatFilter,
};
use crate::native_support::{assert_coeus_matches_coeus, make_native_image, native_vals};
use coeus_core::SequentialBackend;

fn scattered(dims: [usize; 3]) -> Vec<f32> {
    let n = dims[0] * dims[1] * dims[2];
    (0..n).map(|i| ((i * 37) % 17) as f32).collect()
}

// ── Erosion / dilation ────────────────────────────────────────────────────────

#[test]
fn erosion_matches_coeus() {
    let dims = [4usize, 5, 6];
    assert_coeus_matches_coeus(
        scattered(dims),
        dims,
        |img| GrayscaleErosion::new(1).apply(img).expect("coeus erosion"),
        |img, backend| GrayscaleErosion::new(1).apply_native(img, backend),
    );
}

#[test]
fn dilation_matches_coeus() {
    let dims = [4usize, 5, 6];
    assert_coeus_matches_coeus(
        scattered(dims),
        dims,
        |img| GrayscaleDilation::new(1).apply(img).expect("coeus dilation"),
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
fn opening_matches_coeus() {
    let dims = [4usize, 5, 6];
    assert_coeus_matches_coeus(
        scattered(dims),
        dims,
        |img| {
            GrayscaleOpeningFilter::new(1)
                .apply(img)
                .expect("coeus opening")
        },
        |img, backend| GrayscaleOpeningFilter::new(1).apply_native(img, backend),
    );
}

#[test]
fn closing_matches_coeus() {
    let dims = [4usize, 5, 6];
    assert_coeus_matches_coeus(
        scattered(dims),
        dims,
        |img| {
            GrayscaleClosingFilter::new(1)
                .apply(img)
                .expect("coeus closing")
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
fn white_top_hat_matches_coeus() {
    let dims = [4usize, 5, 6];
    assert_coeus_matches_coeus(
        scattered(dims),
        dims,
        |img| {
            WhiteTopHatFilter::new(1)
                .apply(img)
                .expect("coeus white top-hat")
        },
        |img, backend| WhiteTopHatFilter::new(1).apply_native(img, backend),
    );
}

#[test]
fn black_top_hat_matches_coeus() {
    let dims = [4usize, 5, 6];
    assert_coeus_matches_coeus(
        scattered(dims),
        dims,
        |img| {
            BlackTopHatFilter::new(1)
                .apply(img)
                .expect("coeus black top-hat")
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
fn hit_or_miss_matches_coeus() {
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
    assert_coeus_matches_coeus(
        vals,
        dims,
        |img| {
            HitOrMissTransform::new(1, 1)
                .apply(img)
                .expect("coeus hit-or-miss")
        },
        |img, backend| HitOrMissTransform::new(1, 1).apply_native(img, backend),
    );
}

// ── Morphological gradient / Laplacian (added: native path) ─────────────────────

mod gradient_and_laplace {
    use super::*;
    use crate::morphology::{GrayscaleMorphologicalGradientFilter, MorphologicalLaplacian};

    #[test]
    fn gradient_matches_coeus() {
        let dims = [4usize, 5, 6];
        assert_coeus_matches_coeus(
            scattered(dims),
            dims,
            |img| {
                GrayscaleMorphologicalGradientFilter::new(1)
                    .apply(img)
                    .expect("coeus gradient")
            },
            |img, backend| GrayscaleMorphologicalGradientFilter::new(1).apply_native(img, backend),
        );
    }

    #[test]
    fn oracle_gradient_of_constant_is_zero() {
        let img = make_native_image(vec![4.0f32; 27], [3, 3, 3]);
        let out = GrayscaleMorphologicalGradientFilter::new(1)
            .apply_native(&img, &SequentialBackend)
            .expect("native gradient");
        for &v in &native_vals(&out) {
            assert_eq!(v, 0.0, "gradient of a constant field must be zero");
        }
    }

    #[test]
    fn laplace_matches_coeus() {
        let dims = [4usize, 5, 6];
        assert_coeus_matches_coeus(
            scattered(dims),
            dims,
            |img| {
                MorphologicalLaplacian::new(1)
                    .apply(img)
                    .expect("coeus laplace")
            },
            |img, backend| MorphologicalLaplacian::new(1).apply_native(img, backend),
        );
    }

    #[test]
    fn oracle_laplace_of_constant_is_zero() {
        let img = make_native_image(vec![4.0f32; 27], [3, 3, 3]);
        let out = MorphologicalLaplacian::new(1)
            .apply_native(&img, &SequentialBackend)
            .expect("native laplace");
        for &v in &native_vals(&out) {
            assert_eq!(
                v, 0.0,
                "morphological Laplacian of a constant field must be zero"
            );
        }
    }
}

// ── Fill-hole (grayscale + binary) ──────────────────────────────────────────────

mod fillhole {
    use super::*;
    use crate::morphology::{BinaryFillholeFilter, GrayscaleFillholeFilter};

    /// A bright wall enclosing one dark interior voxel (a hole).
    fn walled_pit() -> ([usize; 3], Vec<f32>) {
        let dims = [3usize, 3, 3];
        let mut v = vec![10.0f32; 27];
        v[13] = 0.0; // centre voxel is the pit
        (dims, v)
    }

    #[test]
    fn grayscale_matches_coeus() {
        let (dims, v) = walled_pit();
        assert_coeus_matches_coeus(
            v,
            dims,
            |img| {
                GrayscaleFillholeFilter::new()
                    .apply(img)
                    .expect("coeus fillhole")
            },
            |img, backend| GrayscaleFillholeFilter::new().apply_native(img, backend),
        );
    }

    #[test]
    fn oracle_grayscale_raises_pit() {
        let (dims, v) = walled_pit();
        let img = make_native_image(v, dims);
        let out = GrayscaleFillholeFilter::new()
            .apply_native(&img, &SequentialBackend)
            .expect("native fillhole");
        // The enclosed pit is raised to the surrounding wall level (10).
        assert_eq!(
            native_vals(&out)[13],
            10.0,
            "enclosed pit must be filled to wall level"
        );
    }

    #[test]
    fn binary_matches_coeus() {
        let (dims, v) = walled_pit();
        // Binary image: wall = 1, pit = 0.
        let bin: Vec<f32> = v.iter().map(|&x| if x > 5.0 { 1.0 } else { 0.0 }).collect();
        assert_coeus_matches_coeus(
            bin,
            dims,
            |img| {
                BinaryFillholeFilter::new()
                    .apply(img)
                    .expect("coeus binary fillhole")
            },
            |img, backend| BinaryFillholeFilter::new().apply_native(img, backend),
        );
    }
}

// ── Voting binary ──────────────────────────────────────────────────────────────

mod voting_binary {
    use super::*;
    use crate::morphology::VotingBinaryImageFilter;

    #[test]
    fn matches_coeus() {
        let dims = [4usize, 4, 4];
        let n = dims[0] * dims[1] * dims[2];
        let vals: Vec<f32> = (0..n).map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }).collect();
        assert_coeus_matches_coeus(
            vals,
            dims,
            |img| {
                VotingBinaryImageFilter::default()
                    .apply(img)
                    .expect("coeus voting")
            },
            |img, backend| VotingBinaryImageFilter::default().apply_native(img, backend),
        );
    }
}

// ── Geodesic reconstruction (two-input) + reconstruction opening/closing ────────

mod reconstruction {
    use super::*;
    use crate::morphology::{
        ClosingByReconstructionFilter, GrayscaleGeodesicDilationFilter,
        OpeningByReconstructionFilter,
    };
    use crate::native_support::assert_coeus_matches_coeus_pair;

    #[test]
    fn geodesic_dilation_matches_coeus() {
        let dims = [3usize, 4, 5];
        let n = dims[0] * dims[1] * dims[2];
        let mask: Vec<f32> = (0..n).map(|i| ((i * 11) % 19) as f32).collect();
        // Marker ≤ mask (dilation reconstruction precondition).
        let marker: Vec<f32> = mask.iter().map(|&m| m * 0.5).collect();
        assert_coeus_matches_coeus_pair(
            marker,
            mask,
            dims,
            |mk, ms| {
                GrayscaleGeodesicDilationFilter::new()
                    .apply(mk, ms)
                    .expect("coeus geodesic")
            },
            |mk, ms, backend| GrayscaleGeodesicDilationFilter::new().apply_native(mk, ms, backend),
        );
    }

    #[test]
    fn opening_by_reconstruction_matches_coeus() {
        let dims = [4usize, 5, 6];
        assert_coeus_matches_coeus(
            scattered(dims),
            dims,
            |img| {
                OpeningByReconstructionFilter::new(1)
                    .apply(img)
                    .expect("coeus OBR")
            },
            |img, backend| OpeningByReconstructionFilter::new(1).apply_native(img, backend),
        );
    }

    #[test]
    fn closing_by_reconstruction_matches_coeus() {
        let dims = [4usize, 5, 6];
        assert_coeus_matches_coeus(
            scattered(dims),
            dims,
            |img| {
                ClosingByReconstructionFilter::new(1)
                    .apply(img)
                    .expect("coeus CBR")
            },
            |img, backend| ClosingByReconstructionFilter::new(1).apply_native(img, backend),
        );
    }
}
