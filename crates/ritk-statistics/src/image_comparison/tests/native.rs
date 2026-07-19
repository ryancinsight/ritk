//! Coeus-native image comparison metrics: analytical oracles plus differential
//! parity against the Burn path.
//!
//! ## Tolerances
//! - `hausdorff_distance`, `mean_surface_distance`, `ssim`, `similarity_index`:
//!   the native adapter and the Burn adapter delegate to the **same** host core
//!   (`hausdorff_from_flat` / `msd_from_flat` / `ssim_from_slices` /
//!   `similarity_index_from_slices`), so results are bitwise identical — asserted
//!   with `PARITY_EXACT = 0.0`.
//! - `dice_coefficient`, `psnr`: the Burn path reduces in `f32`, the native core
//!   in `f64`, so the two agree only to the `f32` accumulation error. For the
//!   `n ≤ 64` fixtures here the bound is `n · ε_f32 · max|term| ≈ 64 · 6e-8 · 1 ≈
//!   4e-6`; `PARITY_DIFF = 1e-4` covers it with margin.

use super::{make_image, F32_TOL};
use crate::image_comparison as burn_metrics;
use crate::image_comparison::native as native_metrics;
use coeus_core::MoiraiBackend;
use ritk_image::Image as NativeImage;
use ritk_spatial::{Direction, Point, Spacing};

/// Native adapter and Burn adapter share the identical host core → bitwise equal.
const PARITY_EXACT: f32 = 0.0;
/// f64-core vs f32-reduction differential bound for the fixtures used here.
const PARITY_DIFF: f32 = 1e-4;

fn make_native<const D: usize>(
    data: Vec<f32>,
    dims: [usize; D],
) -> NativeImage<f32, MoiraiBackend, D> {
    NativeImage::from_flat(
        data,
        dims,
        Point::new([0.0_f64; D]),
        Spacing::new([1.0_f64; D]),
        Direction::identity(),
    )
    .expect("native image construction")
}

// ── Dice ────────────────────────────────────────────────────────────────────

#[test]
fn dice_identical_masks_is_one() {
    let img = make_native(vec![1.0f32; 27], [3, 3, 3]);
    let dice = native_metrics::dice_coefficient(&img, &img).unwrap();
    assert!(
        (dice - 1.0).abs() < F32_TOL,
        "identical masks -> Dice = 1.0, got {dice}"
    );
}

#[test]
fn dice_both_empty_is_one() {
    let z = make_native(vec![0.0; 8], [8]);
    assert!((native_metrics::dice_coefficient(&z, &z).unwrap() - 1.0).abs() < F32_TOL);
}

#[test]
fn dice_known_half_overlap() {
    // pred = {0,1,2,3}, gt = {2,3,4,5}: |∩| = 2, |P| = |G| = 4 → 2*2/8 = 0.5.
    let pred = vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let gt = vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0];
    let dice =
        native_metrics::dice_coefficient(&make_native(pred, [8]), &make_native(gt, [8])).unwrap();
    assert!((dice - 0.5).abs() < F32_TOL, "Dice = 0.5, got {dice}");
}

#[test]
fn dice_matches_burn() {
    let pred = vec![1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
    let gt = vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0];
    let bd = burn_metrics::dice_coefficient(
        &make_image(pred.clone(), [8]),
        &make_image(gt.clone(), [8]),
    )
    .expect("matched image shapes");
    let nd =
        native_metrics::dice_coefficient(&make_native(pred, [8]), &make_native(gt, [8])).unwrap();
    assert!((bd - nd).abs() <= PARITY_DIFF, "dice burn={bd} native={nd}");
}

#[test]
fn dice_length_mismatch_errors() {
    let a = make_native(vec![1.0; 4], [4]);
    let b = make_native(vec![1.0; 8], [8]);
    assert!(native_metrics::dice_coefficient(&a, &b).is_err());
}

// ── Similarity index ─────────────────────────────────────────────────────────

#[test]
fn similarity_index_matches_burn_exactly() {
    let a = vec![0.0, 1.0, 1.0, 0.0, 2.0, 2.0];
    let b = vec![0.0, 1.0, 0.0, 1.0, 2.0, 0.0];
    let bs =
        burn_metrics::similarity_index(&make_image(a.clone(), [6]), &make_image(b.clone(), [6]))
            .expect("matched image shapes");
    let ns = native_metrics::similarity_index(&make_native(a, [6]), &make_native(b, [6])).unwrap();
    assert!((bs - ns).abs() <= PARITY_EXACT, "SI burn={bs} native={ns}");
    // 4/7 analytical oracle.
    assert!((ns - 4.0 / 7.0).abs() < F32_TOL, "SI = 4/7, got {ns}");
}

// ── PSNR ────────────────────────────────────────────────────────────────────

#[test]
fn psnr_identical_is_infinity() {
    let img = make_native(vec![1.0, 2.0, 3.0, 4.0, 5.0], [5]);
    let p = native_metrics::psnr(&img, &img, 255.0).unwrap();
    assert!(p.is_infinite() && p > 0.0, "identical -> +inf, got {p}");
}

#[test]
fn psnr_known_value() {
    // constant error 0.1 with MAX=1: MSE = 0.01, PSNR = 10*log10(1/0.01) = 20 dB.
    let img = make_native(vec![0.0, 0.0], [2]);
    let reference = make_native(vec![0.1, 0.1], [2]);
    let p = native_metrics::psnr(&img, &reference, 1.0).unwrap();
    assert!((p - 20.0).abs() < 1e-3, "PSNR ~= 20 dB, got {p}");
}

#[test]
fn psnr_matches_burn() {
    let a = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let b = vec![0.2, 0.9, 2.3, 2.7, 4.1, 5.4, 5.6, 7.2];
    let bp = burn_metrics::psnr(
        &make_image(a.clone(), [8]),
        &make_image(b.clone(), [8]),
        10.0,
    );
    let np = native_metrics::psnr(&make_native(a, [8]), &make_native(b, [8]), 10.0).unwrap();
    assert!((bp - np).abs() <= PARITY_DIFF, "psnr burn={bp} native={np}");
}

// ── SSIM ────────────────────────────────────────────────────────────────────

#[test]
fn ssim_identical_is_one() {
    let img = make_native(vec![1.0, 2.0, 3.0, 4.0, 5.0], [5]);
    let s = native_metrics::ssim(&img, &img, 5.0).unwrap();
    assert!(
        (s - 1.0).abs() < F32_TOL,
        "identical -> SSIM = 1.0, got {s}"
    );
}

#[test]
fn ssim_matches_burn_exactly() {
    let a = vec![1.0, 3.0, 5.0, 7.0];
    let b = vec![2.0, 4.0, 6.0, 8.0];
    let bs = burn_metrics::ssim(
        &make_image(a.clone(), [4]),
        &make_image(b.clone(), [4]),
        10.0,
    );
    let ns = native_metrics::ssim(&make_native(a, [4]), &make_native(b, [4]), 10.0).unwrap();
    assert!(
        (bs - ns).abs() <= PARITY_EXACT,
        "SSIM burn={bs} native={ns}"
    );
}

// ── Surface distances ────────────────────────────────────────────────────────

#[test]
fn hausdorff_identical_is_zero() {
    let mut mask = vec![0.0f32; 27];
    mask[13] = 1.0; // single interior voxel: its own boundary
    let img = make_native(mask, [3, 3, 3]);
    let hd = native_metrics::hausdorff_distance(&img, &img, &[1.0, 1.0, 1.0]).unwrap();
    assert!(hd.abs() < F32_TOL, "identical masks -> HD = 0, got {hd}");
}

#[test]
fn hausdorff_matches_burn_exactly() {
    let mut pred = vec![0.0f32; 27];
    let mut gt = vec![0.0f32; 27];
    for v in pred.iter_mut().take(9) {
        *v = 1.0; // first z-slice
    }
    for v in gt.iter_mut().take(18).skip(9) {
        *v = 1.0; // second z-slice
    }
    let spacing = [1.0, 1.0, 1.0];
    let bh = burn_metrics::hausdorff_distance(
        &make_image(pred.clone(), [3, 3, 3]),
        &make_image(gt.clone(), [3, 3, 3]),
        &spacing,
    );
    let nh = native_metrics::hausdorff_distance(
        &make_native(pred, [3, 3, 3]),
        &make_native(gt, [3, 3, 3]),
        &spacing,
    )
    .unwrap();
    assert!((bh - nh).abs() <= PARITY_EXACT, "HD burn={bh} native={nh}");
}

#[test]
fn mean_surface_distance_matches_burn_exactly() {
    let mut pred = vec![0.0f32; 27];
    let mut gt = vec![0.0f32; 27];
    for v in pred.iter_mut().take(9) {
        *v = 1.0;
    }
    for v in gt.iter_mut().take(18).skip(9) {
        *v = 1.0;
    }
    let spacing = [1.0, 1.0, 1.0];
    let bm = burn_metrics::mean_surface_distance(
        &make_image(pred.clone(), [3, 3, 3]),
        &make_image(gt.clone(), [3, 3, 3]),
        &spacing,
    );
    let nm = native_metrics::mean_surface_distance(
        &make_native(pred, [3, 3, 3]),
        &make_native(gt, [3, 3, 3]),
        &spacing,
    )
    .unwrap();
    assert!((bm - nm).abs() <= PARITY_EXACT, "MSD burn={bm} native={nm}");
}
