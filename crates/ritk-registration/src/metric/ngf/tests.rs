use super::fixed_prep::NgfFixedPrep;
use super::scalar::*;
use super::NormalizedGradientField;
use crate::metric::Metric;

use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use ritk_transform::TranslationTransform;

type B = NdArray<f32>;

fn image2d(data: Vec<f32>, shape: [usize; 2]) -> Image<B, 2> {
    let device = Default::default();
    let tensor = Tensor::from_data(TensorData::new(data, Shape::new(shape)), &device);
    Image::new(
        tensor,
        Point::new([0.0, 0.0]),
        Spacing::new([1.0, 1.0]),
        Direction::identity(),
    )
}

fn vertical_edge(w: usize, h: usize, at: usize, sign: f32) -> Vec<f32> {
    let mut v = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            v[y * w + x] = if x < at { 0.0 } else { sign };
        }
    }
    v
}

/// Cross-modal sign invariance: a co-located edge with OPPOSITE contrast
/// (bright→dark vs dark→bright) scores exactly the same as an identical-
/// contrast edge — the squared gradient dot product makes a bright-CT /
/// dark-MR boundary register just like a same-sign one.
#[test]
fn ngf_is_sign_invariant() {
    let (w, h) = (8usize, 8usize);
    let f = vertical_edge(w, h, 4, 1.0);
    let same = ngf_scalar(&f, &f, &[h, w], None, None);
    let opposite = ngf_scalar(&f, &vertical_edge(w, h, 4, -1.0), &[h, w], None, None);
    assert!(same > 0.0, "self-NGF should be positive, got {same}");
    assert!(
        (same - opposite).abs() < 1e-4,
        "opposite contrast must score equal: same {same} vs opposite {opposite}"
    );
}

/// NGF of perpendicular edges (uncorrelated orientation) is well below that of
/// aligned edges — the property that lets NGF recover a rotation that intensity
/// MI cannot.
#[test]
fn aligned_beats_perpendicular() {
    let (w, h) = (8usize, 8usize);
    let vert = vertical_edge(w, h, 4, 1.0); // gradient in x
    let mut horiz = vec![0.0f32; w * h]; // gradient in y
    for y in 0..h {
        for x in 0..w {
            horiz[y * w + x] = if y < 4 { 0.0 } else { 1.0 };
        }
    }
    let aligned = ngf_scalar(&vert, &vert, &[h, w], None, None);
    let perpendicular = ngf_scalar(&vert, &horiz, &[h, w], None, None);
    assert!(
        aligned > perpendicular + 0.1,
        "aligned {aligned} should exceed perpendicular {perpendicular}"
    );
}

/// Center weighting makes a CENTRAL edge mismatch dominate the metric over a
/// stronger PERIPHERAL one — the skull-domination fix. Fixed and moving agree
/// at the periphery but disagree centrally; the uniform NGF barely drops
/// (periphery dominates), while the center-Gaussian-weighted NGF drops sharply
/// because the central disagreement now carries the weight.
#[test]
fn center_weight_emphasizes_central_mismatch() {
    let (w, h) = (32usize, 32usize);
    // Fixed: strong peripheral vertical edges (cols 2 and 29) + a central one (col 16).
    let mut f = vec![0.0f32; w * h];
    let mut m = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let periph = if x == 2 || x == 29 { 1.0 } else { 0.0 };
            f[y * w + x] = periph + if x == 16 { 1.0 } else { 0.0 };
            // Moving matches the periphery but the central edge is displaced
            // (col 20 instead of 16) — a purely central disagreement.
            m[y * w + x] = periph + if x == 20 { 1.0 } else { 0.0 };
        }
    }
    let shape = [h, w];
    let spacing = [1.0_f64, 1.0_f64];
    let uniform = ngf_scalar(&f, &m, &shape, None, None);
    let wfield = center_gaussian_weight_field(&shape, None, &spacing, 0.4);
    let weighted = ngf_scalar(&f, &m, &shape, None, Some(&wfield));
    // The central mismatch costs MORE under center weighting: weighted NGF
    // (similarity) is strictly lower than the periphery-dominated uniform NGF.
    assert!(
        weighted < uniform - 1e-3,
        "center weighting should penalise the central mismatch: \
         weighted {weighted} vs uniform {uniform}"
    );
    // Weight field is a valid Gaussian: positive at the centre, ~0 at corners.
    let center = wfield[(h / 2) * w + w / 2];
    assert!(center > 0.9, "center weight {center} should be near 1");
    assert!(wfield[0] < center, "corner weight {} < center", wfield[0]);
}

/// End-to-end through the `Metric` trait: registering the moving edge onto the
/// fixed edge (identity) gives a lower loss than a translation that pulls the
/// edges apart. The edge varies along the d1 (x/column) axis; with an identity
/// direction the corresponding WORLD component is index 1, so the displacing
/// translation is `[0, dx]` (world component 1 = x).
#[test]
fn metric_loss_lower_when_aligned() {
    let (w, h) = (16usize, 16usize);
    let img = image2d(vertical_edge(w, h, 8, 1.0), [h, w]);
    let metric = NormalizedGradientField::new();
    let device = Default::default();
    let loss = |dx: f32| {
        let t = TranslationTransform::<B, 2>::new(Tensor::from_data(
            TensorData::new(vec![0.0_f32, dx], [2]),
            &device,
        ));
        metric
            .forward(&img, &img, &t)
            .into_data()
            .to_vec::<f32>()
            .unwrap()[0]
    };
    let aligned = loss(0.0);
    let shifted = loss(4.0);
    assert!(
        aligned < 0.0,
        "aligned loss should be negative, got {aligned}"
    );
    assert!(
        aligned < shifted,
        "aligned loss {aligned} should be below shifted {shifted}"
    );
}

/// The stochastic-sample NGF path computes the SAME value as the dense path
/// when the subset is complete (every voxel sampled) — verifying the
/// neighbour-gather + finite-difference gradient and η over the sample subset
/// reproduce the dense metric. A strided half-subset stays a close estimate.
#[test]
fn sampled_ngf_matches_dense() {
    let (w, h) = (24usize, 24usize);
    let img = image2d(vertical_edge(w, h, 12, 1.0), [h, w]);
    let device = Default::default();
    let ident = TranslationTransform::<B, 2>::new(Tensor::from_data(
        TensorData::new(vec![0.0_f32, 0.0], [2]),
        &device,
    ));
    let dense = NgfFixedPrep::<B, 2>::new(&img, None, None).eval(&img, &ident);
    let full = NgfFixedPrep::<B, 2>::new_sampled(&img, None, None, w * h).eval(&img, &ident);
    let half = NgfFixedPrep::<B, 2>::new_sampled(&img, None, None, w * h / 2).eval(&img, &ident);
    assert!(
        dense > 0.0,
        "dense self-NGF should be positive, got {dense}"
    );
    assert!(
        (dense - full).abs() < 1e-3,
        "full-subset sampled {full} must equal dense {dense}"
    );
    assert!(
        (dense - half).abs() < 0.15 * dense,
        "half-subset sampled {half} should approximate dense {dense}"
    );
}
