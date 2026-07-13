//! TransMorph unit tests: end-to-end shape plus finite-difference gradient
//! checks on the differentiable building blocks.
//!
//! The full network's minimum input is `32³` (patch-4 embedding followed by
//! three stride-2 downsamplings), too large for a many-evaluation
//! finite-difference sweep. Backward correctness is therefore verified on the
//! individual autograd sub-graphs — window attention, a shifted Swin block, and
//! the spatial transformer — each sized so its normalization/interpolation is
//! non-vacuous, and a single end-to-end forward confirms the assembled shapes.
//!
//! The gradient oracle reduces each output against a fixed pseudo-random linear
//! functional rather than a bare sum. A bare sum of a projection output is
//! nearly flat in the input (the projection's column sums cancel), so its
//! central-difference estimate is dominated by f32 round-off; a random
//! functional is generically sensitive, giving a well-conditioned oracle.

use super::config::{TransMorphConfig, TransformIntegration};
use super::spatial_transform::SpatialTransformer;
use super::swin::{SwinTransformerBlock, WindowAttention};
use coeus_autograd::{mul, sum, Var};
use coeus_core::SequentialBackend;
use coeus_tensor::Tensor;

type B = SequentialBackend;

/// Deterministic, non-degenerate values in a bounded range.
fn ramp(shape: &[usize]) -> Vec<f32> {
    let n: usize = shape.iter().product();
    (0..n).map(|i| ((i % 17) as f32) / 17.0 - 0.5).collect()
}

fn var(shape: &[usize], data: &[f32], grad: bool) -> Var<f32, B> {
    Var::new(Tensor::from_slice_on(shape, data, &SequentialBackend), grad)
}

/// Reduce `y` against a fixed pseudo-random linear functional `Σ wᵢ yᵢ`.
fn functional(y: &Var<f32, B>) -> Var<f32, B> {
    let shape = y.tensor.shape();
    let n: usize = shape.iter().product();
    let w: Vec<f32> = (0..n)
        .map(|i| (((i.wrapping_mul(2_654_435_761)) % 1000) as f32) / 1000.0 - 0.5)
        .collect();
    let wv = var(shape, &w, false);
    sum(&mul(y, &wv))
}

#[test]
fn test_transmorph_forward_shapes() {
    let config = TransMorphConfig {
        in_channels: 1,
        embed_dim: 8,
        out_channels: 3,
        window_size: 4,
        integration: TransformIntegration::Direct,
        integration_steps: 4,
    };
    let model: super::model::TransMorph<B> = config.init();

    let shape = [1usize, 1, 32, 32, 32];
    let input = var(&shape, &ramp(&shape), false);
    let out = model.forward(&input);

    assert_eq!(out.warped.tensor.shape(), &[1, 1, 32, 32, 32]);
    assert_eq!(out.flow.tensor.shape(), &[1, 3, 32, 32, 32]);
    assert!(
        out.flow.tensor.as_slice().iter().all(|v| v.is_finite()),
        "flow field must be finite"
    );
}

#[test]
fn named_parameters_cover_every_layer_uniquely() {
    let model: super::model::TransMorph<B> = TransMorphConfig::new(1, 8, 3).init();
    let named = model.named_parameters();
    assert_eq!(named.len(), model.parameters().len());
    let names: std::collections::HashSet<_> = named.iter().map(|p| p.name.clone()).collect();
    assert_eq!(names.len(), named.len(), "parameter paths must be unique");
    assert!(names.contains("patch_embed.weight"));
    assert!(names.contains("stage2.0.attention.relative_position_bias_table"));
    assert!(names.contains("flow_conv.bias"));
}

/// Central-difference gradient check of `d(functional(out))/d(x)` versus
/// autograd. `probes` index the input leaf.
///
/// Central differences are `O(h²)`-accurate; a bound of `tol_factor·(1+|g|)`
/// covers truncation plus f32 rounding without masking a defect. A non-vacuity
/// guard rejects an all-zero probed gradient.
fn assert_fd_matches<F>(
    shape: &[usize],
    base: &[f32],
    probes: &[usize],
    h: f32,
    tol_factor: f32,
    forward: F,
) where
    F: Fn(&Var<f32, B>) -> Var<f32, B>,
{
    let input = var(shape, base, true);
    let loss = functional(&forward(&input));
    loss.backward();
    let grad = input.grad().expect("input gradient present");
    let grad = grad.as_slice();

    let mut max_analytic = 0.0f32;
    for &idx in probes {
        let mut plus = base.to_vec();
        let mut minus = base.to_vec();
        plus[idx] += h;
        minus[idx] -= h;
        let fp = functional(&forward(&var(shape, &plus, false)));
        let fm = functional(&forward(&var(shape, &minus, false)));
        let fd = (fp.tensor.as_slice()[0] - fm.tensor.as_slice()[0]) / (2.0 * h);
        let analytic = grad[idx];
        max_analytic = max_analytic.max(analytic.abs());
        let diff = (fd - analytic).abs();
        let tol = tol_factor * (1.0 + analytic.abs());
        assert!(
            diff <= tol,
            "grad mismatch at input[{idx}]: fd={fd}, autograd={analytic}, |Δ|={diff} > {tol}"
        );
    }
    assert!(
        max_analytic > 1e-6,
        "gradient check is vacuous — all probed gradients are ~0 ({max_analytic})"
    );
}

#[test]
fn window_attention_gradient_matches_finite_difference() {
    // One 2³ window, 4 channels, 2 heads: exercises the Q/K/V/proj projections,
    // relative-position-bias gather, batched matmul, and softmax backward.
    let attn = WindowAttention::<B>::new(4, 2, 2, 0xA11CE);
    let shape = [1usize, 8, 4];
    let base = ramp(&shape);
    assert_fd_matches(&shape, &base, &[0, 11, 27], 1.0 / 128.0, 2e-2, |x| {
        attn.forward(x, None)
    });
}

#[test]
fn shifted_swin_block_gradient_matches_finite_difference() {
    // 4³ token volume, window 2, shift 1: exercises LayerNorm, cyclic-shift roll,
    // masked shifted-window attention, window partition/reverse, and the MLP.
    let block = SwinTransformerBlock::<B>::new(4, 2, 2, 1, 2.0, 0xBEEF);
    let shape = [1usize, 4, 4, 4, 4];
    let base = ramp(&shape);
    assert_fd_matches(&shape, &base, &[0, 100, 200], 1.0 / 128.0, 2e-2, |x| {
        block.forward(x)
    });
}

#[test]
fn spatial_transformer_gradient_wrt_flow_matches_finite_difference() {
    // Warp a linear intensity field by a sub-voxel displacement so every sample
    // lands strictly inside a trilinear cell (away from voxel-center kinks),
    // where grid-sample's coordinate gradient is smooth.
    let stn = SpatialTransformer::new();
    let img_shape = [1usize, 1, 6, 6, 6];
    let (d, h, w) = (6usize, 6, 6);
    let denom = |e: usize| (e - 1) as f32;
    let mut img = vec![0.0f32; d * h * w];
    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                img[z * (h * w) + y * w + x] =
                    0.3 * z as f32 / denom(d) + 0.5 * y as f32 / denom(h) + 0.2 * x as f32 / denom(w);
            }
        }
    }
    let image = var(&img_shape, &img, false);

    // Small sub-voxel displacement: samples land off-center but strictly
    // in-bounds. Probes target the interior voxel (z,y,x)=(2,2,2) on each of the
    // three flow channels (offsets 0, 216, 432 into the [1,3,6,6,6] field), well
    // away from the zero-padding boundary.
    let flow_shape = [1usize, 3, 6, 6, 6];
    let flow_base: Vec<f32> = (0..flow_shape.iter().product::<usize>())
        .map(|i| 0.17 + 0.03 * ((i % 4) as f32))
        .collect();
    let interior = 2 * 36 + 2 * 6 + 2; // (z,y,x)=(2,2,2)
    assert_fd_matches(
        &flow_shape,
        &flow_base,
        &[interior, 216 + interior, 432 + interior],
        1.0 / 256.0,
        3e-2,
        |flow| stn.forward(&image, flow),
    );
}
