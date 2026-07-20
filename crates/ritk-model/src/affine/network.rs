//! Affine registration CNN, Coeus-native.
//!
//! A five-stage `Conv3d → InstanceNorm3d → ReLU` feature extractor followed by
//! global average pooling and a linear head predicting the 12 parameters of a
//! `3×4` affine matrix. The head output is offset by the flattened identity
//! transform so an untrained network starts near the identity map (the standard
//! spatial-transformer initialization).
//!
//! Built on [`coeus_nn`] layers over [`coeus_autograd::Var`]; gradients flow to
//! every layer parameter through the autograd graph. No Coeus tensors, modules,
//! or backends cross this boundary.

use coeus_autograd::{add, relu, reshape, Parameter, Var};
use coeus_core::{Float, MoiraiBackend};
use coeus_nn::module::Module;
use coeus_nn::{Conv3d, GlobalAvgPool3d, InstanceNorm3d, Linear};
use coeus_ops::BackendOps;
use coeus_tensor::Tensor;

/// Number of predicted affine parameters (a flattened `3×4` matrix).
const AFFINE_PARAMS: usize = 12;
/// Isotropic convolution kernel side length.
const KERNEL: usize = 3;
/// Isotropic convolution stride (halves each spatial extent per stage).
const STRIDE: usize = 2;
/// Isotropic zero-padding (`(KERNEL - 1) / 2`, keeping the halving exact).
const PADDING: usize = 1;
/// Convolution dilation.
const DILATION: usize = 1;
/// Input channel count (concatenated moving + fixed volumes).
const IN_CHANNELS: usize = 2;
/// InstanceNorm numerical-stability constant.
const NORM_EPS: f64 = 1e-5;
/// Base seed for deterministic Kaiming initialization; the odd increment
/// (golden-ratio constant) decorrelates successive layers' weight draws.
const INIT_SEED: u64 = 0x1234_5678;
/// Golden-ratio odd increment for per-layer seed advancement.
const SEED_STEP: u64 = 0x9E37_79B9_7F4A_7C15;

/// Row-major flattened identity of the `3×4` affine matrix `[I₃ | 0]`.
///
/// Added to the linear-head output so the network initializes at the identity
/// warp. The concrete values are structural (the affine identity), not a tuned
/// constant.
#[inline]
fn identity_affine<T: Float>() -> [T; AFFINE_PARAMS] {
    let (o, z) = (T::one(), T::zero());
    [o, z, z, z, z, o, z, z, z, z, o, z]
}

/// Configuration for [`AffineNetwork`]: per-stage output channel counts.
#[derive(Debug, Clone)]
pub struct AffineNetworkConfig {
    /// Output channels for each of the five convolution stages.
    pub channels: Vec<usize>,
}

impl Default for AffineNetworkConfig {
    fn default() -> Self {
        Self {
            channels: vec![16, 32, 64, 128, 256],
        }
    }
}

impl AffineNetworkConfig {
    /// Instantiate an [`AffineNetwork`] over element type `T` and backend `B`.
    ///
    /// Convolution and linear weights are Kaiming-uniform-initialized (the
    /// fan-in-scaled scheme the original Coeus model relied on), keeping the
    /// per-filter response non-degenerate; biases start at zero.
    ///
    /// # Panics
    /// Panics if `channels` does not contain exactly five entries.
    pub fn init<T, B>(&self) -> AffineNetwork<T, B>
    where
        T: Float + coeus_leto::RandomScalar,
        B: BackendOps<T> + Default,
    {
        assert_eq!(
            self.channels.len(),
            5,
            "AffineNetworkConfig requires exactly five stage channel counts"
        );
        let c = &self.channels;
        let mut seed = INIT_SEED;
        let mut make_conv = |in_ch: usize, out_ch: usize| {
            let mut layer =
                Conv3d::<T, B>::with_params(in_ch, out_ch, KERNEL, STRIDE, PADDING, DILATION, true);
            let fan_in = in_ch * KERNEL * KERNEL * KERNEL;
            seed = seed.wrapping_add(SEED_STEP);
            coeus_nn::init::kaiming_uniform_with_seed(&mut layer.weight, fan_in, seed);
            layer
        };
        let conv1 = make_conv(IN_CHANNELS, c[0]);
        let conv2 = make_conv(c[0], c[1]);
        let conv3 = make_conv(c[1], c[2]);
        let conv4 = make_conv(c[2], c[3]);
        let conv5 = make_conv(c[3], c[4]);

        let mut fc = Linear::new(c[4], AFFINE_PARAMS, true);
        seed = seed.wrapping_add(SEED_STEP);
        coeus_nn::init::kaiming_uniform_with_seed(&mut fc.weight, c[4], seed);

        AffineNetwork {
            conv1,
            bn1: InstanceNorm3d::new(c[0], NORM_EPS),
            conv2,
            bn2: InstanceNorm3d::new(c[1], NORM_EPS),
            conv3,
            bn3: InstanceNorm3d::new(c[2], NORM_EPS),
            conv4,
            bn4: InstanceNorm3d::new(c[3], NORM_EPS),
            conv5,
            bn5: InstanceNorm3d::new(c[4], NORM_EPS),
            fc,
        }
    }
}

/// Affine-parameter regression CNN.
///
/// `forward` maps a `[B, 2, D, H, W]` moving/fixed volume pair to `[B, 12]`
/// affine parameters (identity-offset).
#[derive(Clone)]
pub struct AffineNetwork<T: Float, B: BackendOps<T> + Default = MoiraiBackend> {
    conv1: Conv3d<T, B>,
    bn1: InstanceNorm3d<T, B>,
    conv2: Conv3d<T, B>,
    bn2: InstanceNorm3d<T, B>,
    conv3: Conv3d<T, B>,
    bn3: InstanceNorm3d<T, B>,
    conv4: Conv3d<T, B>,
    bn4: InstanceNorm3d<T, B>,
    conv5: Conv3d<T, B>,
    bn5: InstanceNorm3d<T, B>,
    fc: Linear<T, B>,
}

/// Append `module`'s named parameters under `prefix` to `out`.
fn extend_named<T, B, M>(out: &mut Vec<Parameter<T, B>>, prefix: &str, module: &M)
where
    T: Float,
    B: BackendOps<T> + Default,
    M: Module<T, B>,
{
    out.extend(
        module
            .named_parameters()
            .into_iter()
            .map(|parameter| parameter.with_prefix(prefix)),
    );
}

impl<T, B> Module<T, B> for AffineNetwork<T, B>
where
    T: Float,
    B: BackendOps<T> + Default,
{
    fn parameters(&self) -> Vec<Var<T, B>> {
        let mut params = Vec::new();
        for module in self.stages() {
            params.extend(module.parameters());
        }
        params.extend(self.fc.parameters());
        params
    }

    fn named_parameters(&self) -> Vec<Parameter<T, B>> {
        let mut named = Vec::new();
        extend_named(&mut named, "conv1", &self.conv1);
        extend_named(&mut named, "bn1", &self.bn1);
        extend_named(&mut named, "conv2", &self.conv2);
        extend_named(&mut named, "bn2", &self.bn2);
        extend_named(&mut named, "conv3", &self.conv3);
        extend_named(&mut named, "bn3", &self.bn3);
        extend_named(&mut named, "conv4", &self.conv4);
        extend_named(&mut named, "bn4", &self.bn4);
        extend_named(&mut named, "conv5", &self.conv5);
        extend_named(&mut named, "bn5", &self.bn5);
        extend_named(&mut named, "fc", &self.fc);
        named
    }

    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        let mut x = relu(&self.bn1.forward(&self.conv1.forward(input)));
        x = relu(&self.bn2.forward(&self.conv2.forward(&x)));
        x = relu(&self.bn3.forward(&self.conv3.forward(&x)));
        x = relu(&self.bn4.forward(&self.conv4.forward(&x)));
        x = relu(&self.bn5.forward(&self.conv5.forward(&x)));

        // Global average pool [B, C, D, H, W] → [B, C, 1, 1, 1] → [B, C].
        let pooled = GlobalAvgPool3d::<T, B>::new().forward(&x);
        let shape = pooled.tensor.shape();
        let (batch, channels) = (shape[0], shape[1]);
        let pooled = reshape(&pooled, [batch, channels]);

        let predicted = self.fc.forward(&pooled);

        // Offset by the identity affine so the untrained map is near-identity.
        let identity = identity_affine::<T>();
        let mut identity_batch = Vec::with_capacity(batch * AFFINE_PARAMS);
        for _ in 0..batch {
            identity_batch.extend_from_slice(&identity);
        }
        let identity = Var::new(
            Tensor::from_slice_on([batch, AFFINE_PARAMS], &identity_batch, &B::default()),
            false,
        );
        add(&predicted, &identity)
    }

    fn load_parameters(&mut self, params: &[Var<T, B>]) {
        let mut offset = 0;
        let mut load = |module: &mut dyn Module<T, B>| {
            let count = module.parameters().len();
            module.load_parameters(&params[offset..offset + count]);
            offset += count;
        };
        load(&mut self.conv1);
        load(&mut self.bn1);
        load(&mut self.conv2);
        load(&mut self.bn2);
        load(&mut self.conv3);
        load(&mut self.bn3);
        load(&mut self.conv4);
        load(&mut self.bn4);
        load(&mut self.conv5);
        load(&mut self.bn5);
        load(&mut self.fc);
    }
}

impl<T, B> AffineNetwork<T, B>
where
    T: Float,
    B: BackendOps<T> + Default,
{
    /// The five `conv → norm` stages as trait objects, in forward order.
    fn stages(&self) -> [&dyn Module<T, B>; 10] {
        [
            &self.conv1,
            &self.bn1,
            &self.conv2,
            &self.bn2,
            &self.conv3,
            &self.bn3,
            &self.conv4,
            &self.bn4,
            &self.conv5,
            &self.bn5,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;

    type B = SequentialBackend;

    fn sequential_input(shape: [usize; 5]) -> Var<f32, B> {
        let n: usize = shape.iter().product();
        // Deterministic, non-degenerate values in a bounded range.
        let data: Vec<f32> = (0..n).map(|i| ((i % 17) as f32) / 17.0 - 0.5).collect();
        Var::new(
            Tensor::from_slice_on(shape, &data, &SequentialBackend),
            true,
        )
    }

    #[test]
    fn forward_output_shape_is_batch_by_12() {
        let net = AffineNetworkConfig::default().init::<f32, B>();
        let input = sequential_input([1, 2, 8, 8, 8]);
        let out = net.forward(&input);
        assert_eq!(out.tensor.shape(), &[1, 12]);
    }

    #[test]
    fn forward_single_batch_produces_finite_12_parameters() {
        // InstanceNorm normalizes per-instance, so batch_size = 1 is valid and
        // must not produce NaN (which per-batch variance would for BatchNorm).
        let net = AffineNetworkConfig::default().init::<f32, B>();
        let input = sequential_input([1, 2, 8, 8, 8]);
        let out = net.forward(&input);
        let vals = out.tensor.as_slice();
        assert_eq!(vals.len(), 12, "output must carry 12 affine parameters");
        for (i, &v) in vals.iter().enumerate() {
            assert!(
                v.is_finite(),
                "param[{i}] must be finite for a batch of 1: {v}"
            );
        }
    }

    #[test]
    fn named_parameters_cover_every_layer_uniquely() {
        let net = AffineNetworkConfig::default().init::<f32, B>();
        let named = net.named_parameters();
        // 5 convs (weight+bias) + 5 norms (weight+bias) + fc (weight+bias).
        assert_eq!(named.len(), 22);
        assert_eq!(named.len(), net.parameters().len());
        let names: std::collections::HashSet<_> = named.iter().map(|p| p.name.clone()).collect();
        assert_eq!(names.len(), named.len(), "parameter paths must be unique");
        assert!(names.contains("conv1.weight"));
        assert!(names.contains("bn5.bias"));
        assert!(names.contains("fc.weight"));
    }

    /// Finite-difference gradient check on the network input.
    ///
    /// Perturbing the input leaf and comparing the central-difference estimate
    /// of `d(sum(output))/d(input)` against the autograd gradient validates the
    /// full forward+backward chain (conv → norm → relu → pool → linear → add).
    ///
    /// The `36³` input keeps every InstanceNorm's spatial extent `> 1` through
    /// all five stages (`36→18→9→5→3→2`); a smaller volume would collapse a
    /// stage to a single voxel, where zero variance makes the norm output
    /// constant and the gradient vanish (a genuine property of the architecture,
    /// not a defect — but it would make this check vacuous).
    ///
    /// Tolerance: central differences are `O(h²)`-accurate; with `h = 2⁻⁷` and
    /// bounded inputs the truncation error is `~10⁻⁴`, and the per-element
    /// gradients here are `O(1)`. A bound of `2e-2·(1+|g|)` accommodates
    /// truncation plus f32 rounding across the deep chain without masking a real
    /// defect. A non-vacuity guard asserts the probed gradients are not all zero.
    #[test]
    fn finite_difference_gradient_matches_autograd() {
        // Narrow channels keep the deep 36³ chain tractable in a debug build
        // while exercising the identical conv → norm → relu → pool → linear →
        // add graph the gradient check validates (channel width does not change
        // the differentiated computation).
        let net = AffineNetworkConfig {
            channels: vec![2, 3, 4, 5, 6],
        }
        .init::<f32, B>();
        let shape = [1usize, 2, 36, 36, 36];
        let n: usize = shape.iter().product();
        let base: Vec<f32> = (0..n).map(|i| ((i % 17) as f32) / 17.0 - 0.5).collect();

        let input = Var::new(
            Tensor::from_slice_on(shape, &base, &SequentialBackend),
            true,
        );
        let out = net.forward(&input);
        let loss = coeus_autograd::sum(&out);
        loss.backward();
        let grad = input.grad().expect("input gradient present");
        let grad = grad.as_slice();

        let h = 1.0f32 / 128.0;
        let probes = [0usize, 6151, 40000];
        let mut max_analytic = 0.0f32;
        for &idx in &probes {
            let mut plus = base.clone();
            let mut minus = base.clone();
            plus[idx] += h;
            minus[idx] -= h;
            let fp = coeus_autograd::sum(&net.forward(&Var::new(
                Tensor::from_slice_on(shape, &plus, &SequentialBackend),
                false,
            )));
            let fm = coeus_autograd::sum(&net.forward(&Var::new(
                Tensor::from_slice_on(shape, &minus, &SequentialBackend),
                false,
            )));
            let fd = (fp.tensor.as_slice()[0] - fm.tensor.as_slice()[0]) / (2.0 * h);
            let analytic = grad[idx];
            max_analytic = max_analytic.max(analytic.abs());
            let diff = (fd - analytic).abs();
            let tol = 2e-2 * (1.0 + analytic.abs());
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
}
