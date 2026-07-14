//! Selective State Space (S6) module, Coeus-native.
//!
//! Coeus-native reimplementation of the Mamba/S6 selective state space block
//! (`SelectiveStateSpace`, still Burn-typed for the
//! registration consumer). Input-dependent parameters `Δ`, `B`, `C` are
//! projected from the input; the discrete-time linear recurrence
//! `h_t = Ā_t ⊙ h_{t-1} + B̄_t·x_t` is evaluated by the differentiable
//! [`coeus_autograd::selective_scan`] primitive, with the discretization
//! `Ā = exp(Δ·A)`, `B̄ = Δ·B` and the output projection `y_t = (C_t ⊙ h_t)`
//! summed over the state axis composing around it from element-wise autograd
//! ops. Gradients flow to every projection weight, to `A` (via `a_log`), and to
//! the skip parameter `D`.
//!
//! Built on [`coeus_nn`]/[`coeus_autograd`] over [`coeus_autograd::Var`]; no
//! Burn tensors, modules, or backends cross this boundary. Concrete `f32`,
//! matching the [`coeus_autograd::selective_scan`]/interpolation subsystem.

use coeus_autograd::{
    exp, mul, neg, permute, reshape, selective_scan, sigmoid, slice, softplus, sum_axis, unsqueeze,
    Parameter, Var,
};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_nn::module::Module;
use coeus_nn::Linear;
use coeus_ops::BackendOps;
use coeus_tensor::Tensor;

/// Base seed for deterministic Kaiming initialization; the odd increment
/// (golden-ratio constant) decorrelates successive projections' weight draws.
const INIT_SEED: u64 = 0x5342_4D53; // "SSMS"
/// Golden-ratio odd increment for per-layer seed advancement.
const SEED_STEP: u64 = 0x9E37_79B9_7F4A_7C15;

/// Configuration for [`SelectiveStateSpace`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectiveStateSpaceConfig {
    /// Input channel dimension.
    pub input_dim: usize,
    /// Output channel dimension (typically `input_dim * expand_factor`).
    pub output_dim: usize,
    /// State dimension `N` of the SSM.
    pub state_dim: usize,
    /// Expansion factor for the inner (hidden) dimension.
    pub expand_factor: usize,
    /// Rank for the low-rank `Δ` parameterization.
    pub dt_rank: usize,
}

impl SelectiveStateSpaceConfig {
    /// Configuration with `input_dim`/`output_dim` and the paper defaults
    /// (`state_dim = 16`, `expand_factor = 2`, `dt_rank = 16`).
    #[must_use]
    pub fn new_with_dims(input_dim: usize, output_dim: usize) -> Self {
        Self {
            input_dim,
            output_dim,
            state_dim: 16,
            expand_factor: 2,
            dt_rank: 16,
        }
    }

    /// Instantiate a [`SelectiveStateSpace`] over backend `B`.
    #[must_use]
    pub fn init<B>(&self) -> SelectiveStateSpace<B>
    where
        B: Backend + BackendOps<f32> + Default,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
    {
        SelectiveStateSpace::new(self, INIT_SEED)
    }
}

/// Selective State Space (S6) block.
///
/// `forward` maps a `[batch, seq, input_dim]` sequence to
/// `[batch, seq, output_dim]`; [`SelectiveStateSpace::forward_3d`] adapts a
/// `[batch, channels, D, H, W]` volume by flattening the spatial axes into the
/// sequence axis.
#[derive(Clone)]
pub struct SelectiveStateSpace<B: Backend + BackendOps<f32> + Default> {
    in_proj: Linear<f32, B>,
    out_proj: Linear<f32, B>,
    dt_in_proj: Linear<f32, B>,
    dt_proj: Linear<f32, B>,
    b_proj: Linear<f32, B>,
    c_proj: Linear<f32, B>,
    /// Log-parameterized state matrix `A = -exp(a_log)`, flattened
    /// `[inner_dim * state_dim]`.
    a_log: Var<f32, B>,
    /// Skip-connection gain `D`, `[inner_dim]`.
    d: Var<f32, B>,
    input_dim: usize,
    output_dim: usize,
    state_dim: usize,
    expand_factor: usize,
}

impl<B> SelectiveStateSpace<B>
where
    B: Backend + BackendOps<f32> + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    /// Construct with Kaiming-uniform projection weights (fan-in scaled, the
    /// scheme the original Burn model relied on), zero biases, HiPPO-real
    /// `a_log` initialization (`A[n] = -(n+1)`, i.e. `a_log[n] = ln(n+1)`), and
    /// `D = 1`. `seed` seeds the deterministic weight draws.
    #[must_use]
    pub fn new(config: &SelectiveStateSpaceConfig, seed: u64) -> Self {
        let backend = B::default();
        let inner_dim = config.input_dim * config.expand_factor;

        let mut current = seed;
        let mut make_linear = |in_features: usize, out_features: usize| {
            let mut layer = Linear::<f32, B>::new(in_features, out_features, true);
            current = current.wrapping_add(SEED_STEP);
            coeus_nn::init::kaiming_uniform_with_seed(&mut layer.weight, in_features, current);
            layer
        };

        let in_proj = make_linear(config.input_dim, inner_dim * 2);
        let out_proj = make_linear(inner_dim, config.output_dim);
        let dt_in_proj = make_linear(inner_dim, config.dt_rank);
        let dt_proj = make_linear(config.dt_rank, inner_dim);
        let b_proj = make_linear(inner_dim, config.state_dim);
        let c_proj = make_linear(inner_dim, config.state_dim);

        // HiPPO-real A: A[i] = -(n+1), n = i mod state_dim, stored as its log.
        let a_init: Vec<f32> = (0..inner_dim * config.state_dim)
            .map(|i| ((i % config.state_dim) as f32 + 1.0).ln())
            .collect();
        let a_log = Var::new(
            Tensor::from_slice_on([inner_dim * config.state_dim], &a_init, &backend),
            true,
        );
        let d = Var::new(Tensor::ones_on([inner_dim], &backend), true);

        Self {
            in_proj,
            out_proj,
            dt_in_proj,
            dt_proj,
            b_proj,
            c_proj,
            a_log,
            d,
            input_dim: config.input_dim,
            output_dim: config.output_dim,
            state_dim: config.state_dim,
            expand_factor: config.expand_factor,
        }
    }

    /// Selective-scan forward for a `[batch, seq, input_dim]` sequence.
    ///
    /// # Panics
    /// If `input` is not rank-3 or its last axis is not `input_dim`.
    #[must_use]
    pub fn forward(&self, input: &Var<f32, B>) -> Var<f32, B> {
        let sh = input.tensor.shape();
        assert_eq!(sh.len(), 3, "SelectiveStateSpace: input must be rank-3");
        assert_eq!(
            sh[2], self.input_dim,
            "SelectiveStateSpace: input last axis must equal input_dim"
        );
        let (batch, seq) = (sh[0], sh[1]);
        let inner = self.input_dim * self.expand_factor;

        // Input projection, split into the SSM branch `x` and the gating
        // residual (channels `[0, inner)` and `[inner, 2·inner)`).
        let proj = self.in_proj.forward(input);
        let x = slice(&proj, &[(0, batch), (0, seq), (0, inner)]);
        let residual = slice(&proj, &[(0, batch), (0, seq), (inner, inner * 2)]);

        // Input-dependent Δ: low-rank projection then softplus (Δ > 0).
        let dt = softplus(&self.dt_proj.forward(&self.dt_in_proj.forward(&x)));
        let b = self.b_proj.forward(&x); // [batch, seq, state]
        let c = self.c_proj.forward(&x); // [batch, seq, state]

        // A = -exp(a_log) reshaped to [inner, state].
        let a = reshape(&neg(&exp(&self.a_log)), [inner, self.state_dim]);

        // Discretize and assemble the scan inputs, all broadcast to
        // [batch, seq, inner, state] so `selective_scan` sees matched shapes.
        let dt_exp = unsqueeze(&dt, 3); // [batch, seq, inner, 1]
        let a_exp = unsqueeze(&unsqueeze(&a, 0), 0); // [1, 1, inner, state]
        let a_bar = exp(&mul(&dt_exp, &a_exp)); // Ā = exp(Δ·A)

        let b_exp = unsqueeze(&b, 2); // [batch, seq, 1, state]
        let b_bar = mul(&dt_exp, &b_exp); // B̄ = Δ·B
        let x_exp = unsqueeze(&x, 3); // [batch, seq, inner, 1]
        let u = mul(&b_bar, &x_exp); // U = B̄·x

        // h_t = Ā_t ⊙ h_{t-1} + U_t along the sequence axis.
        let h = selective_scan(&a_bar, &u); // [batch, seq, inner, state]

        // y = (C ⊙ h).sum(state). `sum_axis` keeps the reduced axis, so drop it.
        let c_exp = unsqueeze(&c, 2); // [batch, seq, 1, state]
        let y = reshape(&sum_axis(&mul(&h, &c_exp), 3), [batch, seq, inner]);

        // Gated skip connection: y ⊙ σ(residual) ⊙ D.
        let d_exp = unsqueeze(&unsqueeze(&self.d, 0), 0); // [1, 1, inner]
        let gated = mul(&mul(&y, &sigmoid(&residual)), &d_exp);

        self.out_proj.forward(&gated)
    }

    /// Selective-scan forward for a `[batch, channels, D, H, W]` volume, with
    /// `channels == input_dim`. Spatial axes are flattened row-major into the
    /// sequence axis, scanned, and restored to `[batch, output_dim, D, H, W]`.
    #[must_use]
    pub fn forward_3d(&self, input: &Var<f32, B>) -> Var<f32, B> {
        let sh = input.tensor.shape();
        assert_eq!(sh.len(), 5, "SelectiveStateSpace: forward_3d input rank-5");
        let (batch, channels) = (sh[0], sh[1]);
        let (dd, hh, ww) = (sh[2], sh[3], sh[4]);
        let seq = dd * hh * ww;

        let flat = reshape(&permute(input, &[0, 2, 3, 4, 1]), [batch, seq, channels]);
        let out = self.forward(&flat); // [batch, seq, output_dim]
        let restored = permute(&out, &[0, 2, 1]); // [batch, output_dim, seq]
        reshape(&restored, [batch, self.output_dim, dd, hh, ww])
    }
}

impl<B> Module<f32, B> for SelectiveStateSpace<B>
where
    B: Backend + BackendOps<f32> + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    fn parameters(&self) -> Vec<Var<f32, B>> {
        let mut params = Vec::new();
        for layer in [
            &self.in_proj,
            &self.out_proj,
            &self.dt_in_proj,
            &self.dt_proj,
            &self.b_proj,
            &self.c_proj,
        ] {
            params.extend(layer.parameters());
        }
        params.push(self.a_log.clone());
        params.push(self.d.clone());
        params
    }

    fn named_parameters(&self) -> Vec<Parameter<f32, B>> {
        let mut named = Vec::new();
        for (prefix, layer) in [
            ("in_proj", &self.in_proj),
            ("out_proj", &self.out_proj),
            ("dt_in_proj", &self.dt_in_proj),
            ("dt_proj", &self.dt_proj),
            ("b_proj", &self.b_proj),
            ("c_proj", &self.c_proj),
        ] {
            named.extend(
                layer
                    .named_parameters()
                    .into_iter()
                    .map(|parameter| parameter.with_prefix(prefix)),
            );
        }
        named.push(Parameter::new(self.a_log.clone(), "a_log"));
        named.push(Parameter::new(self.d.clone(), "d"));
        named
    }

    fn forward(&self, input: &Var<f32, B>) -> Var<f32, B> {
        SelectiveStateSpace::forward(self, input)
    }

    fn load_parameters(&mut self, params: &[Var<f32, B>]) {
        let mut offset = 0;
        for layer in [
            &mut self.in_proj,
            &mut self.out_proj,
            &mut self.dt_in_proj,
            &mut self.dt_proj,
            &mut self.b_proj,
            &mut self.c_proj,
        ] {
            let count = layer.parameters().len();
            layer.load_parameters(&params[offset..offset + count]);
            offset += count;
        }
        self.a_log = params[offset].clone();
        self.d = params[offset + 1].clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_autograd::{mul as vmul, sum};
    use coeus_core::SequentialBackend;

    type Bk = SequentialBackend;

    fn ramp(shape: &[usize]) -> Vec<f32> {
        let n: usize = shape.iter().product();
        (0..n).map(|i| ((i % 17) as f32) / 17.0 - 0.5).collect()
    }

    fn var(shape: &[usize], data: &[f32], grad: bool) -> Var<f32, Bk> {
        Var::new(Tensor::from_slice_on(shape, data, &SequentialBackend), grad)
    }

    /// Reduce `y` against a fixed pseudo-random linear functional `Σ wᵢ yᵢ`, a
    /// well-conditioned oracle (a bare sum is nearly flat in the input because
    /// the projection column sums cancel, so its central difference is
    /// dominated by f32 round-off).
    fn functional(y: &Var<f32, Bk>) -> Var<f32, Bk> {
        let shape = y.tensor.shape();
        let n: usize = shape.iter().product();
        let w: Vec<f32> = (0..n)
            .map(|i| (((i.wrapping_mul(2_654_435_761)) % 1000) as f32) / 1000.0 - 0.5)
            .collect();
        sum(&vmul(y, &var(shape, &w, false)))
    }

    fn small_ssm() -> SelectiveStateSpace<Bk> {
        let config = SelectiveStateSpaceConfig {
            input_dim: 4,
            output_dim: 4,
            state_dim: 3,
            expand_factor: 2,
            dt_rank: 3,
        };
        config.init()
    }

    #[test]
    fn forward_shapes_and_finite() {
        let ssm = small_ssm();
        let shape = [2usize, 5, 4];
        let out = ssm.forward(&var(&shape, &ramp(&shape), false));
        assert_eq!(out.tensor.shape(), &[2, 5, 4]);
        let data = out.tensor.as_slice();
        assert!(data.iter().all(|v| v.is_finite()), "output must be finite");
        assert!(
            data.iter().any(|&v| v.abs() > 1e-6),
            "output must not be trivially zero"
        );
    }

    #[test]
    fn forward_3d_shapes() {
        let ssm = small_ssm();
        let shape = [1usize, 4, 2, 3, 3];
        let out = ssm.forward_3d(&var(&shape, &ramp(&shape), false));
        assert_eq!(out.tensor.shape(), &[1, 4, 2, 3, 3]);
        assert!(out.tensor.as_slice().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn named_parameters_unique_and_complete() {
        let ssm = small_ssm();
        let named = ssm.named_parameters();
        assert_eq!(named.len(), ssm.parameters().len());
        let names: std::collections::HashSet<_> = named.iter().map(|p| p.name.clone()).collect();
        assert_eq!(names.len(), named.len(), "parameter paths must be unique");
        assert!(names.contains("in_proj.weight"));
        assert!(names.contains("out_proj.bias"));
        assert!(names.contains("a_log"));
        assert!(names.contains("d"));
    }

    /// Central-difference gradient check of `d(functional(forward(x)))/dx`
    /// against autograd. Central differences are `O(h²)`-accurate; the bound
    /// `tol_factor·(1+|g|)` covers truncation plus f32 rounding. A non-vacuity
    /// guard rejects an all-zero probed gradient. `seq = 5` and `state = 3`
    /// give the scan and state axes extent > 1 so the recurrence is exercised.
    fn assert_fd_matches<F>(shape: &[usize], base: &[f32], probes: &[usize], forward: F)
    where
        F: Fn(&Var<f32, Bk>) -> Var<f32, Bk>,
    {
        let h = 1.0f32 / 128.0;
        let tol_factor = 3e-2f32;
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
    fn selective_scan_block_gradient_matches_finite_difference() {
        // Exercises the full native forward/backward: input projection,
        // input-dependent Δ/B/C projections, discretization, the differentiable
        // selective scan along the length axis, C-projection reduction, the
        // sigmoid-gated skip connection, and the output projection.
        let ssm = small_ssm();
        let shape = [1usize, 5, 4];
        let base = ramp(&shape);
        assert_fd_matches(&shape, &base, &[0, 7, 13, 19], |x| ssm.forward(x));
    }
}
