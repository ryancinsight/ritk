//! 3-D window multi-head self-attention, Coeus-native.
//!
//! Windowed self-attention with a learned relative-position bias, as used by the
//! Swin transformer. Query/key/value/output projections are [`coeus_nn::Linear`]
//! layers; the relative-position bias table is a trainable [`coeus_autograd::Var`]
//! gathered by a precomputed index. Attention scores, bias, optional shift mask,
//! softmax, and the value aggregation all run through [`coeus_autograd`] ops, so
//! gradients flow to every projection and to the bias table.

use coeus_autograd::{
    add, index_select, matmul, permute, reshape, scalar_mul, softmax, transpose, Parameter, Var,
};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_nn::module::Module;
use coeus_nn::Linear;
use coeus_ops::BackendOps;
use coeus_tensor::Tensor;

/// Windowed multi-head self-attention with relative-position bias.
#[derive(Clone)]
pub struct WindowAttention<B: Backend + BackendOps<f32> + Default> {
    query: Linear<f32, B>,
    key: Linear<f32, B>,
    value: Linear<f32, B>,
    proj: Linear<f32, B>,
    /// `[(2M-1)^3, num_heads]` learned bias per relative offset and head.
    relative_position_bias_table: Var<f32, B>,
    /// Flattened `[N*N]` gather index into the bias table (`N = M^3`).
    relative_position_index: Var<f32, B>,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl<B> WindowAttention<B>
where
    B: Backend + BackendOps<f32> + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    /// Construct window attention over `input_dim` channels split into
    /// `num_heads` heads, on a cubic window of side `window_size`.
    ///
    /// The bias table is `Normal(0, 0.02)`-initialized (the Swin scheme) and the
    /// projection weights Kaiming-uniform; [`Linear::new`] alone leaves weights
    /// at ones, which would make every head identical.
    ///
    /// # Panics
    /// Panics if `input_dim` is not divisible by `num_heads`.
    pub fn new(input_dim: usize, num_heads: usize, window_size: usize, seed: u64) -> Self {
        assert_eq!(
            input_dim % num_heads,
            0,
            "input_dim must be divisible by num_heads"
        );
        let head_dim = input_dim / num_heads;
        let m = window_size;
        let backend = B::default();

        let num_relative_distance = (2 * m - 1).pow(3);
        let mut table = Var::new(
            Tensor::zeros_on([num_relative_distance, num_heads], &backend),
            true,
        );
        coeus_nn::init::normal_with_seed(&mut table, 0.0, 0.02, seed);

        let index = Self::compute_relative_position_index(m);
        let index: Vec<f32> = index.into_iter().map(|i| i as f32).collect();
        let relative_position_index =
            Var::new(Tensor::from_slice_on([index.len()], &index, &backend), false);

        let make_linear = |offset: u64| {
            let mut layer = Linear::new(input_dim, input_dim, true);
            coeus_nn::init::kaiming_uniform_with_seed(
                &mut layer.weight,
                input_dim,
                seed.wrapping_add(offset),
            );
            layer
        };

        WindowAttention {
            query: make_linear(1),
            key: make_linear(2),
            value: make_linear(3),
            proj: make_linear(4),
            relative_position_bias_table: table,
            relative_position_index,
            num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5) as f32,
        }
    }

    /// Precompute the `[N*N]` relative-position index for a cubic window.
    ///
    /// For window side `m`, coordinate pair `(i, j)` maps to the flattened index
    /// of their relative offset in the `(2m-1)^3` offset grid — the standard Swin
    /// relative-position encoding.
    fn compute_relative_position_index(m: usize) -> Vec<i32> {
        let mut coords = Vec::with_capacity(m * m * m);
        for d in 0..m {
            for h in 0..m {
                for w in 0..m {
                    coords.push((d as i32, h as i32, w as i32));
                }
            }
        }

        let n = m * m * m;
        let range = 2 * m as i32 - 1;
        let mut index = Vec::with_capacity(n * n);
        for &(d1, h1, w1) in coords.iter() {
            for &(d2, h2, w2) in coords.iter() {
                let rd = (d1 - d2) + (m as i32 - 1);
                let rh = (h1 - h2) + (m as i32 - 1);
                let rw = (w1 - w2) + (m as i32 - 1);
                index.push(rd * range * range + rh * range + rw);
            }
        }
        index
    }

    /// Attention over `[B_windows, N, C]` tokens.
    ///
    /// `mask`, when present, is `[B_windows, 1, N, N]` with `0`/`-100` entries
    /// (the shifted-window connectivity mask), added to the pre-softmax scores.
    pub fn forward(&self, x: &Var<f32, B>, mask: Option<&Var<f32, B>>) -> Var<f32, B> {
        let shape = x.tensor.shape();
        let (b, n, c) = (shape[0], shape[1], shape[2]);
        let (nh, hd) = (self.num_heads, self.head_dim);

        let project = |lin: &Linear<f32, B>| {
            let y = lin.forward(x);
            permute(&reshape(&y, [b, n, nh, hd]), &[0, 2, 1, 3])
        };
        let q = project(&self.query); // [B, nH, N, hd]
        let k = project(&self.key);
        let v = project(&self.value);

        // Scaled dot-product scores: (Q Kᵀ) · scale → [B, nH, N, N].
        let attn = scalar_mul(&matmul(&q, &transpose(&k, 2, 3)), self.scale);

        // Relative-position bias: gather [N*N, nH] → [nH, N, N] → [1, nH, N, N].
        let bias =
            index_select(&self.relative_position_bias_table, 0, &self.relative_position_index);
        let bias = permute(&reshape(&bias, [n, n, nh]), &[2, 0, 1]);
        let bias = reshape(&bias, [1, nh, n, n]);
        let attn = add(&attn, &bias);

        let attn = match mask {
            Some(mask) => add(&attn, mask),
            None => attn,
        };
        let attn = softmax(&attn, -1);

        // Aggregate values and merge heads: [B, nH, N, hd] → [B, N, C].
        let out = matmul(&attn, &v);
        let out = reshape(&permute(&out, &[0, 2, 1, 3]), [b, n, c]);
        self.proj.forward(&out)
    }

    /// Trainable parameters in forward order.
    pub fn parameters(&self) -> Vec<Var<f32, B>> {
        let mut params = self.query.parameters();
        params.extend(self.key.parameters());
        params.extend(self.value.parameters());
        params.extend(self.proj.parameters());
        params.push(self.relative_position_bias_table.clone());
        params
    }

    /// Trainable parameters with stable hierarchical names.
    pub fn named_parameters(&self) -> Vec<Parameter<f32, B>> {
        let mut named = Vec::new();
        for (prefix, lin) in [
            ("query", &self.query),
            ("key", &self.key),
            ("value", &self.value),
            ("proj", &self.proj),
        ] {
            named.extend(lin.named_parameters().into_iter().map(|p| p.with_prefix(prefix)));
        }
        named.push(Parameter::new(
            self.relative_position_bias_table.clone(),
            "relative_position_bias_table",
        ));
        named
    }
}
