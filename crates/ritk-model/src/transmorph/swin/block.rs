//! Swin transformer block (3-D), Coeus-native.
//!
//! One Swin block: `LayerNorm â†’ (shifted) window attention â†’ residual`, then
//! `LayerNorm â†’ MLP â†’ residual`, over a `[B, D, H, W, C]` token volume. Window
//! partitioning, cyclic shift, and the shifted-window connectivity mask are all
//! expressed with [`coeus_autograd`] shape ops, so gradients flow end to end.
//! No Burn tensors, modules, or backends cross this boundary.

use super::{attention::WindowAttention, mlp::Mlp};
use coeus_autograd::{add, cat, permute, reshape, roll, slice, Parameter, Var};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_nn::module::Module;
use coeus_nn::normalization::LayerNorm;
use coeus_ops::BackendOps;
use coeus_tensor::Tensor;

/// LayerNorm numerical-stability constant (matches the Burn `LayerNorm` default).
const NORM_EPS: f64 = 1e-5;
/// Masked-attention additive penalty for cross-region window pairs.
const MASK_PENALTY: f32 = -100.0;

/// A single 3-D Swin transformer block.
#[derive(Clone)]
pub struct SwinTransformerBlock<B: Backend + BackendOps<f32> + Default> {
    norm1: LayerNorm<f32, B>,
    attention: WindowAttention<B>,
    norm2: LayerNorm<f32, B>,
    mlp: Mlp<B>,
    shift_size: usize,
    window_size: usize,
}

impl<B> SwinTransformerBlock<B>
where
    B: Backend + BackendOps<f32> + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    /// Construct a Swin block over `input_dim` channels.
    ///
    /// `shift_size = 0` yields a regular-window block; a positive shift yields the
    /// shifted-window variant with a connectivity mask. `mlp_ratio` scales the
    /// MLP hidden width.
    pub fn new(
        input_dim: usize,
        num_heads: usize,
        window_size: usize,
        shift_size: usize,
        mlp_ratio: f64,
        seed: u64,
    ) -> Self {
        let mlp_hidden_dim = (input_dim as f64 * mlp_ratio) as usize;
        Self {
            norm1: LayerNorm::new(input_dim, NORM_EPS),
            attention: WindowAttention::new(input_dim, num_heads, window_size, seed),
            norm2: LayerNorm::new(input_dim, NORM_EPS),
            mlp: Mlp::new(input_dim, mlp_hidden_dim, seed ^ 0x1357_9BDF),
            shift_size,
            window_size,
        }
    }

    /// Forward pass over a `[B, D, H, W, C]` token volume.
    pub fn forward(&self, x: &Var<f32, B>) -> Var<f32, B> {
        let sh = x.tensor.shape();
        let (b, d, h, w, c) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
        let ws = self.window_size;

        let pad_d = (ws - d % ws) % ws;
        let pad_h = (ws - h % ws) % ws;
        let pad_w = (ws - w % ws) % ws;
        let needs_pad = pad_d > 0 || pad_h > 0 || pad_w > 0;

        let shortcut = x.clone();
        let normed = self.norm1.forward_nd(x);

        let (normed, pd, ph, pw) = if needs_pad {
            let padded = self.pad_tensor(&normed, pad_d, pad_h, pad_w);
            let (pd, ph, pw) = {
                let s = padded.tensor.shape();
                (s[1], s[2], s[3])
            };
            (padded, pd, ph, pw)
        } else {
            (normed, d, h, w)
        };

        let x_att = if self.shift_size > 0 {
            let s = self.shift_size as isize;
            let shifted = roll(&normed, &[-s, -s, -s], &[1, 2, 3]);
            let windows = self.window_partition(&shifted);
            let mask = self.attention_mask(b, pd, ph, pw);
            let attended = self.attention.forward(&windows, Some(&mask));
            let reversed = self.window_reverse(&attended, b, pd, ph, pw, c);
            roll(&reversed, &[s, s, s], &[1, 2, 3])
        } else {
            let windows = self.window_partition(&normed);
            let attended = self.attention.forward(&windows, None);
            self.window_reverse(&attended, b, pd, ph, pw, c)
        };

        let x_att = if needs_pad {
            slice(&x_att, &[(0, b), (0, d), (0, h), (0, w), (0, c)])
        } else {
            x_att
        };

        let x = add(&shortcut, &x_att);

        // MLP branch with pre-norm residual.
        let normed2 = self.norm2.forward_nd(&x);
        let mlp_out = self.mlp.forward(&normed2);
        add(&x, &mlp_out)
    }

    /// Zero-pad `[B, D, H, W, C]` on the trailing side of the spatial axes so
    /// each spatial extent becomes a multiple of the window size.
    fn pad_tensor(&self, x: &Var<f32, B>, pad_d: usize, pad_h: usize, pad_w: usize) -> Var<f32, B> {
        let backend = B::default();
        let mut x = x.clone();
        if pad_d > 0 {
            let s = x.tensor.shape();
            let pad = Var::new(
                Tensor::zeros_on([s[0], pad_d, s[2], s[3], s[4]], &backend),
                false,
            );
            x = cat(&[&x, &pad], 1);
        }
        if pad_h > 0 {
            let s = x.tensor.shape();
            let pad = Var::new(
                Tensor::zeros_on([s[0], s[1], pad_h, s[3], s[4]], &backend),
                false,
            );
            x = cat(&[&x, &pad], 2);
        }
        if pad_w > 0 {
            let s = x.tensor.shape();
            let pad = Var::new(
                Tensor::zeros_on([s[0], s[1], s[2], pad_w, s[4]], &backend),
                false,
            );
            x = cat(&[&x, &pad], 3);
        }
        x
    }

    /// Partition `[B, D, H, W, C]` into non-overlapping cubic windows,
    /// producing `[BÂ·(D/ws)Â·(H/ws)Â·(W/ws), wsÂ³, C]`.
    ///
    /// Reshapes proceed one axis at a time to keep every intermediate tensor at
    /// most rank-6.
    fn window_partition(&self, x: &Var<f32, B>) -> Var<f32, B> {
        let sh = x.tensor.shape();
        let (b, d, h, w, c) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
        let ws = self.window_size;

        let x = permute(&reshape(x, [b, d / ws, ws, h, w, c]), &[0, 1, 3, 4, 2, 5]);
        let x = reshape(&x, [b * (d / ws), h, w, ws, c]);
        let x = permute(
            &reshape(&x, [b * (d / ws), h / ws, ws, w, ws, c]),
            &[0, 1, 3, 4, 2, 5],
        );
        let x = reshape(&x, [b * (d / ws) * (h / ws), w, ws, ws, c]);
        let x = permute(
            &reshape(&x, [b * (d / ws) * (h / ws), w / ws, ws, ws, ws, c]),
            &[0, 1, 3, 4, 2, 5],
        );
        reshape(&x, [b * (d / ws) * (h / ws) * (w / ws), ws * ws * ws, c])
    }

    /// Inverse of [`Self::window_partition`]: reassemble `[B, D, H, W, C]` from
    /// `[BÂ·num_windows, wsÂ³, C]`.
    fn window_reverse(
        &self,
        windows: &Var<f32, B>,
        b: usize,
        d: usize,
        h: usize,
        w: usize,
        c: usize,
    ) -> Var<f32, B> {
        let ws = self.window_size;

        let x = reshape(windows, [b * (d / ws) * (h / ws), w / ws, ws, ws, ws, c]);
        let x = permute(&x, &[0, 1, 4, 2, 3, 5]);
        let x = reshape(&x, [b * (d / ws) * (h / ws), w, ws, ws, c]);
        let x = reshape(&x, [b * (d / ws), h / ws, w, ws, ws, c]);
        let x = permute(&x, &[0, 1, 4, 2, 3, 5]);
        let x = reshape(&x, [b * (d / ws), h, w, ws, c]);
        let x = reshape(&x, [b, d / ws, h, w, ws, c]);
        let x = permute(&x, &[0, 1, 4, 2, 3, 5]);
        reshape(&x, [b, d, h, w, c])
    }

    /// Build the shifted-window attention mask `[bÂ·num_windows, 1, N, N]`.
    ///
    /// Voxels are labeled by the shift region they belong to; window-local pairs
    /// spanning two regions receive [`MASK_PENALTY`] (softmax â†’ ~0), the rest
    /// `0`. The region volume is partitioned by the identical
    /// [`Self::window_partition`] routing, so mask columns align exactly with the
    /// attention windows.
    fn attention_mask(&self, b: usize, d: usize, h: usize, w: usize) -> Var<f32, B> {
        let backend = B::default();
        let ws = self.window_size;
        let s = self.shift_size;

        // Region-id volume [1, D, H, W, 1].
        let mut img = vec![0.0f32; d * h * w];
        let d_slices = [(0usize, d - s), (d - s, d)];
        let h_slices = [(0usize, h - s), (h - s, h)];
        let w_slices = [(0usize, w - s), (w - s, w)];
        let mut cnt = 0.0f32;
        for (d0, d1) in d_slices {
            for (h0, h1) in h_slices {
                for (w0, w1) in w_slices {
                    for dd in d0..d1 {
                        for hh in h0..h1 {
                            for ww in w0..w1 {
                                img[dd * (h * w) + hh * w + ww] = cnt;
                            }
                        }
                    }
                    cnt += 1.0;
                }
            }
        }
        let img_var = Var::new(
            Tensor::from_slice_on([1, d, h, w, 1], &img, &backend),
            false,
        );

        // Route region ids through the same window partition to get [nw, wsÂ³].
        let ids = self.window_partition(&img_var);
        let ids = ids.tensor.as_slice();
        let n = ws * ws * ws;
        let num_windows = ids.len() / n;

        // Pairwise cross-region penalty, replicated across the batch.
        let mut mask = vec![0.0f32; b * num_windows * n * n];
        for batch in 0..b {
            for win in 0..num_windows {
                let base = ((batch * num_windows + win) * n) * n;
                let id_base = win * n;
                for p in 0..n {
                    let idp = ids[id_base + p];
                    for q in 0..n {
                        if idp != ids[id_base + q] {
                            mask[base + p * n + q] = MASK_PENALTY;
                        }
                    }
                }
            }
        }
        Var::new(
            Tensor::from_slice_on([b * num_windows, 1, n, n], &mask, &backend),
            false,
        )
    }

    /// Trainable parameters in forward order.
    pub fn parameters(&self) -> Vec<Var<f32, B>> {
        let mut params = self.norm1.parameters();
        params.extend(self.attention.parameters());
        params.extend(self.norm2.parameters());
        params.extend(self.mlp.parameters());
        params
    }

    /// Trainable parameters with stable hierarchical names.
    pub fn named_parameters(&self) -> Vec<Parameter<f32, B>> {
        let mut named: Vec<Parameter<f32, B>> = self
            .norm1
            .named_parameters()
            .into_iter()
            .map(|p| p.with_prefix("norm1"))
            .collect();
        named.extend(
            self.attention
                .named_parameters()
                .into_iter()
                .map(|p| p.with_prefix("attention")),
        );
        named.extend(
            self.norm2
                .named_parameters()
                .into_iter()
                .map(|p| p.with_prefix("norm2")),
        );
        named.extend(
            self.mlp
                .named_parameters()
                .into_iter()
                .map(|p| p.with_prefix("mlp")),
        );
        named
    }
}
