use super::{
    attention::{WindowAttention, WindowAttentionConfig},
    mlp::{Mlp, MlpConfig},
};
use burn::{
    nn::{LayerNorm, LayerNormConfig},
    prelude::*,
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub struct SwinTransformerBlock<B: Backend> {
    norm1: LayerNorm<B>,
    attention: WindowAttention<B>,
    norm2: LayerNorm<B>,
    mlp: Mlp<B>,
    shift_size: usize,
    window_size: usize,
}

#[derive(Config, Debug)]
pub struct SwinTransformerBlockConfig {
    pub input_dim: usize,
    pub num_heads: usize,
    pub window_size: usize,
    pub shift_size: usize,
    pub mlp_ratio: f64,
    #[config(default = 0.0)]
    pub dropout: f64,
    #[config(default = 0.0)]
    pub attention_dropout: f64,
}

impl SwinTransformerBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SwinTransformerBlock<B> {
        let mlp_hidden_dim = (self.input_dim as f64 * self.mlp_ratio) as usize;

        SwinTransformerBlock {
            norm1: LayerNormConfig::new(self.input_dim).init(device),
            attention: WindowAttentionConfig::new(self.input_dim, self.num_heads, self.window_size)
                .with_dropout(self.attention_dropout)
                .init(device),
            norm2: LayerNormConfig::new(self.input_dim).init(device),
            mlp: MlpConfig::new(self.input_dim, mlp_hidden_dim)
                .with_dropout(self.dropout)
                .init(device),
            shift_size: self.shift_size,
            window_size: self.window_size,
        }
    }
}

impl<B: Backend> SwinTransformerBlock<B> {
    fn create_mask(
        &self,
        b: usize,
        d: usize,
        h: usize,
        w: usize,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        let mut img_mask = Tensor::zeros([1, d, h, w, 1], device);
        let s = self.shift_size;

        let d_slices = [(0, d - s), (d - s, d)];
        let h_slices = [(0, h - s), (h - s, h)];
        let w_slices = [(0, w - s), (w - s, w)];

        let mut cnt = 0.0;
        for (d_start, d_end) in d_slices {
            for (h_start, h_end) in h_slices {
                for (w_start, w_end) in w_slices {
                    if d_end > d_start && h_end > h_start && w_end > w_start {
                        let region = Tensor::ones(
                            [1, d_end - d_start, h_end - h_start, w_end - w_start, 1],
                            device,
                        ) * cnt;
                        img_mask = img_mask.slice_assign(
                            [0..1, d_start..d_end, h_start..h_end, w_start..w_end, 0..1],
                            region,
                        );
                    }
                    cnt += 1.0;
                }
            }
        }

        let mask_windows = self.window_partition(img_mask);
        let [num_windows, ws3, _] = mask_windows.dims();
        let mask_windows = mask_windows.reshape([num_windows, ws3]);

        let attn_mask =
            mask_windows.clone().unsqueeze_dim::<3>(1) - mask_windows.unsqueeze_dim::<3>(2);

        let mask = attn_mask.not_equal_elem(0.0).float().mul_scalar(-100.0);

        let [num_windows, n, _] = mask.dims();
        let mask = mask.reshape([1, num_windows, 1, n, n]);
        let mask = mask.repeat(&[b, 1, 1, 1, 1]);
        mask.reshape([b * num_windows, 1, n, n])
    }

    /// Forward pass for Swin Transformer Block.
    /// Input: [Batch, Depth, Height, Width, Channels] (Channels last for easier linear ops, or we transpose)
    /// Burn usually prefers [Batch, Channels, D, H, W] for Conv, but [Batch, ..., Channels] for Linear/Attention.
    /// Let's assume input is [Batch, D, H, W, C].
    pub fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let [b, d, h, w, c] = x.dims();
        let ws = self.window_size;

        let pad_d = (ws - d % ws) % ws;
        let pad_h = (ws - h % ws) % ws;
        let pad_w = (ws - w % ws) % ws;
        let needs_pad = pad_d > 0 || pad_h > 0 || pad_w > 0;

        let shortcut = x.clone();

        // 1. Norm 1
        let x = self.norm1.forward(x);

        // Pad if needed
        let (x, padded_d, padded_h, padded_w) = if needs_pad {
            let x = self.pad_tensor(x, pad_d, pad_h, pad_w);
            let [_, pd, ph, pw, _] = x.dims();
            (x, pd, ph, pw)
        } else {
            (x, d, h, w)
        };

        // 2. Window Attention (with Shift if shift_size > 0)
        let x_att = if self.shift_size > 0 {
            // Shifted Window Attention
            let shifted_x = self.cyclic_shift(x.clone(), -(self.shift_size as i32));
            let attn_windows = self.window_partition(shifted_x);

            // Create mask for shifted window attention
            let mask = self.create_mask(b, padded_d, padded_h, padded_w, &x.device());

            // Apply attention
            let attn_windows = self.attention.forward(attn_windows, Some(mask));

            let shifted_x = self.window_reverse(attn_windows, padded_d, padded_h, padded_w);
            self.cyclic_shift(shifted_x, self.shift_size as i32)
        } else {
            // Standard Window Attention
            let windows = self.window_partition(x);
            let attn_windows = self.attention.forward(windows, None);
            self.window_reverse(attn_windows, padded_d, padded_h, padded_w)
        };

        // Crop if needed
        let x_att = if needs_pad {
            let ranges = [0..b, 0..d, 0..h, 0..w, 0..c];
            x_att.slice(ranges)
        } else {
            x_att
        };

        // Residual connection
        let x = shortcut + x_att;

        // 3. MLP with Norm 2 and Residual
        let shortcut = x.clone();
        let x = self.norm2.forward(x);
        let x = self.mlp.forward(x);

        shortcut + x
    }

    fn pad_tensor(
        &self,
        x: Tensor<B, 5>,
        pad_d: usize,
        pad_h: usize,
        pad_w: usize,
    ) -> Tensor<B, 5> {
        let [b, _d, h, w, c] = x.dims();
        let device = x.device();
        let mut x = x;

        if pad_d > 0 {
            let padding = Tensor::zeros([b, pad_d, h, w, c], &device);
            x = Tensor::cat(vec![x, padding], 1);
        }

        let [_, d_new, _, _, _] = x.dims();
        if pad_h > 0 {
            let padding = Tensor::zeros([b, d_new, pad_h, w, c], &device);
            x = Tensor::cat(vec![x, padding], 2);
        }

        let [_, _, h_new, _, _] = x.dims();
        if pad_w > 0 {
            let padding = Tensor::zeros([b, d_new, h_new, pad_w, c], &device);
            x = Tensor::cat(vec![x, padding], 3);
        }

        x
    }

    fn cyclic_shift(&self, x: Tensor<B, 5>, shift: i32) -> Tensor<B, 5> {
        // Simple cyclic shift implementation along D, H, W dimensions
        // x: [B, D, H, W, C]

        if shift == 0 {
            return x;
        }

        // Implementation of roll for D, H, W using slice and cat.
        let x = self.roll(x, shift, 1); // D
        let x = self.roll(x, shift, 2); // H
        let x = self.roll(x, shift, 3); // W
        x
    }

    fn roll(&self, x: Tensor<B, 5>, shift: i32, dim: usize) -> Tensor<B, 5> {
        let dims = x.dims();
        let size = dims[dim];
        let shift = shift % size as i32;
        let shift = if shift < 0 {
            shift + size as i32
        } else {
            shift
        } as usize;

        if shift == 0 {
            return x;
        }

        // split at size - shift
        let split_idx = size - shift;

        // Create ranges for first part (0..split_idx) and second part (split_idx..size)
        // Since we can't easily iterate/map into [Range; 5], we construct manually.
        let mut ranges_a = [0..dims[0], 0..dims[1], 0..dims[2], 0..dims[3], 0..dims[4]];
        let mut ranges_b = [0..dims[0], 0..dims[1], 0..dims[2], 0..dims[3], 0..dims[4]];

        ranges_a[dim] = 0..split_idx;
        ranges_b[dim] = split_idx..size;

        let a = x.clone().slice(ranges_a);
        let b = x.slice(ranges_b);

        // cat b then a
        Tensor::cat(vec![b, a], dim)
    }

    fn window_partition(&self, x: Tensor<B, 5>) -> Tensor<B, 3> {
        // Input: [B, D, H, W, C]
        // Output: [Num_Windows * B, Window_Size^3, C]
        // We use step-by-step reshaping to avoid >6 dims (NdArray limit)

        let [b, d, h, w, c] = x.dims();
        let ws = self.window_size;

        // 1. D-split: [B, D, H, W, C] -> [B, D/ws, ws, H, W, C] (6D)
        // Permute to [B, D/ws, H, W, ws, C]
        let x = x
            .reshape([b, d / ws, ws, h, w, c])
            .permute([0, 1, 3, 4, 2, 5]);

        // Flatten B and D/ws: [B * D/ws, H, W, ws, C] (5D)
        let x = x.reshape([b * (d / ws), h, w, ws, c]);

        // 2. H-split: [B', H, W, ws, C] -> [B', H/ws, ws, W, ws, C] (6D)
        // Permute to [B', H/ws, W, ws(D), ws(H), C]
        let x = x
            .reshape([b * (d / ws), h / ws, ws, w, ws, c])
            .permute([0, 1, 3, 4, 2, 5]);

        // Flatten B' and H/ws: [B * D/ws * H/ws, W, ws, ws, C] (5D)
        let x = x.reshape([b * (d / ws) * (h / ws), w, ws, ws, c]);

        // 3. W-split: [B'', W, ws, ws, C] -> [B'', W/ws, ws, ws, ws, C] (6D)
        // Permute to [B'', W/ws, ws(D), ws(H), ws(W), C]
        let x = x
            .reshape([b * (d / ws) * (h / ws), w / ws, ws, ws, ws, c])
            .permute([0, 1, 3, 4, 2, 5]);

        // Final flatten
        x.reshape([b * (d / ws) * (h / ws) * (w / ws), ws * ws * ws, c])
    }

    fn window_reverse(&self, windows: Tensor<B, 3>, d: usize, h: usize, w: usize) -> Tensor<B, 5> {
        // Input: [B * num_windows, ws*ws*ws, C]
        // Output: [B, D, H, W, C]
        let ws = self.window_size;
        let [total_windows, _, c] = windows.dims();
        let b = total_windows / ((d / ws) * (h / ws) * (w / ws));

        // 1. Un-flatten W: [B'', W/ws, ws(D), ws(H), ws(W), C]
        let x = windows.reshape([b * (d / ws) * (h / ws), w / ws, ws, ws, ws, c]);

        // Permute back: [B'', W/ws, ws(W), ws(D), ws(H), C]
        let x = x.permute([0, 1, 4, 2, 3, 5]);

        // Merge W: [B'', W, ws, ws, C]
        let x = x.reshape([b * (d / ws) * (h / ws), w, ws, ws, c]);

        // 2. Un-flatten H: [B', H, W, ws(D), ws(H), C] (Actually x is [B', W, ws(D), ws(H), C] from step 1?)
        // Wait, step 1 output is [B'', W, ws(D), ws(H), C]
        // B'' = B * D/ws * H/ws
        // So x is [B * D/ws * H/ws, W, ws(D), ws(H), C]

        // Split B'' to recover H/ws
        let x = x.reshape([b * (d / ws), h / ws, w, ws, ws, c]); // [B', H/ws, W, ws(D), ws(H), C]

        // Permute to [B', H/ws, ws(H), W, ws(D), C]
        let x = x.permute([0, 1, 4, 2, 3, 5]);

        // Merge H: [B', H, W, ws(D), C]
        let x = x.reshape([b * (d / ws), h, w, ws, c]);

        // 3. Un-flatten D: [B, D/ws, H, W, ws(D), C]
        let x = x.reshape([b, d / ws, h, w, ws, c]);

        // Permute to [B, D/ws, ws(D), H, W, C]
        let x = x.permute([0, 1, 4, 2, 3, 5]);

        // Merge D
        x.reshape([b, d, h, w, c])
    }
}
