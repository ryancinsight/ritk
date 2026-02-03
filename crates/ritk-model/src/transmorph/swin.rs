use burn::{
    module::Param,
    nn::{
        LayerNorm, LayerNormConfig, Linear, LinearConfig, Dropout, DropoutConfig,
        Gelu,
    },
    prelude::*,
    tensor::{activation::softmax, backend::Backend, Tensor},
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
            attention: WindowAttentionConfig::new(
                self.input_dim,
                self.num_heads,
                self.window_size,
            )
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
    fn create_mask(&self, b: usize, d: usize, h: usize, w: usize, device: &B::Device) -> Tensor<B, 4> {
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
                        let region = Tensor::ones([1, d_end - d_start, h_end - h_start, w_end - w_start, 1], device) * cnt;
                        img_mask = img_mask.slice_assign(
                            [0..1, d_start..d_end, h_start..h_end, w_start..w_end, 0..1],
                            region
                        );
                    }
                    cnt += 1.0;
                }
            }
        }
        
        let mask_windows = self.window_partition(img_mask);
        let mask_windows = mask_windows.squeeze::<2>(2);
        
        let attn_mask = mask_windows.clone().unsqueeze_dim::<3>(1) - mask_windows.unsqueeze_dim::<3>(2);
        
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
    
    fn pad_tensor(&self, x: Tensor<B, 5>, pad_d: usize, pad_h: usize, pad_w: usize) -> Tensor<B, 5> {
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
        let shift = if shift < 0 { shift + size as i32 } else { shift } as usize;
        
        if shift == 0 { return x; }
        
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
        let x = x.reshape([b, d/ws, ws, h, w, c])
                 .permute([0, 1, 3, 4, 2, 5]);
        
        // Flatten B and D/ws: [B * D/ws, H, W, ws, C] (5D)
        let x = x.reshape([b * (d/ws), h, w, ws, c]);
        
        // 2. H-split: [B', H, W, ws, C] -> [B', H/ws, ws, W, ws, C] (6D)
        // Permute to [B', H/ws, W, ws(D), ws(H), C]
        let x = x.reshape([b * (d/ws), h/ws, ws, w, ws, c])
                 .permute([0, 1, 3, 4, 2, 5]);
                 
        // Flatten B' and H/ws: [B * D/ws * H/ws, W, ws, ws, C] (5D)
        let x = x.reshape([b * (d/ws) * (h/ws), w, ws, ws, c]);
        
        // 3. W-split: [B'', W, ws, ws, C] -> [B'', W/ws, ws, ws, ws, C] (6D)
        // Permute to [B'', W/ws, ws(D), ws(H), ws(W), C]
        let x = x.reshape([b * (d/ws) * (h/ws), w/ws, ws, ws, ws, c])
                 .permute([0, 1, 3, 4, 2, 5]);
                 
        // Final flatten
        x.reshape([b * (d/ws) * (h/ws) * (w/ws), ws * ws * ws, c])
    }

    fn window_reverse(&self, windows: Tensor<B, 3>, d: usize, h: usize, w: usize) -> Tensor<B, 5> {
        // Input: [B * num_windows, ws*ws*ws, C]
        // Output: [B, D, H, W, C]
        let ws = self.window_size;
        let [total_windows, _, c] = windows.dims();
        let b = total_windows / ((d / ws) * (h / ws) * (w / ws));
        
        // 1. Un-flatten W: [B'', W/ws, ws(D), ws(H), ws(W), C]
        let x = windows.reshape([b * (d/ws) * (h/ws), w/ws, ws, ws, ws, c]);
        
        // Permute back: [B'', W/ws, ws(W), ws(D), ws(H), C]
        let x = x.permute([0, 1, 4, 2, 3, 5]);
        
        // Merge W: [B'', W, ws, ws, C]
        let x = x.reshape([b * (d/ws) * (h/ws), w, ws, ws, c]);
        
        // 2. Un-flatten H: [B', H, W, ws(D), ws(H), C] (Actually x is [B', W, ws(D), ws(H), C] from step 1?)
        // Wait, step 1 output is [B'', W, ws(D), ws(H), C]
        // B'' = B * D/ws * H/ws
        // So x is [B * D/ws * H/ws, W, ws(D), ws(H), C]
        
        // Split B'' to recover H/ws
        let x = x.reshape([b * (d/ws), h/ws, w, ws, ws, c]); // [B', H/ws, W, ws(D), ws(H), C]
        
        // Permute to [B', H/ws, ws(H), W, ws(D), C]
        let x = x.permute([0, 1, 4, 2, 3, 5]);
        
        // Merge H: [B', H, W, ws(D), C]
        let x = x.reshape([b * (d/ws), h, w, ws, c]);
        
        // 3. Un-flatten D: [B, D/ws, H, W, ws(D), C]
        let x = x.reshape([b, d/ws, h, w, ws, c]);
        
        // Permute to [B, D/ws, ws(D), H, W, C]
        let x = x.permute([0, 1, 4, 2, 3, 5]);
        
        // Merge D
        x.reshape([b, d, h, w, c])
    }
}

use burn::tensor::Int;

#[derive(Module, Debug)]
pub struct WindowAttention<B: Backend> {
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    proj: Linear<B>,
    relative_position_bias_table: Param<Tensor<B, 2>>,
    relative_position_index: Tensor<B, 1, Int>, // Flattened [M^3 * M^3]
    num_heads: usize,
    head_dim: usize,
    window_size: usize,
    scale: f64,
    dropout: Dropout,
}

#[derive(Config, Debug)]
pub struct WindowAttentionConfig {
    input_dim: usize,
    num_heads: usize,
    window_size: usize,
    #[config(default = 0.0)]
    dropout: f64,
}

impl WindowAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> WindowAttention<B> {
        let head_dim = self.input_dim / self.num_heads;
        let m = self.window_size;
        
        // Initialize relative position bias table
        // Shape: [(2M-1)^3, num_heads]
        let num_relative_distance = (2 * m - 1).pow(3);
        let table = Tensor::random(
            [num_relative_distance, self.num_heads], 
            burn::tensor::Distribution::Normal(0.0, 0.02), 
            device
        );
        
        // Compute relative position index
        // We compute this on CPU and move to device because it's complex logic
        let index = Self::compute_relative_position_index(m);
        let index = Tensor::<B, 1, Int>::from_ints(index.as_slice(), device);

        WindowAttention {
            query: LinearConfig::new(self.input_dim, self.input_dim).init(device),
            key: LinearConfig::new(self.input_dim, self.input_dim).init(device),
            value: LinearConfig::new(self.input_dim, self.input_dim).init(device),
            proj: LinearConfig::new(self.input_dim, self.input_dim).init(device),
            relative_position_bias_table: Param::from_tensor(table),
            relative_position_index: index,
            num_heads: self.num_heads,
            head_dim,
            window_size: m,
            scale: (head_dim as f64).powf(-0.5),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
    
    fn compute_relative_position_index(m: usize) -> Vec<i32> {
        // Coordinate grid
        let mut coords = Vec::with_capacity(m * m * m);
        for d in 0..m {
            for h in 0..m {
                for w in 0..m {
                    coords.push((d as i32, h as i32, w as i32));
                }
            }
        }
        
        let n = m * m * m;
        let mut index = Vec::with_capacity(n * n);
        
        for i in 0..n {
            let (d1, h1, w1) = coords[i];
            for j in 0..n {
                let (d2, h2, w2) = coords[j];
                
                let rd = (d1 - d2) + (m as i32 - 1);
                let rh = (h1 - h2) + (m as i32 - 1);
                let rw = (w1 - w2) + (m as i32 - 1);
                
                let range = 2 * m as i32 - 1;
                let idx = rd * range * range + rh * range + rw;
                index.push(idx);
            }
        }
        index
    }
}

impl<B: Backend> WindowAttention<B> {
    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 4>>) -> Tensor<B, 3> {
        // x: [Batch_Windows, N, C] where N = ws^3
        let [b, n, c] = x.dims();
        
        let q = self.query.forward(x.clone())
            .reshape([b, n, self.num_heads, self.head_dim])
            .permute([0, 2, 1, 3]); // [B, NumHeads, N, HeadDim]
            
        let k = self.key.forward(x.clone())
            .reshape([b, n, self.num_heads, self.head_dim])
            .permute([0, 2, 1, 3]);
            
        let v = self.value.forward(x.clone())
            .reshape([b, n, self.num_heads, self.head_dim])
            .permute([0, 2, 1, 3]);

        // Attention score: Q @ K^T * scale
        let mut attn = q.matmul(k.transpose()) * self.scale;
        
        // Add relative position bias
        // table: [L, NumHeads] where L = (2M-1)^3
        // index: [N*N]
        // bias: gather(table, index) -> [N*N, NumHeads]
        // reshape -> [N, N, NumHeads]
        // permute -> [NumHeads, N, N]
        // unsqueeze -> [1, NumHeads, N, N]
        
        let table = self.relative_position_bias_table.val();
        let index = self.relative_position_index.clone();
        
        let bias = table.select(0, index)
            .reshape([n, n, self.num_heads])
            .permute([2, 0, 1])
            .unsqueeze::<4>();
            
        attn = attn + bias;

        // Apply mask if provided (for shifted window attention)
        if let Some(mask) = mask {
            // mask shape: [B, 1, N, N]
            attn = attn + mask;
        }

        let attn = softmax(attn, 3);
        let attn = self.dropout.forward(attn);

        // x = attn @ V
        let x = attn.matmul(v)
            .permute([0, 2, 1, 3]) // [B, N, NumHeads, HeadDim]
            .reshape([b, n, c]);
            
        self.proj.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    fc1: Linear<B>,
    act: Gelu,
    fc2: Linear<B>,
    dropout: Dropout,
}

#[derive(Config, Debug)]
pub struct MlpConfig {
    input_dim: usize,
    hidden_dim: usize,
    #[config(default = 0.0)]
    dropout: f64,
}

impl MlpConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mlp<B> {
        Mlp {
            fc1: LinearConfig::new(self.input_dim, self.hidden_dim).init(device),
            act: Gelu::new(),
            fc2: LinearConfig::new(self.hidden_dim, self.input_dim).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> Mlp<B> {
    pub fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let x = self.fc1.forward(x);
        let x = self.act.forward(x);
        let x = self.dropout.forward(x);
        let x = self.fc2.forward(x);
        self.dropout.forward(x)
    }
}
