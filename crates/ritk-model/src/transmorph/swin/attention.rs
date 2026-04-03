use burn::{
    module::Param,
    nn::{Dropout, DropoutConfig, Linear, LinearConfig},
    prelude::*,
    tensor::{activation::softmax, backend::Backend, Int, Tensor},
};

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
            device,
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

        let q = self
            .query
            .forward(x.clone())
            .reshape([b, n, self.num_heads, self.head_dim])
            .permute([0, 2, 1, 3]); // [B, NumHeads, N, HeadDim]

        let k = self
            .key
            .forward(x.clone())
            .reshape([b, n, self.num_heads, self.head_dim])
            .permute([0, 2, 1, 3]);

        let v = self
            .value
            .forward(x.clone())
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

        let bias = table
            .select(0, index)
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
        let x = attn
            .matmul(v)
            .permute([0, 2, 1, 3]) // [B, N, NumHeads, HeadDim]
            .reshape([b, n, c]);

        self.proj.forward(x)
    }
}
