//! Three-dimensional shifted-window attention.

use coeus_autograd::{
    add, index_select, matmul, permute, reshape, scalar_mul, softmax, swapaxes, unsqueeze, Var,
};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_nn::{init, Dropout, Linear, Module};
use coeus_ops::BackendOps;
use coeus_tensor::Tensor;

/// Windowed multi-head self-attention with relative position bias.
#[derive(Clone)]
pub struct WindowAttention<B>
where
    B: Backend + BackendOps<f32>,
{
    query: Linear<f32, B>,
    key: Linear<f32, B>,
    value: Linear<f32, B>,
    projection: Linear<f32, B>,
    relative_position_bias: Var<f32, B>,
    relative_position_index: Var<f32, B>,
    heads: usize,
    head_dim: usize,
    scale: f32,
    dropout: Dropout,
}

/// Window-attention configuration.
#[derive(Debug, Clone, Copy)]
pub struct WindowAttentionConfig {
    input_dim: usize,
    heads: usize,
    window_size: usize,
    dropout: f64,
}

impl WindowAttentionConfig {
    /// Construct a window-attention configuration.
    #[must_use]
    pub const fn new(input_dim: usize, heads: usize, window_size: usize) -> Self {
        Self {
            input_dim,
            heads,
            window_size,
            dropout: 0.0,
        }
    }

    /// Set attention dropout probability.
    #[must_use]
    pub const fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Initialize attention on backend `B`.
    #[must_use]
    pub fn init<B>(self) -> WindowAttention<B>
    where
        B: Backend + BackendOps<f32>,
    {
        assert_eq!(
            self.input_dim % self.heads,
            0,
            "attention width must be divisible by head count"
        );
        let head_dim = self.input_dim / self.heads;
        let distances = (2 * self.window_size - 1).pow(3);
        let mut relative_position_bias = Var::new(
            Tensor::zeros_on([distances, self.heads], &B::default()),
            true,
        );
        init::normal_with_seed(&mut relative_position_bias, 0.0, 0.02, 42);
        let index = compute_relative_position_index(self.window_size)
            .into_iter()
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        let relative_position_index = Var::new(
            Tensor::from_slice_on([index.len()], &index, &B::default()),
            false,
        );
        let mut attention = WindowAttention {
            query: Linear::new(self.input_dim, self.input_dim, true),
            key: Linear::new(self.input_dim, self.input_dim, true),
            value: Linear::new(self.input_dim, self.input_dim, true),
            projection: Linear::new(self.input_dim, self.input_dim, true),
            relative_position_bias,
            relative_position_index,
            heads: self.heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            dropout: Dropout::new(self.dropout),
        };
        crate::initialization::linear(&mut attention.query, self.input_dim, self.input_dim, 211);
        crate::initialization::linear(&mut attention.key, self.input_dim, self.input_dim, 212);
        crate::initialization::linear(&mut attention.value, self.input_dim, self.input_dim, 213);
        crate::initialization::linear(
            &mut attention.projection,
            self.input_dim,
            self.input_dim,
            214,
        );
        attention
    }
}

impl<B> WindowAttention<B>
where
    B: Backend + BackendOps<f32>,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    /// Apply attention to `[windows, voxels, channels]`.
    #[must_use]
    pub fn forward_with_mask(
        &self,
        input: &Var<f32, B>,
        mask: Option<&Var<f32, B>>,
    ) -> Var<f32, B> {
        let shape = input.tensor.shape();
        assert_eq!(shape.len(), 3, "window attention requires rank three");
        let (windows, voxels, channels) = (shape[0], shape[1], shape[2]);
        let project = |linear: &Linear<f32, B>| {
            let value = linear.forward(input);
            let value = reshape(&value, [windows, voxels, self.heads, self.head_dim]);
            permute(&value, &[0, 2, 1, 3])
        };
        let query = project(&self.query);
        let key = project(&self.key);
        let value = project(&self.value);
        let key = swapaxes(&key, 2, 3);
        let scores = matmul(&query, &key);
        let scores = reshape(&scores, [windows, self.heads, voxels, voxels]);
        let scores = scalar_mul(&scores, self.scale);

        let bias = index_select(
            &self.relative_position_bias,
            0,
            &self.relative_position_index,
        );
        let bias = reshape(&bias, [voxels, voxels, self.heads]);
        let bias = permute(&bias, &[2, 0, 1]);
        let bias = unsqueeze(&bias, 0);
        assert_eq!(
            scores.tensor.shape(),
            &[windows, self.heads, voxels, voxels],
            "attention score shape must preserve window and head axes"
        );
        assert_eq!(
            bias.tensor.shape(),
            &[1, self.heads, voxels, voxels],
            "relative-position bias must broadcast over windows"
        );
        let mut scores = add(&scores, &bias);
        if let Some(mask) = mask {
            scores = add(&scores, mask);
        }
        let weights = self.dropout.forward(&softmax(&scores, 3));
        let output = matmul(&weights, &value);
        let output = reshape(&output, [windows, self.heads, voxels, self.head_dim]);
        let output = permute(&output, &[0, 2, 1, 3]);
        let output = reshape(&output, [windows, voxels, channels]);
        self.projection.forward(&output)
    }
}

impl<B> Module<f32, B> for WindowAttention<B>
where
    B: Backend + BackendOps<f32>,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    fn parameters(&self) -> Vec<Var<f32, B>> {
        let mut parameters = self.query.parameters();
        parameters.extend(self.key.parameters());
        parameters.extend(self.value.parameters());
        parameters.extend(self.projection.parameters());
        parameters.push(self.relative_position_bias.clone());
        parameters
    }

    fn forward(&self, input: &Var<f32, B>) -> Var<f32, B> {
        self.forward_with_mask(input, None)
    }

    fn load_parameters(&mut self, parameters: &[Var<f32, B>]) {
        let segment = self.query.parameters().len();
        self.query.load_parameters(&parameters[0..segment]);
        self.key.load_parameters(&parameters[segment..2 * segment]);
        self.value
            .load_parameters(&parameters[2 * segment..3 * segment]);
        self.projection
            .load_parameters(&parameters[3 * segment..4 * segment]);
        self.relative_position_bias = parameters[4 * segment].clone();
    }

    fn train(&mut self, mode: bool) {
        self.dropout.set_training(mode);
    }
}

fn compute_relative_position_index(window_size: usize) -> Vec<usize> {
    let coordinates = (0..window_size)
        .flat_map(|depth| {
            (0..window_size)
                .flat_map(move |height| (0..window_size).map(move |width| (depth, height, width)))
        })
        .collect::<Vec<_>>();
    let range = 2 * window_size - 1;
    let mut index = Vec::with_capacity(coordinates.len() * coordinates.len());
    for &(depth, height, width) in &coordinates {
        for &(other_depth, other_height, other_width) in &coordinates {
            let relative_depth = depth + window_size - 1 - other_depth;
            let relative_height = height + window_size - 1 - other_height;
            let relative_width = width + window_size - 1 - other_width;
            index.push(relative_depth * range * range + relative_height * range + relative_width);
        }
    }
    index
}
