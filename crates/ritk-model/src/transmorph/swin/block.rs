//! Three-dimensional Swin transformer block.

use coeus_autograd::{add, pad, permute, reshape, roll, slice, Var};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_nn::{LayerNorm, Module};
use coeus_ops::BackendOps;
use coeus_tensor::Tensor;

use super::{
    attention::{WindowAttention, WindowAttentionConfig},
    mlp::{Mlp, MlpConfig},
};

/// Shifted-window transformer block for channel-last volumes.
#[derive(Clone)]
pub struct SwinTransformerBlock<B>
where
    B: Backend + BackendOps<f32>,
{
    first_norm: LayerNorm<f32, B>,
    attention: WindowAttention<B>,
    second_norm: LayerNorm<f32, B>,
    mlp: Mlp<B>,
    shift_size: usize,
    window_size: usize,
}

/// Swin block configuration.
#[derive(Debug, Clone, Copy)]
pub struct SwinTransformerBlockConfig {
    input_dim: usize,
    heads: usize,
    window_size: usize,
    shift_size: usize,
    mlp_ratio: f64,
    dropout: f64,
    attention_dropout: f64,
}

impl SwinTransformerBlockConfig {
    /// Construct a Swin block configuration.
    #[must_use]
    pub const fn new(
        input_dim: usize,
        heads: usize,
        window_size: usize,
        shift_size: usize,
        mlp_ratio: f64,
    ) -> Self {
        Self {
            input_dim,
            heads,
            window_size,
            shift_size,
            mlp_ratio,
            dropout: 0.0,
            attention_dropout: 0.0,
        }
    }

    /// Initialize this block on backend `B`.
    #[must_use]
    pub fn init<B>(self) -> SwinTransformerBlock<B>
    where
        B: Backend + BackendOps<f32>,
    {
        let hidden_dim = (self.input_dim as f64 * self.mlp_ratio) as usize;
        SwinTransformerBlock {
            first_norm: LayerNorm::new(self.input_dim, 1e-5),
            attention: WindowAttentionConfig::new(self.input_dim, self.heads, self.window_size)
                .with_dropout(self.attention_dropout)
                .init(),
            second_norm: LayerNorm::new(self.input_dim, 1e-5),
            mlp: MlpConfig::new(self.input_dim, hidden_dim)
                .with_dropout(self.dropout)
                .init(),
            shift_size: self.shift_size,
            window_size: self.window_size,
        }
    }
}

impl<B> Module<f32, B> for SwinTransformerBlock<B>
where
    B: Backend + BackendOps<f32>,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    fn parameters(&self) -> Vec<Var<f32, B>> {
        let mut parameters = self.first_norm.parameters();
        parameters.extend(self.attention.parameters());
        parameters.extend(self.second_norm.parameters());
        parameters.extend(self.mlp.parameters());
        parameters
    }

    fn forward(&self, input: &Var<f32, B>) -> Var<f32, B> {
        let shape = input.tensor.shape();
        assert_eq!(shape.len(), 5, "Swin block requires rank-five input");
        let (batch, depth, height, width, channels) =
            (shape[0], shape[1], shape[2], shape[3], shape[4]);
        let window = self.window_size;
        let pad_depth = (window - depth % window) % window;
        let pad_height = (window - height % window) % window;
        let pad_width = (window - width % window) % window;

        let shortcut = input.clone();
        let normalized = self.first_norm.forward_nd(input);
        let padded = pad(
            &normalized,
            &[
                (0, 0),
                (0, pad_depth),
                (0, pad_height),
                (0, pad_width),
                (0, 0),
            ],
            0.0,
        );
        let padded_shape = padded.tensor.shape();
        let (padded_depth, padded_height, padded_width) =
            (padded_shape[1], padded_shape[2], padded_shape[3]);

        let attended = if self.shift_size == 0 {
            let windows = self.window_partition(&padded);
            let windows = self.attention.forward(&windows);
            self.window_reverse(&windows, padded_depth, padded_height, padded_width)
        } else {
            let shift = -(self.shift_size as isize);
            let shifted = roll(&padded, &[shift, shift, shift], &[1, 2, 3]);
            let windows = self.window_partition(&shifted);
            let mask = attention_mask::<B>(
                batch,
                padded_depth,
                padded_height,
                padded_width,
                window,
                self.shift_size,
            );
            let windows = self.attention.forward_with_mask(&windows, Some(&mask));
            let shifted = self.window_reverse(&windows, padded_depth, padded_height, padded_width);
            roll(
                &shifted,
                &[
                    self.shift_size as isize,
                    self.shift_size as isize,
                    self.shift_size as isize,
                ],
                &[1, 2, 3],
            )
        };
        let attended = slice(
            &attended,
            &[
                (0, batch),
                (0, depth),
                (0, height),
                (0, width),
                (0, channels),
            ],
        );
        let residual = add(&shortcut, &attended);
        let normalized = self.second_norm.forward_nd(&residual);
        add(&residual, &self.mlp.forward(&normalized))
    }

    fn load_parameters(&mut self, parameters: &[Var<f32, B>]) {
        let first = self.first_norm.parameters().len();
        let attention = self.attention.parameters().len();
        let second = self.second_norm.parameters().len();
        self.first_norm.load_parameters(&parameters[..first]);
        self.attention
            .load_parameters(&parameters[first..first + attention]);
        self.second_norm
            .load_parameters(&parameters[first + attention..first + attention + second]);
        self.mlp
            .load_parameters(&parameters[first + attention + second..]);
    }

    fn train(&mut self, mode: bool) {
        self.attention.train(mode);
        self.mlp.train(mode);
    }
}

impl<B> SwinTransformerBlock<B>
where
    B: Backend + BackendOps<f32>,
{
    fn window_partition(&self, input: &Var<f32, B>) -> Var<f32, B> {
        let shape = input.tensor.shape();
        let (batch, depth, height, width, channels) =
            (shape[0], shape[1], shape[2], shape[3], shape[4]);
        let window = self.window_size;
        let partitioned = reshape(
            input,
            [batch, depth / window, window, height, width, channels],
        );
        let partitioned = permute(&partitioned, &[0, 1, 3, 4, 2, 5]);
        let partitioned = reshape(
            &partitioned,
            [batch * (depth / window), height, width, window, channels],
        );
        let partitioned = reshape(
            &partitioned,
            [
                batch * (depth / window),
                height / window,
                window,
                width,
                window,
                channels,
            ],
        );
        let partitioned = permute(&partitioned, &[0, 1, 3, 4, 2, 5]);
        let partitioned = reshape(
            &partitioned,
            [
                batch * (depth / window) * (height / window),
                width,
                window,
                window,
                channels,
            ],
        );
        let partitioned = reshape(
            &partitioned,
            [
                batch * (depth / window) * (height / window),
                width / window,
                window,
                window,
                window,
                channels,
            ],
        );
        let partitioned = permute(&partitioned, &[0, 1, 3, 4, 2, 5]);
        reshape(
            &partitioned,
            [
                batch * (depth / window) * (height / window) * (width / window),
                window * window * window,
                channels,
            ],
        )
    }

    fn window_reverse(
        &self,
        windows: &Var<f32, B>,
        depth: usize,
        height: usize,
        width: usize,
    ) -> Var<f32, B> {
        let window = self.window_size;
        let shape = windows.tensor.shape();
        let channels = shape[2];
        let windows_per_batch = (depth / window) * (height / window) * (width / window);
        let batch = shape[0] / windows_per_batch;
        let restored = reshape(
            windows,
            [
                batch * (depth / window) * (height / window),
                width / window,
                window,
                window,
                window,
                channels,
            ],
        );
        let restored = permute(&restored, &[0, 1, 4, 2, 3, 5]);
        let restored = reshape(
            &restored,
            [
                batch * (depth / window) * (height / window),
                width,
                window,
                window,
                channels,
            ],
        );
        let restored = reshape(
            &restored,
            [
                batch * (depth / window),
                height / window,
                width,
                window,
                window,
                channels,
            ],
        );
        let restored = permute(&restored, &[0, 1, 4, 2, 3, 5]);
        let restored = reshape(
            &restored,
            [batch * (depth / window), height, width, window, channels],
        );
        let restored = reshape(
            &restored,
            [batch, depth / window, height, width, window, channels],
        );
        let restored = permute(&restored, &[0, 1, 4, 2, 3, 5]);
        reshape(&restored, [batch, depth, height, width, channels])
    }
}

fn attention_mask<B>(
    batch: usize,
    depth: usize,
    height: usize,
    width: usize,
    window: usize,
    shift: usize,
) -> Var<f32, B>
where
    B: Backend + BackendOps<f32>,
{
    let windows_depth = depth / window;
    let windows_height = height / window;
    let windows_width = width / window;
    let voxels = window * window * window;
    let windows = windows_depth * windows_height * windows_width;
    let mut values = Vec::with_capacity(batch * windows * voxels * voxels);
    for _ in 0..batch {
        for window_depth in 0..windows_depth {
            for window_height in 0..windows_height {
                for window_width in 0..windows_width {
                    let labels = (0..window)
                        .flat_map(|local_depth| {
                            (0..window).flat_map(move |local_height| {
                                (0..window).map(move |local_width| {
                                    let z = window_depth * window + local_depth;
                                    let y = window_height * window + local_height;
                                    let x = window_width * window + local_width;
                                    usize::from(z >= depth - shift) * 4
                                        + usize::from(y >= height - shift) * 2
                                        + usize::from(x >= width - shift)
                                })
                            })
                        })
                        .collect::<Vec<_>>();
                    for &left in &labels {
                        for &right in &labels {
                            values.push(if left == right { 0.0 } else { -100.0 });
                        }
                    }
                }
            }
        }
    }
    Var::new(
        Tensor::from_slice_on([batch * windows, 1, voxels, voxels], &values, &B::default()),
        false,
    )
}
