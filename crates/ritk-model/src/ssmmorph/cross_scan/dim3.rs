//! 3D Cross-Scan dimension tracking
use super::directions::ScanDirection;
use burn::prelude::*;

/// 3D Cross-Scan operation
///
/// Transforms 3D volumetric features into sequences along multiple directions
pub struct Scan3D;

impl Scan3D {
    /// Scan tensor along specified direction
    ///
    /// # Arguments
    /// * `input` - Input tensor [batch, channels, depth, height, width]
    /// * `direction` - Scan direction
    ///
    /// # Returns
    /// * Scanned tensor [batch, channels, seq_len] where seq_len = depth * height * width
    pub fn scan<B: Backend>(input: Tensor<B, 5>, direction: ScanDirection) -> Tensor<B, 3> {
        let [batch, channels, depth, height, width] = input.dims();
        let seq_len = depth * height * width;

        match direction {
            ScanDirection::HorizontalForward => {
                // Standard flatten: [D, H, W] -> [D*H*W]
                input
                    .permute([0, 1, 2, 3, 4])
                    .reshape([batch, channels, seq_len])
            }
            ScanDirection::HorizontalReverse => {
                let flipped = Self::flip_horizontal(input);
                flipped
                    .permute([0, 1, 2, 3, 4])
                    .reshape([batch, channels, seq_len])
            }
            ScanDirection::VerticalForward => {
                // Swap H and W dimensions
                let permuted = input.permute([0, 1, 2, 4, 3]);
                permuted.reshape([batch, channels, seq_len])
            }
            ScanDirection::VerticalReverse => {
                let flipped = Self::flip_vertical(input);
                let permuted = flipped.permute([0, 1, 2, 4, 3]);
                permuted.reshape([batch, channels, seq_len])
            }
            ScanDirection::DepthForward => {
                // Scan along depth: [D, H, W] -> [H, W, D]
                let permuted = input.permute([0, 1, 3, 4, 2]);
                permuted.reshape([batch, channels, seq_len])
            }
            ScanDirection::DepthReverse => {
                let flipped = Self::flip_depth(input);
                let permuted = flipped.permute([0, 1, 3, 4, 2]);
                permuted.reshape([batch, channels, seq_len])
            }
        }
    }

    /// Merge scanned sequences back to volumetric tensor
    pub fn merge<B: Backend>(
        scanned: Tensor<B, 3>,
        depth: usize,
        height: usize,
        width: usize,
        direction: ScanDirection,
    ) -> Tensor<B, 5> {
        let [batch, channels, _] = scanned.dims();

        match direction {
            ScanDirection::HorizontalForward => {
                scanned.reshape([batch, channels, depth, height, width])
            }
            ScanDirection::HorizontalReverse => {
                let reshaped = scanned.reshape([batch, channels, depth, height, width]);
                Self::flip_horizontal(reshaped)
            }
            ScanDirection::VerticalForward => {
                let reshaped = scanned.reshape([batch, channels, depth, width, height]);
                reshaped.permute([0, 1, 2, 4, 3])
            }
            ScanDirection::VerticalReverse => {
                let reshaped = scanned.reshape([batch, channels, depth, width, height]);
                let permuted = reshaped.permute([0, 1, 2, 4, 3]);
                Self::flip_vertical(permuted)
            }
            ScanDirection::DepthForward => {
                let reshaped = scanned.reshape([batch, channels, height, width, depth]);
                reshaped.permute([0, 1, 4, 2, 3])
            }
            ScanDirection::DepthReverse => {
                let reshaped = scanned.reshape([batch, channels, height, width, depth]);
                let permuted = reshaped.permute([0, 1, 4, 2, 3]);
                Self::flip_depth(permuted)
            }
        }
    }

    /// Flip tensor horizontally (along width)
    fn flip_horizontal<B: Backend>(input: Tensor<B, 5>) -> Tensor<B, 5> {
        let [_batch, _channels, _depth, _height, width] = input.dims();
        let device = input.device();

        let indices: Vec<i64> = (0..width).map(|i| (width - 1 - i) as i64).collect();
        let index_tensor = Tensor::from_data(indices.as_slice(), &device);

        input.select(4, index_tensor)
    }

    /// Flip tensor vertically (along height)
    fn flip_vertical<B: Backend>(input: Tensor<B, 5>) -> Tensor<B, 5> {
        let [_batch, _channels, _depth, height, _width] = input.dims();
        let device = input.device();

        let indices: Vec<i64> = (0..height).map(|i| (height - 1 - i) as i64).collect();
        let index_tensor = Tensor::from_data(indices.as_slice(), &device);

        input.select(3, index_tensor)
    }

    /// Flip tensor along depth
    fn flip_depth<B: Backend>(input: Tensor<B, 5>) -> Tensor<B, 5> {
        let [_batch, _channels, depth, _height, _width] = input.dims();
        let device = input.device();

        let indices: Vec<i64> = (0..depth).map(|i| (depth - 1 - i) as i64).collect();
        let index_tensor = Tensor::from_data(indices.as_slice(), &device);

        input.select(2, index_tensor)
    }
}
