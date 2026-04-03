//! 2D Cross-Scan mechanisms and dimension mappings
use super::directions::ScanDirection;
use burn::prelude::*;

/// 2D Cross-Scan operation
///
/// Transforms 2D spatial features into sequences along multiple directions
pub struct Scan2D;

impl Scan2D {
    /// Scan tensor along specified direction
    ///
    /// # Arguments
    /// * `input` - Input tensor [batch, channels, height, width]
    /// * `direction` - Scan direction
    ///
    /// # Returns
    /// * Scanned tensor [batch, channels, seq_len] where seq_len = height * width
    pub fn scan<B: Backend>(input: Tensor<B, 4>, direction: ScanDirection) -> Tensor<B, 3> {
        let [batch, channels, height, width] = input.dims();
        let _device = input.device();

        match direction {
            ScanDirection::HorizontalForward => {
                // Reshape to [batch, channels, height, width] -> [batch, channels, height*width]
                // Row-major order (left-to-right, top-to-bottom)
                input
                    .permute([0, 1, 2, 3])
                    .reshape([batch, channels, height * width])
            }
            ScanDirection::HorizontalReverse => {
                // Flip horizontally then reshape
                let flipped = Self::flip_horizontal(input);
                flipped
                    .permute([0, 1, 2, 3])
                    .reshape([batch, channels, height * width])
            }
            ScanDirection::VerticalForward => {
                // Transpose to make vertical scan horizontal, then reshape
                // [batch, channels, height, width] -> [batch, channels, width, height]
                let transposed = input.permute([0, 1, 3, 2]);
                transposed.reshape([batch, channels, height * width])
            }
            ScanDirection::VerticalReverse => {
                // Flip vertically, transpose, then reshape
                let flipped = Self::flip_vertical(input);
                let transposed = flipped.permute([0, 1, 3, 2]);
                transposed.reshape([batch, channels, height * width])
            }
            _ => panic!("Invalid 2D scan direction"),
        }
    }

    /// Merge scanned sequences back to spatial tensor
    ///
    /// # Arguments
    /// * `scanned` - Scanned tensor [batch, channels, seq_len]
    /// * `height` - Output height
    /// * `width` - Output width
    /// * `direction` - Original scan direction
    ///
    /// # Returns
    /// * Reconstructed tensor [batch, channels, height, width]
    pub fn merge<B: Backend>(
        scanned: Tensor<B, 3>,
        height: usize,
        width: usize,
        direction: ScanDirection,
    ) -> Tensor<B, 4> {
        let [batch, channels, _] = scanned.dims();

        match direction {
            ScanDirection::HorizontalForward => scanned.reshape([batch, channels, height, width]),
            ScanDirection::HorizontalReverse => {
                let reshaped = scanned.reshape([batch, channels, height, width]);
                Self::flip_horizontal(reshaped)
            }
            ScanDirection::VerticalForward => {
                let reshaped = scanned.reshape([batch, channels, width, height]);
                reshaped.permute([0, 1, 3, 2])
            }
            ScanDirection::VerticalReverse => {
                let reshaped = scanned.reshape([batch, channels, width, height]);
                let transposed = reshaped.permute([0, 1, 3, 2]);
                Self::flip_vertical(transposed)
            }
            _ => panic!("Invalid 2D scan direction"),
        }
    }

    /// Flip tensor horizontally
    fn flip_horizontal<B: Backend>(input: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_batch, _channels, _height, width] = input.dims();
        let device = input.device();

        // Create index tensor for reverse order along width
        let indices: Vec<i64> = (0..width).map(|i| (width - 1 - i) as i64).collect();
        let index_tensor = Tensor::from_data(indices.as_slice(), &device);

        // Index select along width dimension (dim 3)
        input.select(3, index_tensor)
    }

    /// Flip tensor vertically
    fn flip_vertical<B: Backend>(input: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_batch, _channels, height, _width] = input.dims();
        let device = input.device();

        // Create index tensor for reverse order along height
        let indices: Vec<i64> = (0..height).map(|i| (height - 1 - i) as i64).collect();
        let index_tensor = Tensor::from_data(indices.as_slice(), &device);

        // Index select along height dimension (dim 2)
        input.select(2, index_tensor)
    }
}
