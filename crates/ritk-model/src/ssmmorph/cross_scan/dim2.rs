//! Two-dimensional directional sequence views.

use coeus_autograd::{flip, permute, reshape, Var};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_ops::BackendOps;

use super::directions::ScanDirection;
use crate::ModelError;

/// Bidirectional row-major and column-major scans of planar features.
pub struct Scan2D;

impl Scan2D {
    /// Convert `[batch, channels, height, width]` into a directional sequence.
    pub fn scan<B>(input: &Var<f32, B>, direction: ScanDirection) -> Result<Var<f32, B>, ModelError>
    where
        B: Backend + BackendOps<f32>,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
    {
        let shape = input.tensor.shape();
        if shape.len() != 4 {
            return Err(ModelError::Shape {
                operation: "Scan2D::scan",
                expected: "[batch, channels, height, width]",
                actual: shape.to_vec() });
        }
        let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
        let ordered = match direction {
            ScanDirection::HorizontalForward => input.clone(),
            ScanDirection::HorizontalReverse => flip(input, 3),
            ScanDirection::VerticalForward => permute(input, &[0, 1, 3, 2]),
            ScanDirection::VerticalReverse => permute(&flip(input, 2), &[0, 1, 3, 2]),
            ScanDirection::DepthForward | ScanDirection::DepthReverse => {
                return Err(ModelError::Shape {
                    operation: "Scan2D::scan",
                    expected: "a planar scan direction",
                    actual: shape.to_vec() });
            }
        };
        Ok(reshape(&ordered, [batch, channels, height * width]))
    }

    /// Restore a directional sequence to `[batch, channels, height, width]`.
    pub fn merge<B>(
        scanned: &Var<f32, B>,
        height: usize,
        width: usize,
        direction: ScanDirection,
    ) -> Result<Var<f32, B>, ModelError>
    where
        B: Backend + BackendOps<f32>,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
    {
        let shape = scanned.tensor.shape();
        if shape.len() != 3 || shape[2] != height * width {
            return Err(ModelError::Shape {
                operation: "Scan2D::merge",
                expected: "[batch, channels, height * width]",
                actual: shape.to_vec() });
        }
        let (batch, channels) = (shape[0], shape[1]);
        Ok(match direction {
            ScanDirection::HorizontalForward => reshape(scanned, [batch, channels, height, width]),
            ScanDirection::HorizontalReverse => {
                flip(&reshape(scanned, [batch, channels, height, width]), 3)
            }
            ScanDirection::VerticalForward => permute(
                &reshape(scanned, [batch, channels, width, height]),
                &[0, 1, 3, 2],
            ),
            ScanDirection::VerticalReverse => flip(
                &permute(
                    &reshape(scanned, [batch, channels, width, height]),
                    &[0, 1, 3, 2],
                ),
                2,
            ),
            ScanDirection::DepthForward | ScanDirection::DepthReverse => {
                return Err(ModelError::Shape {
                    operation: "Scan2D::merge",
                    expected: "a planar scan direction",
                    actual: shape.to_vec() });
            }
        })
    }
}
