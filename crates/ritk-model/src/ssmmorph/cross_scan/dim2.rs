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
                actual: shape.to_vec(),
            });
        }
        let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
        let ordered = match direction {
            ScanDirection::HorizontalForward => input.clone(),
            ScanDirection::HorizontalReverse => flip(input, 3).map_err(ModelError::backend)?,
            ScanDirection::VerticalForward => {
                permute(input, &[0, 1, 3, 2]).map_err(ModelError::backend)?
            }
            ScanDirection::VerticalReverse => {
                let flipped = flip(input, 2).map_err(ModelError::backend)?;
                permute(&flipped, &[0, 1, 3, 2]).map_err(ModelError::backend)?
            }
            ScanDirection::DepthForward | ScanDirection::DepthReverse => {
                return Err(ModelError::Shape {
                    operation: "Scan2D::scan",
                    expected: "a planar scan direction",
                    actual: shape.to_vec(),
                });
            }
        };
        reshape(&ordered, [batch, channels, height * width]).map_err(ModelError::backend)
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
                actual: shape.to_vec(),
            });
        }
        let (batch, channels) = (shape[0], shape[1]);
        Ok(match direction {
            ScanDirection::HorizontalForward => {
                reshape(scanned, [batch, channels, height, width]).map_err(ModelError::backend)?
            }
            ScanDirection::HorizontalReverse => {
                let flat = reshape(scanned, [batch, channels, height, width])
                    .map_err(ModelError::backend)?;
                flip(&flat, 3).map_err(ModelError::backend)?
            }
            ScanDirection::VerticalForward => {
                let flat = reshape(scanned, [batch, channels, width, height])
                    .map_err(ModelError::backend)?;
                permute(&flat, &[0, 1, 3, 2]).map_err(ModelError::backend)?
            }
            ScanDirection::VerticalReverse => {
                let flat = reshape(scanned, [batch, channels, width, height])
                    .map_err(ModelError::backend)?;
                let ordered = permute(&flat, &[0, 1, 3, 2]).map_err(ModelError::backend)?;
                flip(&ordered, 2).map_err(ModelError::backend)?
            }
            ScanDirection::DepthForward | ScanDirection::DepthReverse => {
                return Err(ModelError::Shape {
                    operation: "Scan2D::merge",
                    expected: "a planar scan direction",
                    actual: shape.to_vec(),
                });
            }
        })
    }
}
