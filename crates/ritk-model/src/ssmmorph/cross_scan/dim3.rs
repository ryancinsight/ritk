//! Three-dimensional directional sequence views.

use coeus_autograd::{flip, permute, reshape, Var};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_ops::BackendOps;

use super::directions::ScanDirection;
use crate::ModelError;

/// Bidirectional scans along every spatial axis of a volume.
pub struct Scan3D;

impl Scan3D {
    /// Convert `[batch, channels, depth, height, width]` into a directional sequence.
    pub fn scan<B>(input: &Var<f32, B>, direction: ScanDirection) -> Result<Var<f32, B>, ModelError>
    where
        B: Backend + BackendOps<f32>,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
    {
        let shape = input.tensor.shape();
        if shape.len() != 5 {
            return Err(ModelError::Shape {
                operation: "Scan3D::scan",
                expected: "[batch, channels, depth, height, width]",
                actual: shape.to_vec(),
            });
        }
        let (batch, channels, depth, height, width) =
            (shape[0], shape[1], shape[2], shape[3], shape[4]);
        let ordered = match direction {
            ScanDirection::HorizontalForward => input.clone(),
            ScanDirection::HorizontalReverse => flip(input, 4).map_err(ModelError::backend)?,
            ScanDirection::VerticalForward => {
                permute(input, &[0, 1, 2, 4, 3]).map_err(ModelError::backend)?
            }
            ScanDirection::VerticalReverse => {
                let flipped = flip(input, 3).map_err(ModelError::backend)?;
                permute(&flipped, &[0, 1, 2, 4, 3]).map_err(ModelError::backend)?
            }
            ScanDirection::DepthForward => {
                permute(input, &[0, 1, 3, 4, 2]).map_err(ModelError::backend)?
            }
            ScanDirection::DepthReverse => {
                let flipped = flip(input, 2).map_err(ModelError::backend)?;
                permute(&flipped, &[0, 1, 3, 4, 2]).map_err(ModelError::backend)?
            }
        };
        reshape(&ordered, [batch, channels, depth * height * width]).map_err(ModelError::backend)
    }

    /// Restore a directional sequence to `[batch, channels, depth, height, width]`.
    pub fn merge<B>(
        scanned: &Var<f32, B>,
        depth: usize,
        height: usize,
        width: usize,
        direction: ScanDirection,
    ) -> Result<Var<f32, B>, ModelError>
    where
        B: Backend + BackendOps<f32>,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
    {
        let shape = scanned.tensor.shape();
        if shape.len() != 3 || shape[2] != depth * height * width {
            return Err(ModelError::Shape {
                operation: "Scan3D::merge",
                expected: "[batch, channels, depth * height * width]",
                actual: shape.to_vec(),
            });
        }
        let (batch, channels) = (shape[0], shape[1]);
        Ok(match direction {
            ScanDirection::HorizontalForward => {
                reshape(scanned, [batch, channels, depth, height, width])
                    .map_err(ModelError::backend)?
            }
            ScanDirection::HorizontalReverse => {
                let flat = reshape(scanned, [batch, channels, depth, height, width])
                    .map_err(ModelError::backend)?;
                flip(&flat, 4).map_err(ModelError::backend)?
            }
            ScanDirection::VerticalForward => {
                let flat = reshape(scanned, [batch, channels, depth, width, height])
                    .map_err(ModelError::backend)?;
                permute(&flat, &[0, 1, 2, 4, 3]).map_err(ModelError::backend)?
            }
            ScanDirection::VerticalReverse => {
                let flat = reshape(scanned, [batch, channels, depth, width, height])
                    .map_err(ModelError::backend)?;
                let ordered = permute(&flat, &[0, 1, 2, 4, 3]).map_err(ModelError::backend)?;
                flip(&ordered, 3).map_err(ModelError::backend)?
            }
            ScanDirection::DepthForward => {
                let flat = reshape(scanned, [batch, channels, height, width, depth])
                    .map_err(ModelError::backend)?;
                permute(&flat, &[0, 1, 4, 2, 3]).map_err(ModelError::backend)?
            }
            ScanDirection::DepthReverse => {
                let flat = reshape(scanned, [batch, channels, height, width, depth])
                    .map_err(ModelError::backend)?;
                let ordered = permute(&flat, &[0, 1, 4, 2, 3]).map_err(ModelError::backend)?;
                flip(&ordered, 2).map_err(ModelError::backend)?
            }
        })
    }
}
