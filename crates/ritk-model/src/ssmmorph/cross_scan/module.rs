//! Directional scan orchestration.

use coeus_autograd::{add, scalar_mul, Var};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_ops::BackendOps;

use super::super::policy::ScanDimensionality;
use super::{Scan2D, Scan3D, ScanDirection};
use crate::ModelError;

/// Cross-scan dimensionality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CrossScanConfig {
    dimensionality: ScanDimensionality }

impl CrossScanConfig {
    /// Construct a planar four-direction scan.
    #[must_use]
    pub const fn new_2d() -> Self {
        Self {
            dimensionality: ScanDimensionality::Scan2d }
    }

    /// Construct a volumetric six-direction scan.
    #[must_use]
    pub const fn new_3d() -> Self {
        Self {
            dimensionality: ScanDimensionality::Scan3d }
    }
}

/// Stateless cross-scan strategy.
#[derive(Debug, Clone, Copy)]
pub struct CrossScan {
    dimensionality: ScanDimensionality }

impl CrossScan {
    /// Construct a cross-scan strategy.
    #[must_use]
    pub const fn new(config: CrossScanConfig) -> Self {
        Self {
            dimensionality: config.dimensionality }
    }

    /// Return whether this strategy operates on volumes.
    #[must_use]
    pub const fn is_3d(self) -> bool {
        matches!(self.dimensionality, ScanDimensionality::Scan3d)
    }

    /// Return the canonical directions for this dimensionality.
    #[must_use]
    pub fn directions(self) -> &'static [ScanDirection] {
        match self.dimensionality {
            ScanDimensionality::Scan2d => ScanDirection::all_2d(),
            ScanDimensionality::Scan3d => ScanDirection::all_3d() }
    }

    /// Produce one sequence view for every canonical direction.
    pub fn apply<B>(&self, input: &Var<f32, B>) -> Result<Vec<Var<f32, B>>, ModelError>
    where
        B: Backend + BackendOps<f32>,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
    {
        self.directions()
            .iter()
            .map(|&direction| match self.dimensionality {
                ScanDimensionality::Scan2d => Scan2D::scan(input, direction),
                ScanDimensionality::Scan3d => Scan3D::scan(input, direction) })
            .collect()
    }

    /// Merge and average planar directional sequences.
    pub fn merge_2d<B>(
        &self,
        sequences: &[Var<f32, B>],
        height: usize,
        width: usize,
    ) -> Result<Var<f32, B>, ModelError>
    where
        B: Backend + BackendOps<f32>,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
    {
        if self.dimensionality != ScanDimensionality::Scan2d
            || sequences.len() != ScanDirection::all_2d().len()
        {
            return Err(ModelError::Shape {
                operation: "CrossScan::merge_2d",
                expected: "four planar directional sequences",
                actual: vec![sequences.len()] });
        }
        let merged = sequences
            .iter()
            .zip(ScanDirection::all_2d())
            .map(|(sequence, &direction)| Scan2D::merge(sequence, height, width, direction))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(average(&merged))
    }

    /// Merge and average volumetric directional sequences.
    pub fn merge_3d<B>(
        &self,
        sequences: &[Var<f32, B>],
        depth: usize,
        height: usize,
        width: usize,
    ) -> Result<Var<f32, B>, ModelError>
    where
        B: Backend + BackendOps<f32>,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
    {
        if self.dimensionality != ScanDimensionality::Scan3d
            || sequences.len() != ScanDirection::all_3d().len()
        {
            return Err(ModelError::Shape {
                operation: "CrossScan::merge_3d",
                expected: "six volumetric directional sequences",
                actual: vec![sequences.len()] });
        }
        let merged = sequences
            .iter()
            .zip(ScanDirection::all_3d())
            .map(|(sequence, &direction)| Scan3D::merge(sequence, depth, height, width, direction))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(average(&merged))
    }
}

fn average<B>(values: &[Var<f32, B>]) -> Var<f32, B>
where
    B: Backend + BackendOps<f32>,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    let sum = values[1..]
        .iter()
        .fold(values[0].clone(), |sum, value| add(&sum, value));
    scalar_mul(&sum, 1.0 / values.len() as f32)
}
