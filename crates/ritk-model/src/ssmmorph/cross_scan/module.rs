//! Core CrossScan algorithmic orchestration
use burn::prelude::*;

use super::dim2::Scan2D;
use super::dim3::Scan3D;
use super::directions::ScanDirection;

/// Configuration for Cross-Scan module
#[derive(Config, Debug, PartialEq, Eq)]
pub struct CrossScanConfig {
    /// Whether to use 3D scanning (volumetric) or 2D
    #[config(default = "true")]
    pub use_3d: bool,
    /// Number of directions to scan
    #[config(default = "6")]
    pub num_directions: usize,
}

impl CrossScanConfig {
    /// Create new 2D cross-scan config
    pub fn new_2d() -> Self {
        Self {
            use_3d: false,
            num_directions: 4,
        }
    }

    /// Create new 3D cross-scan config
    pub fn new_3d() -> Self {
        Self {
            use_3d: true,
            num_directions: 6,
        }
    }
}

/// Cross-Scan module for spatial feature processing
#[derive(Debug, Clone)]
pub struct CrossScan {
    config: CrossScanConfig,
}

impl CrossScan {
    /// Create new CrossScan module
    pub fn new(config: &CrossScanConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    /// Check if using 3D scanning
    pub fn use_3d(&self) -> bool {
        self.config.use_3d
    }

    /// Get scan directions based on configuration
    pub fn directions(&self) -> &'static [ScanDirection] {
        if self.config.use_3d {
            ScanDirection::all_3d()
        } else {
            ScanDirection::all_2d()
        }
    }

    /// Apply cross-scan to input tensor
    ///
    /// Returns a vector of scanned sequences, one per direction
    pub fn apply<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Vec<Tensor<B, 3>>
    where
        B: Backend,
    {
        let directions = self.directions();

        if self.config.use_3d {
            // Handle 3D case
            if D != 5 {
                panic!("3D cross-scan requires 5D input [B, C, D, H, W]");
            }
            // Convert to 5D for processing
            let input_5d: Tensor<B, 5> = input.clone().reshape([
                input.dims()[0],
                input.dims()[1],
                input.dims()[2],
                input.dims()[3],
                input.dims()[4],
            ]);

            directions
                .iter()
                .map(|&dir| Scan3D::scan(input_5d.clone(), dir))
                .collect()
        } else {
            // Handle 2D case
            if D != 4 {
                panic!("2D cross-scan requires 4D input [B, C, H, W]");
            }
            let input_4d: Tensor<B, 4> = input.clone().reshape([
                input.dims()[0],
                input.dims()[1],
                input.dims()[2],
                input.dims()[3],
            ]);

            directions
                .iter()
                .map(|&dir| Scan2D::scan(input_4d.clone(), dir))
                .collect()
        }
    }

    /// Merge processed sequences back to spatial tensor (2D)
    pub fn merge_2d<B: Backend>(
        &self,
        sequences: Vec<Tensor<B, 3>>,
        height: usize,
        width: usize,
        directions: &[ScanDirection],
    ) -> Tensor<B, 4> {
        assert_eq!(sequences.len(), directions.len());
        assert!(!self.config.use_3d, "Cannot use merge_2d with 3D config");

        let merged: Vec<Tensor<B, 4>> = sequences
            .into_iter()
            .zip(directions.iter())
            .map(|(seq, &dir)| Scan2D::merge(seq, height, width, dir))
            .collect();

        // Average the results from all directions
        let sum = merged
            .into_iter()
            .fold(None, |acc, t| match acc {
                None => Some(t),
                Some(a) => Some(a + t),
            })
            .expect("At least one direction required");

        sum / (directions.len() as f64)
    }

    /// Merge processed sequences back to spatial tensor (3D)
    pub fn merge_3d<B: Backend>(
        &self,
        sequences: Vec<Tensor<B, 3>>,
        depth: usize,
        height: usize,
        width: usize,
        directions: &[ScanDirection],
    ) -> Tensor<B, 5> {
        assert_eq!(sequences.len(), directions.len());
        assert!(self.config.use_3d, "Cannot use merge_3d with 2D config");

        let merged: Vec<Tensor<B, 5>> = sequences
            .into_iter()
            .zip(directions.iter())
            .map(|(seq, &dir)| Scan3D::merge(seq, depth, height, width, dir))
            .collect();

        // Average the results from all directions
        let sum = merged
            .into_iter()
            .fold(None, |acc, t| match acc {
                None => Some(t),
                Some(a) => Some(a + t),
            })
            .expect("At least one direction required");

        sum / (directions.len() as f64)
    }
}
