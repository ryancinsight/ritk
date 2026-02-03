//! Cross-Scan Mechanism for Spatial State Space Models
//!
//! Implements the 2D/3D cross-scan strategy from VMamba, enabling the SSM to
//! capture spatial dependencies by scanning along multiple directions.
//!
//! # Scan Directions (2D)
//! 1. Horizontal (left-to-right)
//! 2. Horizontal reverse (right-to-left)
//! 3. Vertical (top-to-bottom)
//! 4. Vertical reverse (bottom-to-top)
//!
//! # Scan Directions (3D)
//! Additional depth-wise scans for volumetric data
//!
//! The cross-scan mechanism transforms spatial data into sequences that can be
//! processed by 1D SSMs, then merges the results from all directions.

use burn::prelude::*;

/// Scan direction for cross-scan operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanDirection {
    /// Left to right
    HorizontalForward,
    /// Right to left
    HorizontalReverse,
    /// Top to bottom
    VerticalForward,
    /// Bottom to top
    VerticalReverse,
    /// Front to back (3D)
    DepthForward,
    /// Back to front (3D)
    DepthReverse,
}

impl ScanDirection {
    /// Get all 2D scan directions
    pub fn all_2d() -> &'static [ScanDirection] {
        &[
            ScanDirection::HorizontalForward,
            ScanDirection::HorizontalReverse,
            ScanDirection::VerticalForward,
            ScanDirection::VerticalReverse,
        ]
    }
    
    /// Get all 3D scan directions
    pub fn all_3d() -> &'static [ScanDirection] {
        &[
            ScanDirection::HorizontalForward,
            ScanDirection::HorizontalReverse,
            ScanDirection::VerticalForward,
            ScanDirection::VerticalReverse,
            ScanDirection::DepthForward,
            ScanDirection::DepthReverse,
        ]
    }
}

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
                input.permute([0, 1, 2, 3]).reshape([batch, channels, height * width])
            }
            ScanDirection::HorizontalReverse => {
                // Flip horizontally then reshape
                let flipped = Self::flip_horizontal(input);
                flipped.permute([0, 1, 2, 3]).reshape([batch, channels, height * width])
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
            ScanDirection::HorizontalForward => {
                scanned.reshape([batch, channels, height, width])
            }
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
                input.permute([0, 1, 2, 3, 4]).reshape([batch, channels, seq_len])
            }
            ScanDirection::HorizontalReverse => {
                let flipped = Self::flip_horizontal(input);
                flipped.permute([0, 1, 2, 3, 4]).reshape([batch, channels, seq_len])
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
        let sum = merged.into_iter().fold(None, |acc, t| {
            match acc {
                None => Some(t),
                Some(a) => Some(a + t),
            }
        }).expect("At least one direction required");
        
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
        let sum = merged.into_iter().fold(None, |acc, t| {
            match acc {
                None => Some(t),
                Some(a) => Some(a + t),
            }
        }).expect("At least one direction required");
        
        sum / (directions.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    
    #[test]
    fn test_scan_2d() {
        let device = <NdArray as Backend>::Device::default();
        let input = Tensor::<NdArray, 4>::zeros([2, 8, 16, 16], &device);
        
        let scanned = Scan2D::scan(input.clone(), ScanDirection::HorizontalForward);
        assert_eq!(scanned.dims(), [2, 8, 256]); // 16 * 16 = 256
        
        let merged = Scan2D::merge(scanned, 16, 16, ScanDirection::HorizontalForward);
        assert_eq!(merged.dims(), [2, 8, 16, 16]);
    }
    
    #[test]
    fn test_scan_3d() {
        let device = <NdArray as Backend>::Device::default();
        let input = Tensor::<NdArray, 5>::zeros([2, 8, 4, 8, 8], &device);
        
        let scanned = Scan3D::scan(input.clone(), ScanDirection::DepthForward);
        assert_eq!(scanned.dims(), [2, 8, 256]); // 4 * 8 * 8 = 256
        
        let merged = Scan3D::merge(scanned, 4, 8, 8, ScanDirection::DepthForward);
        assert_eq!(merged.dims(), [2, 8, 4, 8, 8]);
    }
    
    #[test]
    fn test_cross_scan_3d() {
        let device = <NdArray as Backend>::Device::default();
        let config = CrossScanConfig::new_3d();
        let cross_scan = CrossScan::new(&config);
        
        let input = Tensor::<NdArray, 5>::zeros([1, 16, 4, 8, 8], &device);
        let sequences = cross_scan.apply(input);
        
        assert_eq!(sequences.len(), 6); // 6 directions for 3D
    }
}
