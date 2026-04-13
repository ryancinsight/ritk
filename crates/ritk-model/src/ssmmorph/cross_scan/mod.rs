//! Cross-Scan Mechanism for Spatial State Space Models
pub mod dim2;
pub mod dim3;
pub mod directions;
pub mod module;

pub use dim2::Scan2D;
pub use dim3::Scan3D;
pub use directions::ScanDirection;
pub use module::{CrossScan, CrossScanConfig};

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::backend::Backend;
    use burn::tensor::Tensor;
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
