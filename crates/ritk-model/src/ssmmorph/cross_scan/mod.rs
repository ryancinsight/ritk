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
    use coeus_autograd::Var;
    use coeus_core::MoiraiBackend;
    use coeus_tensor::Tensor;

    #[test]
    fn planar_directions_round_trip_exact_values() {
        let input = Var::new(
            Tensor::from_slice_on(
                [1, 1, 2, 3],
                &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                &MoiraiBackend::new(),
            ),
            true,
        );
        for &direction in ScanDirection::all_2d() {
            let scanned = Scan2D::scan(&input, direction).expect("planar input is valid");
            let merged =
                Scan2D::merge(&scanned, 2, 3, direction).expect("planar sequence length is valid");
            let contiguous = coeus_autograd::contiguous(&merged);
            assert_eq!(contiguous.tensor.as_slice(), input.tensor.as_slice());
        }
    }

    #[test]
    fn volumetric_directions_round_trip_exact_values_and_gradient() {
        let input = Var::new(
            Tensor::from_slice_on(
                [1, 1, 2, 2, 2],
                &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                &MoiraiBackend::new(),
            ),
            true,
        );
        let scan = CrossScan::new(CrossScanConfig::new_3d());
        let sequences = scan.apply(&input).expect("volumetric input is valid");
        let merged = scan
            .merge_3d(&sequences, 2, 2, 2)
            .expect("six volumetric sequences are valid");
        assert_eq!(merged.tensor.as_slice(), input.tensor.as_slice());
        merged.backward();
        assert!(input.grad().is_some(), "scan graph must remain connected");
    }

    #[test]
    fn rejects_rank_mismatch() {
        let input = Var::new(Tensor::zeros_on([1, 1, 2, 2], &MoiraiBackend::new()), false);
        let result = CrossScan::new(CrossScanConfig::new_3d()).apply(&input);
        assert!(matches!(result, Err(crate::ModelError::Shape { .. })));
    }
}
