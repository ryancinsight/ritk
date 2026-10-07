use super::super::{write_dicom_series, write_dicom_series_with_metadata, DicomWriteError};
use super::fixtures::{make_image, make_test_metadata, Backend};
use ritk_core::image::Image;
use ritk_image::tensor::Tensor;
use ritk_spatial::{Direction, Point, Spacing};

#[test]
fn shape_and_geometry_boundaries_are_typed() {
    use super::super::pixel_encoding::{validate_image_shape, validate_spatial_metadata};
    for (shape, count, expected) in [
        (
            [1, 0, 1],
            0,
            DicomWriteError::InvalidDimensions {
                depth: 1,
                rows: 0,
                columns: 1,
            },
        ),
        ([2, usize::MAX, 2], 0, DicomWriteError::PixelCountOverflow),
        (
            [1, 2, 3],
            5,
            DicomWriteError::PixelCountMismatch {
                expected: 6,
                actual: 5,
            },
        ),
        (
            [1, 65536, 1],
            65536,
            DicomWriteError::RowsOutOfRange { rows: 65536 },
        ),
        (
            [1, 1, 65536],
            65536,
            DicomWriteError::ColumnsOutOfRange { columns: 65536 },
        ),
    ] {
        assert_eq!(
            validate_image_shape(shape, count)
                .expect_err("invariant: malformed fixture is rejected")
                .downcast_ref::<DicomWriteError>(),
            Some(&expected)
        );
    }
    let shape =
        validate_image_shape([1, 65535, 1], 65535).expect("invariant: prepared fixture is valid");
    assert_eq!(
        (
            shape.rows_attribute,
            shape.columns_attribute,
            shape.total_samples
        ),
        (65535, 1, 65535)
    );
    for (spacing, direction) in [
        ([0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        ([-1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        ([1.0; 3], [1.0, 0.0, 0.0, 2.0, 0.0, 0.0]),
        ([1.0; 3], [2.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        ([1.0; 3], [1.0, 0.0, 0.0, 1.0, 1.0, 0.0]),
        ([1.0; 3], [f64::NAN, 0.0, 0.0, 0.0, 1.0, 0.0]),
    ] {
        assert_eq!(
            validate_spatial_metadata(&spacing, &[0.0; 3], &direction)
                .expect_err("invariant: malformed fixture is rejected")
                .downcast_ref::<DicomWriteError>(),
            Some(&DicomWriteError::InvalidSpatialMetadata)
        );
    }
    let component = f64::from(0.5_f32.sqrt());
    assert_eq!(
        validate_spatial_metadata(
            &[1.0; 3],
            &[0.0; 3],
            &[component, component, 0.0, -component, component, 0.0]
        )
        .expect("invariant: prepared fixture is valid"),
        ()
    );
}

#[test]
fn series_preflight_rejects_later_slice_before_replacing_earlier_slice() {
    let temp = tempfile::tempdir().expect("tempdir");
    let directory = temp.path().join("series");
    std::fs::create_dir(&directory).expect("prior directory");
    let path = directory.join("slice_0000.dcm");
    let original = b"prior first slice";
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        std::fs::write(&path, original).expect("prior file");
        let tensor =
            Tensor::<f32, Backend>::from_slice_on([2, 1, 1], &[0.0, value], &Default::default());
        let image = Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        )
        .expect("image rank");
        for with_metadata in [false, true] {
            let result = if with_metadata {
                write_dicom_series_with_metadata(&directory, &image, Some(&make_test_metadata()))
            } else {
                write_dicom_series(&directory, &image)
            };
            let error = result.expect_err("nonfinite later slice");
            assert_eq!(
                error.downcast_ref::<DicomWriteError>(),
                Some(&DicomWriteError::NonFinitePixel { index: 1 })
            );
            assert_eq!(std::fs::read(&path).expect("prior file remains"), original);
            assert_eq!(
                std::fs::read_dir(&directory)
                    .expect("directory remains")
                    .count(),
                1
            );
        }
    }
}

#[test]
fn metadata_preflight_rejects_invalid_pixel_descriptions_before_output_changes() {
    let temp = tempfile::tempdir().expect("tempdir");
    let directory = temp.path().join("series");
    std::fs::create_dir(&directory).expect("prior directory");
    let path = directory.join("slice_0000.dcm");
    let original = b"prior slice";
    std::fs::write(&path, original).expect("prior file");
    let image = make_image(1, 1, 1, 2.0);
    for (allocated, stored, high) in [(0, 0, 0), (7, 7, 6), (16, 0, 0), (8, 9, 8), (16, 12, 15)] {
        let mut metadata = make_test_metadata();
        metadata.bits_allocated = Some(allocated);
        metadata.bits_stored = Some(stored);
        metadata.high_bit = Some(high);
        let error = write_dicom_series_with_metadata(&directory, &image, Some(&metadata))
            .expect_err("invalid source pixel description");
        assert_eq!(
            error.downcast_ref::<DicomWriteError>(),
            Some(&DicomWriteError::InvalidPixelDescription {
                bits_allocated: allocated,
                bits_stored: stored,
                high_bit: high,
            })
        );
        assert_eq!(std::fs::read(&path).expect("prior file remains"), original);
    }
    let mut metadata = make_test_metadata();
    // The fixture's slice normal is +Z, so both terms must grow on Z.
    metadata.origin[2] = f64::MAX;
    metadata.spacing[0] = f64::MAX;
    let image = make_image(2, 1, 1, 2.0);
    let error = write_dicom_series_with_metadata(&directory, &image, Some(&metadata))
        .expect_err("later derived position overflows");
    assert_eq!(
        error.downcast_ref::<DicomWriteError>(),
        Some(&DicomWriteError::InvalidSpatialMetadata)
    );
    assert_eq!(std::fs::read(&path).expect("prior file remains"), original);
    assert_eq!(
        std::fs::read_dir(&directory)
            .expect("directory remains")
            .count(),
        1
    );
}
