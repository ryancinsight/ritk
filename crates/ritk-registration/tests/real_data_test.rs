//! Integration tests using real medical imaging data.
//!
//! These tests are skipped by default. To run them:
//! 1. Download test data: `cargo xtask download-datasets`
//! 2. Run: `cargo nextest run -p ritk-registration --run-ignored all`

use coeus_core::SequentialBackend;
use ritk_core::spatial::Point3;
use ritk_image::native::Image;
use ritk_interpolation::trilinear_interpolation;
use ritk_io::{format::nifti::native::NiftiReader, ImageReader};
use ritk_spatial::{Direction, Point, Spacing};
use std::path::{Path, PathBuf};

type B = SequentialBackend;
const MNI152_RITK_SHAPE: [usize; 3] = [215, 256, 207];
const MNI152_SPACING: f64 = 0.737_463_116_645_813;
const NIFTI_HEADER_F32_ERROR: f64 = f32::EPSILON as f64;

fn get_test_data_dir() -> Option<PathBuf> {
    let paths = [
        Path::new("test_data"),
        Path::new("../test_data"),
        Path::new("../../test_data"),
    ];

    paths
        .iter()
        .find(|path| path.exists())
        .map(|path| (*path).to_path_buf())
}

fn native_scalar_nifti_fixture(data_dir: &Path) -> PathBuf {
    let path = data_dir.join("ants_example").join("mni152.nii.gz");
    assert!(
        path.is_file(),
        "downloaded test data must contain the 3-D scalar MNI152 NIfTI fixture at {}",
        path.display()
    );
    path
}

fn read_native_nifti(path: &Path) -> Image<f32, B, 3> {
    NiftiReader::new(B::default())
        .read(path)
        .expect("invariant: downloaded NIfTI fixture conforms to the native reader contract")
}

fn identity_samples(image: &Image<f32, B, 3>, points: &[[usize; 3]]) -> Vec<f32> {
    let [depth, height, width] = image.shape();
    let point_count = points.len();
    let mut grid = vec![0.0_f32; 3 * point_count];
    for (index, [z, y, x]) in points.iter().copied().enumerate() {
        grid[index] = z as f32;
        grid[point_count + index] = y as f32;
        grid[2 * point_count + index] = x as f32;
    }

    let input = Image::from_flat_on(
        image.data_vec(),
        [1, 1, depth, height, width],
        Point::origin(),
        Spacing::uniform(1.0),
        Direction::identity(),
        &B::default(),
    )
    .expect("invariant: native image data matches its rank-five interpolation view");
    let coordinates = Image::from_flat_on(
        grid,
        [1, 3, point_count, 1, 1],
        Point::origin(),
        Spacing::uniform(1.0),
        Direction::identity(),
        &B::default(),
    )
    .expect("invariant: identity sample coordinates match the interpolation grid shape");

    trilinear_interpolation(&input, &coordinates)
        .expect("invariant: native identity interpolation accepts generated grid")
        .data_vec()
}

#[test]
#[ignore = "requires test data"]
fn nifti_io_real_data_preserves_valid_metadata() {
    let data_dir = get_test_data_dir()
        .expect("Test data directory not found. Run: cargo xtask download-datasets");
    let file_path = native_scalar_nifti_fixture(&data_dir);

    println!("Testing NIfTI I/O with: {}", file_path.display());
    let image = read_native_nifti(&file_path);
    let shape = image.shape();

    assert_eq!(shape, MNI152_RITK_SHAPE);
    for axis in 0..3 {
        let spacing = image.spacing()[axis];
        assert!(
            (spacing - MNI152_SPACING).abs() <= NIFTI_HEADER_F32_ERROR,
            "MNI152 spacing axis {axis} must preserve the f32 NIfTI header value: expected {MNI152_SPACING}, got {spacing}"
        );
    }
    assert!(
        image.direction().is_orthogonal(),
        "direction matrix should be orthogonal"
    );
}

#[test]
#[ignore = "requires test data"]
fn native_nifti_metadata_round_trips_coordinates() {
    let data_dir = get_test_data_dir()
        .expect("Test data directory not found. Run: cargo xtask download-datasets");
    let file_path = native_scalar_nifti_fixture(&data_dir);
    println!("Testing metadata consistency for: {}", file_path.display());
    let image = read_native_nifti(&file_path);
    let shape = image.shape();
    let index = Point3::new([
        (shape[0] / 2) as f64,
        (shape[1] / 2) as f64,
        (shape[2] / 2) as f64,
    ]);
    let physical = image.transform_continuous_index_to_physical_point(&index);
    let back_to_index = image.transform_physical_point_to_continuous_index(&physical);

    for axis in 0..3 {
        assert!(
            (index[axis] - back_to_index[axis]).abs() < 1e-6,
            "coordinate round-trip drift at axis {axis}: index={}, recovered={}",
            index[axis],
            back_to_index[axis]
        );
    }
}

#[test]
#[ignore = "requires test data"]
fn native_identity_resampling_returns_source_voxels() {
    let data_dir = get_test_data_dir()
        .expect("Test data directory not found. Run: cargo xtask download-datasets");
    let file_path = native_scalar_nifti_fixture(&data_dir);

    println!("Testing native resampling with: {}", file_path.display());
    let image = read_native_nifti(&file_path);
    let [depth, height, width] = image.shape();
    assert!(
        depth > 0 && height > 0 && width > 0,
        "native NIfTI image dimensions must be non-zero"
    );
    let points = [
        [depth / 2, height / 2, width / 2],
        [0, 0, 0],
        [depth - 1, height - 1, width - 1],
    ];
    let values = identity_samples(&image, &points);
    let voxels = image.data_vec();
    let expected: Vec<f32> = points
        .iter()
        .map(|[z, y, x]| voxels[z * height * width + y * width + x])
        .collect();

    assert_eq!(values, expected);
}
