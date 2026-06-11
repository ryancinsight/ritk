pub(crate) use super::{count_foreground, parse_seed, Backend};
pub use super::{run, SegmentArgs};

pub use burn::tensor::backend::Backend as BurnBackend;
pub use burn::tensor::{Shape, Tensor, TensorData};
pub use ritk_core::image::Image;
pub use ritk_segmentation::{
    multi_otsu_threshold, otsu_threshold, KapurThreshold, LiThreshold, TriangleThreshold,
    YenThreshold,
};
pub use ritk_spatial::{Direction, Point, Spacing};
pub use tempfile::tempdir;

mod clustering;
mod entropy_thresholds;
mod level_set;
mod region_growing;
mod threshold;
mod threshold_negative;
mod watershed;

// ── Test image factories ──────────────────────────────────────────────────────

/// Build a 4×4×4 bimodal image.
///
/// The first half of voxels (flat indices 0–31) have intensity 20.0;
/// the second half (32–63) have intensity 200.0.
/// The analytically correct Otsu threshold lies between 20.0 and 200.0.
pub fn make_bimodal_image() -> Image<Backend, 3> {
    let device: <Backend as BurnBackend>::Device = Default::default();
    let values: Vec<f32> = (0..64)
        .map(|i| if i < 32 { 20.0_f32 } else { 200.0_f32 })
        .collect();
    let td = TensorData::new(values, Shape::new([4, 4, 4]));
    let tensor = Tensor::<Backend, 3>::from_data(td, &device);
    Image::new(
        tensor,
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
    )
}

/// Build a 6×6×6 trimodal image for multi-Otsu tests.
///
/// Voxels are split into three equal groups with intensities 30, 130, 230.
pub fn make_trimodal_image() -> Image<Backend, 3> {
    let device: <Backend as BurnBackend>::Device = Default::default();
    let n = 6 * 6 * 6; // 216
    let values: Vec<f32> = (0..n)
        .map(|i| {
            if i < n / 3 {
                30.0_f32
            } else if i < 2 * n / 3 {
                130.0_f32
            } else {
                230.0_f32
            }
        })
        .collect();
    let td = TensorData::new(values, Shape::new([6, 6, 6]));
    let tensor = Tensor::<Backend, 3>::from_data(td, &device);
    Image::new(
        tensor,
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
    )
}

/// Build a 5×5×5 image with a high-intensity sphere at the centre.
///
/// Centre voxel (2,2,2) and its 6 face-adjacent neighbours have intensity
/// 200.0; all other voxels have intensity 10.0.
pub fn make_sphere_image() -> Image<Backend, 3> {
    let device: <Backend as BurnBackend>::Device = Default::default();
    let (nz, ny, nx) = (5usize, 5usize, 5usize);
    let mut values = vec![10.0_f32; nz * ny * nx];
    let high_indices: &[(usize, usize, usize)] = &[
        (2, 2, 2), // centre
        (1, 2, 2), // −Z
        (3, 2, 2), // +Z
        (2, 1, 2), // −Y
        (2, 3, 2), // +Y
        (2, 2, 1), // −X
        (2, 2, 3), // +X
    ];
    for &(z, y, x) in high_indices {
        values[z * ny * nx + y * nx + x] = 200.0;
    }
    let td = TensorData::new(values, Shape::new([nz, ny, nx]));
    let tensor = Tensor::<Backend, 3>::from_data(td, &device);
    Image::new(
        tensor,
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
    )
}

/// Build a 4×4×4 binary image: first 32 voxels = 1.0 (foreground),
/// remaining 32 = 0.0 (background).  Used for distance-transform tests.
pub fn make_binary_image() -> Image<Backend, 3> {
    let device: <Backend as BurnBackend>::Device = Default::default();
    let values: Vec<f32> = (0..64)
        .map(|i| if i < 32 { 1.0_f32 } else { 0.0_f32 })
        .collect();
    let td = TensorData::new(values, Shape::new([4, 4, 4]));
    let tensor = Tensor::<Backend, 3>::from_data(td, &device);
    Image::new(
        tensor,
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
    )
}

/// Build a 4×4×4 image with a smooth ramp 0..63 for watershed / gradient
/// tests.
pub fn make_ramp_image() -> Image<Backend, 3> {
    let device: <Backend as BurnBackend>::Device = Default::default();
    let values: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let td = TensorData::new(values, Shape::new([4, 4, 4]));
    let tensor = Tensor::<Backend, 3>::from_data(td, &device);
    Image::new(
        tensor,
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
    )
}

// ── Helper: default SegmentArgs ───────────────────────────────────────────────

pub fn default_args(
    input: std::path::PathBuf,
    output: std::path::PathBuf,
    method: &str,
) -> SegmentArgs {
    SegmentArgs {
        input,
        output,
        method: method.to_string(),
        ..Default::default()
    }
}
