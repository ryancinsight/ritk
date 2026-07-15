use super::*;
use crate::native_support::LegacyBurnBackend;
use ritk_interpolation::LinearInterpolator;
use ritk_spatial::{Direction2, Point2, Spacing2};
use ritk_tensor_ops::extract_vec_infallible;
use ritk_transform::affine::translation::TranslationTransform;

type TestBackend = LegacyBurnBackend;

/// Bilinear interpolation at exact integer offset yields ≥ this value.
const NEAR_ONE_TOL: f32 = 0.9;
/// Outside-patch region expected to be ≤ this value.
const NEAR_ZERO_TOL: f32 = 0.1;

#[test]
fn test_resample_translation_planar() {
    let device = Default::default();

    // 1. Create a 10x10 image with a 2x2 square in the center (4,4) to (5,5)
    let mut data = vec![0.0; 100];
    data[4 * 10 + 4] = 1.0;
    data[4 * 10 + 5] = 1.0;
    data[5 * 10 + 4] = 1.0;
    data[5 * 10 + 5] = 1.0;

    let tensor = ritk_image::tensor::Tensor::<TestBackend, 2>::from_floats(
        ritk_image::tensor::TensorData::new(data, Shape::new([10, 10])),
        &device,
    );

    let origin = Point2::new([0.0, 0.0]);
    let spacing = Spacing2::new([1.0, 1.0]);
    let direction = Direction2::identity();

    let image = Image::new(tensor, origin, spacing, direction);

    // 2. Define Transform: shift content by +1 row (Y) and +2 cols (X).
    // The offset is in the image's axis-major physical convention [row, col] =
    // [y, x] (matching world_to_index_tensor / index_to_world_tensor), so the
    // output→input map subtracts [1, 2]: output reads input one row up and two
    // columns left, moving the square down one row and right two columns.
    let offset = ritk_image::tensor::Tensor::<TestBackend, 1>::from_floats([-1.0, -2.0], &device);
    let transform = TranslationTransform::<TestBackend, 2>::new(offset);

    // 3. Define Interpolator
    let interpolator = LinearInterpolator::new();

    // 4. Create Resample Filter
    let filter = ResampleImageFilter::new_from_reference(&image, transform, interpolator);

    // 5. Apply
    let result = filter.apply(&image);

    // 6. Verify
    let (data, _) = extract_vec_infallible(&result);
    let slice = data.as_slice();

    // Check index corresponding to physical coordinates (x=6, y=5) -> index 5*10 + 6 = 56
    assert!(slice[56] > NEAR_ONE_TOL);
    // Check index corresponding to physical coordinates (x=7, y=5) -> index 5*10 + 7 = 57
    assert!(slice[57] > NEAR_ONE_TOL);
    // Check index corresponding to physical coordinates (x=6, y=6) -> index 6*10 + 6 = 66
    assert!(slice[66] > NEAR_ONE_TOL);
    // Check index corresponding to physical coordinates (x=7, y=6) -> index 6*10 + 7 = 67
    assert!(slice[67] > NEAR_ONE_TOL);

    // Check original location (4,4) -> 44 should be 0
    assert!(slice[44] < NEAR_ZERO_TOL);
}

/// Regression: identity resampling of an anisotropic 3-D grid (here a z = 1
/// "2-D promoted" volume with z-spacing ≠ in-plane spacing) must reproduce the
/// input exactly. The previous `indices_to_physical` paired innermost-first
/// index columns with axis-major spacing by position, multiplying the x index
/// by the z spacing; identity on a cube hid it, but a z = 1 anisotropic grid
/// collapsed every output row to a constant.
#[test]
fn test_resample_identity_anisotropic_z1_volumetric() {
    use ritk_interpolation::LinearInterpolator;
    use ritk_spatial::{Direction, Point, Spacing};
    use ritk_transform::affine::translation::TranslationTransform;

    let device = Default::default();
    // [z, y, x] = [1, 4, 5], a horizontal ramp varying along x so a column
    // collapse is immediately visible.
    let (nz, ny, nx) = (1usize, 4usize, 5usize);
    let mut data = vec![0.0f32; nz * ny * nx];
    for y in 0..ny {
        for x in 0..nx {
            data[y * nx + x] = (y * 10 + x) as f32;
        }
    }
    let tensor = ritk_image::tensor::Tensor::<TestBackend, 3>::from_floats(
        ritk_image::tensor::TensorData::new(data.clone(), Shape::new([nz, ny, nx])),
        &device,
    );
    // Anisotropic spacing: z = 1.0, y = x = 0.35 (mirrors a promoted 2-D slice).
    let image = Image::new(
        tensor,
        Point::<3>::new([0.0, 0.0, 0.0]),
        Spacing::<3>::new([1.0, 0.35, 0.35]),
        Direction::<3>::identity(),
    );

    let zero = ritk_image::tensor::Tensor::<TestBackend, 1>::from_floats([0.0, 0.0, 0.0], &device);
    let filter = ResampleImageFilter::new(
        [nz, ny, nx],
        *image.origin(),
        *image.spacing(),
        *image.direction(),
        TranslationTransform::<TestBackend, 3>::new(zero),
        LinearInterpolator::new(),
    );
    let result = filter.apply(&image);
    let (out, _) = extract_vec_infallible(&result);

    for (i, (&o, &d)) in out.as_slice().iter().zip(data.iter()).enumerate() {
        assert!(
            (o - d).abs() < 1e-4,
            "identity resample mismatch at flat {i}: got {o}, want {d}"
        );
    }
}
