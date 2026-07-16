use super::*;
use coeus_core::SequentialBackend;
use ritk_image::native::Image;
use ritk_interpolation::LinearInterpolator;
use ritk_spatial::{Direction2, Point2, Spacing2};
use ritk_transform::affine::translation::TranslationTransform;

type TestBackend = SequentialBackend;

/// Bilinear interpolation at exact integer offset yields ≥ this value.
const NEAR_ONE_TOL: f32 = 0.9;
/// Outside-patch region expected to be ≤ this value.
const NEAR_ZERO_TOL: f32 = 0.1;

fn make_image(data: Vec<f32>, shape: [usize; 2]) -> Image<f32, TestBackend, 2> {
    Image::from_flat_on(
        data,
        shape,
        Point2::new([0.0, 0.0]),
        Spacing2::new([1.0, 1.0]),
        Direction2::identity(),
        &TestBackend,
    )
    .expect("valid test image")
}

#[test]
fn test_resample_translation_planar() {
    // 1. Create a 10x10 image with a 2x2 square in the center (4,4) to (5,5)
    let mut data = vec![0.0; 100];
    data[4 * 10 + 4] = 1.0;
    data[4 * 10 + 5] = 1.0;
    data[5 * 10 + 4] = 1.0;
    data[5 * 10 + 5] = 1.0;

    let image = make_image(data, [10, 10]);

    // 2. Define Transform: shift content by +1 row (Y) and +2 cols (X).
    // The offset is in the image's axis-major physical convention [row, col] =
    // [y, x] (matching world_to_index_native / index_to_world_native), so the
    // output→input map subtracts [1, 2]: output reads input one row up and two
    // columns left, moving the square down one row and right two columns.
    let offset = Tensor::<f32, TestBackend>::from_slice_on([2], &[-1.0, -2.0], &TestBackend);
    let transform = TranslationTransform::<TestBackend, 2>::new(offset);

    // 3. Define Interpolator
    let interpolator = LinearInterpolator::new();

    // 4. Create Resample Filter
    let filter = ResampleImageFilter::new_from_reference(&image, transform, interpolator);

    // 5. Apply
    let result = filter.apply(&image, &TestBackend).expect("resample apply");

    // 6. Verify
    let slice = result.data_slice().expect("contiguous result");

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
    use ritk_spatial::{Direction, Point, Spacing};

    // [z, y, x] = [1, 4, 5], a horizontal ramp varying along x so a column
    // collapse is immediately visible.
    let (nz, ny, nx) = (1usize, 4usize, 5usize);
    let mut data = vec![0.0f32; nz * ny * nx];
    for y in 0..ny {
        for x in 0..nx {
            data[y * nx + x] = (y * 10 + x) as f32;
        }
    }
    let image = Image::from_flat_on(
        data.clone(),
        [nz, ny, nx],
        Point::<3>::new([0.0, 0.0, 0.0]),
        Spacing::<3>::new([1.0, 0.35, 0.35]),
        Direction::<3>::identity(),
        &TestBackend,
    )
    .expect("valid test image");

    let zero = Tensor::<f32, TestBackend>::from_slice_on([3], &[0.0, 0.0, 0.0], &TestBackend);
    let filter = ResampleImageFilter::new(
        [nz, ny, nx],
        *image.origin(),
        *image.spacing(),
        *image.direction(),
        TranslationTransform::<TestBackend, 3>::new(zero),
        LinearInterpolator::new(),
    );
    let result = filter.apply(&image, &TestBackend).expect("resample apply");
    let out = result.data_slice().expect("contiguous result");

    for (i, (&o, &d)) in out.iter().zip(data.iter()).enumerate() {
        assert!(
            (o - d).abs() < 1e-4,
            "identity resample mismatch at flat {i}: got {o}, want {d}"
        );
    }
}
