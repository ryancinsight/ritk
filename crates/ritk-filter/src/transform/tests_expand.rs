use super::*;
use ritk_image::test_support as ts;

type B = coeus_core::SequentialBackend;

fn vals(image: &Image<f32, B, 3>) -> Vec<f32> {
    image
        .data_slice()
        .expect("invariant: contiguous host storage")
        .to_vec()
}

/// Expand a 1-D ramp by factor 2 along X. Matches the ITK/sitk grid exactly
/// (verified against `sitk.Expand`): edge-clamp linear interpolation at the
/// half-shifted output grid.
#[test]
fn expand_x_factor_two_matches_itk_grid() {
    let img = ts::make_image::<f32, B, 3>(vec![0.0, 10.0, 20.0, 30.0], [1, 1, 4]);
    let out = ExpandImageFilter::new([1, 1, 2]).apply(&img);
    assert_eq!(out.shape(), [1, 1, 8]);
    let expected = [0.0f32, 2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 30.0];
    for (got, exp) in vals(&out).iter().zip(expected) {
        assert!(
            (got - exp).abs() < 1e-4,
            "expand: got {got}, expected {exp}"
        );
    }
}

/// Factor 1 on every axis is the identity (continuous index `ci(j)=j`).
#[test]
fn expand_factor_one_is_identity() {
    let img = ts::make_image::<f32, B, 3>((0..24).map(|i| i as f32).collect(), [2, 3, 4]);
    let out = ExpandImageFilter::new([1, 1, 1]).apply(&img);
    assert_eq!(out.shape(), [2, 3, 4]);
    assert_eq!(vals(&out), vals(&img));
}

/// Output spacing is `spacing/factor` and origin is shifted by half a voxel.
#[test]
fn expand_updates_spacing_and_origin() {
    use ritk_spatial::{Direction, Point, Spacing};
    let t = ritk_image::tensor::Tensor::<f32, B>::from_slice([1, 1, 2], &[0.0_f32, 4.0]);
    let img = Image::new(
        t,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 2.0]),
        Direction::identity(),
    )
    .expect("invariant: fixture tensor has the declared rank");
    let out = ExpandImageFilter::new([1, 1, 2]).apply(&img);
    // X spacing 2.0/2 = 1.0; X origin 0 - 0.5*2 + 0.5*1 = -0.5.
    assert!((out.spacing().to_array()[2] - 1.0).abs() < 1e-9);
    assert!((out.origin().to_array()[2] - (-0.5)).abs() < 1e-9);
}
