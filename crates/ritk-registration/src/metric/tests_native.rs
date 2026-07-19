use super::{lncc_loss_native, mse_value_native, ncc_loss_native, ngf_value_native};
use coeus_core::SequentialBackend;
use ritk_filter::GaussianSigma;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use ritk_transform::transform::affine::AtlasAffineTransform;

fn ramp() -> Image<f32, SequentialBackend, 3> {
    Image::from_flat_on(
        (0..27).map(|value| value as f32).collect(),
        [3, 3, 3],
        Point::origin(),
        Spacing::uniform(1.0),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("ramp shape and values agree")
}

#[test]
fn native_metrics_score_identical_volumes() {
    let image = ramp();
    let transform = AtlasAffineTransform::<SequentialBackend, 3>::identity(None);

    assert_eq!(mse_value_native(&image, &image, &transform), 0.0);
    let ncc = ncc_loss_native(&image, &image, &transform);
    assert!((ncc + 1.0).abs() <= 16.0 * f32::EPSILON, "NCC loss {ncc}");
    let lncc = lncc_loss_native(
        &image,
        &image,
        &transform,
        GaussianSigma::new_unchecked(1.0),
        1.0e-5,
    );
    assert!(lncc.is_finite() && lncc < -0.9, "LNCC loss {lncc}");
    let ngf = ngf_value_native(&image, &image, &transform, None, None);
    assert!(ngf.is_finite() && ngf > 0.9, "NGF score {ngf}");
}
