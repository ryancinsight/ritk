use super::ReinitializeLevelSetFilter;
use crate::native_support::LegacyBurnBackend;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

type B = LegacyBurnBackend;

fn img(data: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    ts::make_image::<B, 3>(data, dims)
}

/// A unit-gradient linear level set `φ(x) = x − 3.5` is already a signed distance
/// function, so reinitialization returns it unchanged.
#[test]
fn reinitialize_unit_gradient_linear_is_identity() {
    let phi: Vec<f32> = (0..7).map(|x| x as f32 - 3.5).collect();
    let out = ReinitializeLevelSetFilter::new(0.0)
        .apply(&img(phi.clone(), [1, 1, 7]))
        .unwrap();
    let (v, _) = extract_vec_infallible(&out);
    for (g, e) in v.iter().zip(&phi) {
        assert!((g - e).abs() < 1e-5, "reinit {g} != input {e}");
    }
}

/// Reinitialization normalizes the gradient: a scaled linear `φ = 2·(x − 3.5)`
/// reinitializes to the *unit* signed distance `x − 3.5`, independent of scale.
#[test]
fn reinitialize_normalizes_gradient() {
    let phi: Vec<f32> = (0..7).map(|x| 2.0 * (x as f32 - 3.5)).collect();
    let out = ReinitializeLevelSetFilter::new(0.0)
        .apply(&img(phi, [1, 1, 7]))
        .unwrap();
    let (v, _) = extract_vec_infallible(&out);
    let expected: Vec<f32> = (0..7).map(|x| x as f32 - 3.5).collect();
    for (g, e) in v.iter().zip(&expected) {
        assert!((g - e).abs() < 1e-5, "reinit {g} != unit SDF {e}");
    }
}

/// Output geometry matches the input.
#[test]
fn reinitialize_preserves_geometry() {
    let dims = [2, 4, 5];
    let n: usize = dims.iter().product();
    let phi: Vec<f32> = (0..n).map(|i| i as f32 - (n as f32 / 2.0)).collect();
    let out = ReinitializeLevelSetFilter::new(0.0)
        .apply(&img(phi, dims))
        .unwrap();
    assert_eq!(out.shape(), dims);
    assert_eq!(out.spacing()[0], 1.0);
}
