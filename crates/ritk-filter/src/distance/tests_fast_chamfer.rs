use super::{ApproximateSignedDistanceMapFilter, FastChamferDistanceFilter};
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

type B = coeus_core::SequentialBackend;

fn img(data: Vec<f32>, dims: [usize; 3]) -> Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(data, dims)
}

/// A single zero seed in a far field propagates the face weight `0.92644` per
/// step (two passes resolve both directions): `[20,20,0,20,20] → [2w,w,0,w,2w]`.
#[test]
fn fast_chamfer_propagates_face_weight() {
    let f = FastChamferDistanceFilter {
        maximum_distance: 100.0,
    };
    let out = f.apply(&img(vec![20.0, 20.0, 0.0, 20.0, 20.0], [1, 1, 5]));
    let (v, _) = extract_vec_infallible(&out);
    let w = 0.92644f32;
    let expected = [2.0 * w, w, 0.0, w, 2.0 * w];
    for (g, e) in v.iter().zip(expected) {
        assert!((g - e).abs() < 1e-5, "got {g}, want {e}");
    }
}

/// The approximate signed distance map of a binary blob is negative inside,
/// positive outside, with the boundary near zero.
#[test]
fn approximate_signed_distance_sign_convention() {
    let dims = [1, 7, 7];
    let n: usize = dims.iter().product();
    let mut data = vec![0.0f32; n];
    for y in 2..5 {
        for x in 2..5 {
            data[(y) * 7 + x] = 1.0;
        }
    }
    let out = ApproximateSignedDistanceMapFilter::default()
        .apply(&img(data.clone(), dims))
        .expect("infallible: validated precondition");
    let (v, _) = extract_vec_infallible(&out);
    let center = 3 * 7 + 3; // inside the blob
    let corner = 0; // far outside
    assert!(v[center] < 0.0, "inside is negative, got {}", v[center]);
    assert!(v[corner] > 0.0, "outside is positive, got {}", v[corner]);
    // deepest inside is the most negative interior value
    assert!(
        v[center] <= v[2 * 7 + 2] + 1e-6,
        "blob center is the deepest"
    );
}

/// Output geometry matches the input.
#[test]
fn approximate_signed_distance_preserves_geometry() {
    let dims = [2, 4, 5];
    let n: usize = dims.iter().product();
    let mut data = vec![0.0f32; n];
    data[20 + 2 * 5 + 2] = 1.0; // z=1, y=2, x=2: flat index = 1*20 + 2*5 + 2
    let out = ApproximateSignedDistanceMapFilter::default()
        .apply(&img(data, dims))
        .expect("infallible: validated precondition");
    assert_eq!(out.shape(), dims);
    assert_eq!(out.spacing()[0], 1.0);
}
