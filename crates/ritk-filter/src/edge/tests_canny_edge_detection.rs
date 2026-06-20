//! Tests for ITK-exact CannyEdgeDetection. Bit-exactness vs sitk.CannyEdgeDetection
//! is covered by the Python cmake-parity suite; these assert structural invariants.

use super::CannyEdgeDetectionImageFilter;
use burn_ndarray::NdArray;
use ritk_image::test_support as ts;
use ritk_image::Image;

type B = NdArray<f32>;

fn make(v: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    ts::make_image::<B, 3>(v, dims)
}
fn voxels(img: &Image<B, 3>) -> Vec<f32> {
    img.data_slice().into_owned()
}

/// A square produces a closed ring of edge voxels (output is binary 0/1).
#[test]
fn square_produces_binary_edge_ring() {
    let (ny, nx) = (24usize, 24usize);
    let mut f = vec![0.0f32; ny * nx];
    for y in 6..18 {
        for x in 6..18 {
            f[y * nx + x] = 100.0;
        }
    }
    let out = CannyEdgeDetectionImageFilter {
        variance: 2.0,
        maximum_error: 0.01,
        lower_threshold: 2.0,
        upper_threshold: 8.0,
    }
    .apply(&make(f, [1, ny, nx]));
    let v = voxels(&out);
    // Output is strictly binary.
    for &x in &v {
        assert!(x == 0.0 || x == 1.0, "non-binary edge value {}", x);
    }
    // There is a non-trivial number of edges (the square boundary).
    let edges: usize = v.iter().filter(|&&x| x > 0.5).count();
    assert!(edges > 8, "expected an edge ring, got {} edge voxels", edges);
}

/// A uniform image has no gradient, hence no edges.
#[test]
fn uniform_image_has_no_edges() {
    let out = CannyEdgeDetectionImageFilter {
        variance: 1.0,
        ..Default::default()
    }
    .apply(&make(vec![50.0; 16], [1, 4, 4]));
    assert!(voxels(&out).iter().all(|&x| x == 0.0), "uniform image must have no edges");
}
