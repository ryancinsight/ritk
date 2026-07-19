use super::ErodeObjectMorphologyFilter;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

type B = coeus_core::SequentialBackend;

fn make(data: Vec<f32>, dims: [usize; 3]) -> Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(data, dims)
}

/// A solid 7³ object touching every border erodes (radius 1) to its inner 3³
/// core. Boundary voxels are exactly those with a coordinate in {0, 6} (an
/// out-of-image neighbour counts as non-object); painting their 3³ footprints
/// removes a box-dilation-by-1 of that set, i.e. every voxel with a coordinate
/// in {0, 1, 5, 6}. The survivors are precisely the voxels whose every
/// coordinate lies in {2, 3, 4} — 27 voxels.
#[test]
fn erode_object_solid_block_to_inner_core() {
    let n = 7usize;
    let out = ErodeObjectMorphologyFilter::default().apply(&make(vec![1.0; n * n * n], [n, n, n]));
    let (ov, _) = extract_vec_infallible(&out);
    let inner = |c: usize| (2..=4).contains(&c);
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let want = if inner(z) && inner(y) && inner(x) {
                    1.0
                } else {
                    0.0
                };
                let got = ov[(z * n + y) * n + x];
                assert_eq!(got, want, "voxel ({z},{y},{x}): got {got}, want {want}");
            }
        }
    }
    assert_eq!(ov.iter().filter(|&&v| v == 1.0).count(), 27);
}

/// Custom object/background values are honoured, and non-object voxels are left
/// untouched where no boundary footprint reaches them.
#[test]
fn erode_object_custom_values() {
    // A single isolated object voxel (value 5) in a field of 9: it is a boundary
    // voxel (all neighbours non-object), so its r=1 footprint — including itself
    // — becomes background 9, leaving the whole image at 9.
    let (nz, ny, nx) = (5usize, 5, 5);
    let n = nz * ny * nx;
    let mut vals = vec![9.0f32; n];
    vals[(2 * ny + 2) * nx + 2] = 5.0;
    let out =
        ErodeObjectMorphologyFilter::new([1, 1, 1], 5.0, 9.0).apply(&make(vals, [nz, ny, nx]));
    let (ov, _) = extract_vec_infallible(&out);
    assert!(
        ov.iter().all(|&v| v == 9.0),
        "isolated object must be eroded away"
    );
}

/// z=1 (promoted 2-D) regression: the size-1 z axis must not be read as a
/// missing border. A 1×5×5 all-object slice erodes (radius 1) as pure 2-D —
/// boundary voxels are those with y or x in {0,4}; painting their footprints
/// removes y or x in {0,1,3,4}, leaving only (0,2,2). If the z axis were scanned
/// with radius 1, every voxel would be flagged boundary and the whole slice
/// would erode away.
#[test]
fn erode_object_z1_is_two_dimensional() {
    let (nz, ny, nx) = (1usize, 5, 5);
    let out = ErodeObjectMorphologyFilter::default().apply(&make(vec![1.0; ny * nx], [nz, ny, nx]));
    let (ov, _) = extract_vec_infallible(&out);
    assert_eq!(
        ov.iter().filter(|&&v| v == 1.0).count(),
        1,
        "only the 2-D centre voxel survives"
    );
    assert_eq!(ov[2 * nx + 2], 1.0, "centre (0,2,2) must remain object");
}

/// An image with no object voxels is returned unchanged.
#[test]
fn erode_object_no_object_is_identity() {
    let dims = [4usize, 5, 6];
    let n: usize = dims.iter().product();
    let vals: Vec<f32> = (0..n).map(|i| 2.0 + (i % 3) as f32).collect(); // never 1.0
    let img = make(vals.clone(), dims);
    let out = ErodeObjectMorphologyFilter::default().apply(&img);
    let (ov, _) = extract_vec_infallible(&out);
    assert_eq!(ov, vals, "no object voxels → identity");
    assert_eq!(out.shape(), dims);
    assert_eq!(out.spacing()[0], img.spacing()[0]);
}
