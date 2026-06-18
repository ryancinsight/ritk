use super::IsolatedConnectedFilter;
use burn_ndarray::NdArray;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

type B = NdArray<f32>;

fn make(data: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    ts::make_image::<B, 3>(data, dims)
}

/// Two intensity-100 blobs joined by an intensity-150 bridge. With band floor
/// 50 and ceiling 200, the separating upper threshold lands just below 150, so
/// the region grown from seed1 keeps its blob (100 ≤ 149) but excludes the
/// bridge (150) and therefore the second blob/seed.
#[test]
fn isolated_connected_separates_two_blobs() {
    let (ny, nx) = (10usize, 16);
    let mut v = vec![0.0f32; ny * nx];
    let set = |v: &mut Vec<f32>, y: usize, x: usize, val: f32| v[y * nx + x] = val;
    for y in 3..=6 {
        for x in 1..=4 {
            set(&mut v, y, x, 100.0); // blob 1
        }
        for x in 11..=14 {
            set(&mut v, y, x, 100.0); // blob 2
        }
    }
    for y in 4..=5 {
        for x in 5..=10 {
            set(&mut v, y, x, 150.0); // bridge (intermediate intensity)
        }
    }
    let img = make(v, [1, ny, nx]);
    let f = IsolatedConnectedFilter {
        seed1: [0, 4, 2],
        seed2: [0, 4, 13],
        lower: 50.0,
        upper: 200.0,
        ..Default::default()
    };
    let out = f.apply(&img);
    let (ov, _) = extract_vec_infallible(&out);
    assert_eq!(ov[4 * nx + 2], 1.0, "seed1's blob must be kept");
    assert_eq!(ov[4 * nx + 13], 0.0, "seed2's blob must be isolated out");
    assert_eq!(ov[4 * nx + 7], 0.0, "the bridge must be excluded");
    // Every kept voxel is part of blob 1 (columns 1..=4).
    for y in 0..ny {
        for x in 0..nx {
            if ov[y * nx + x] != 0.0 {
                assert!(
                    (1..=4).contains(&x),
                    "kept voxel outside blob 1 at ({y},{x})"
                );
            }
        }
    }
}

/// Custom replace value is written to the kept region.
#[test]
fn isolated_connected_custom_replace_value() {
    let (ny, nx) = (6usize, 6);
    let mut v = vec![0.0f32; ny * nx];
    for y in 1..=4 {
        for x in 1..=4 {
            v[y * nx + x] = 100.0;
        }
    }
    let img = make(v, [1, ny, nx]);
    let f = IsolatedConnectedFilter {
        seed1: [0, 2, 2],
        seed2: [0, 0, 0], // background, never reached
        lower: 50.0,
        upper: 200.0,
        replace_value: 7.0,
        ..Default::default()
    };
    let out = f.apply(&img);
    let (ov, _) = extract_vec_infallible(&out);
    assert_eq!(ov[2 * nx + 2], 7.0, "kept region uses the replace value");
    assert!(ov.iter().all(|&v| v == 0.0 || v == 7.0));
}
