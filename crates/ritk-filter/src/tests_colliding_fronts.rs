use super::CollidingFrontsFilter;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

type B = coeus_core::SequentialBackend;

fn make(data: Vec<f32>, dims: [usize; 3]) -> Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(data, dims)
}

/// Two fronts collide head-on along a 1-D unit-speed line: front 1 (from x=0)
/// has gradient +1, front 2 (from x=6) has gradient âˆ’1, so the potential
/// `âˆ‡T1Â·âˆ‡T2 = âˆ’1` everywhere in the interior. Seeds are pinned to âˆ’1e-6, and the
/// whole line is the connected colliding corridor.
#[test]
fn colliding_fronts_unit_line_potential_is_minus_one() {
    let nx = 7usize;
    let speed = make(vec![1.0f32; nx], [1, 1, nx]);
    let out = CollidingFrontsFilter::new(vec![[0, 0, 0]], vec![[0, 0, 6]]).apply(&speed);
    let (ov, _) = extract_vec_infallible(&out);
    assert!(ov.iter().all(|&v| v <= 1e-6), "potential is â‰¤ 0");
    for &x in &[2usize, 3, 4] {
        assert!(
            (ov[x] + 1.0).abs() < 1e-4,
            "interior potential at {x} = {}, want -1",
            ov[x]
        );
    }
    assert!(
        (ov[0] - (-1e-6)).abs() < 1e-9,
        "seed pinned to negative epsilon"
    );
}

/// Without connectivity the raw `âˆ‡T1Â·âˆ‡T2` potential is returned (no flood-fill
/// restriction), still with the seeds pinned and the interior at âˆ’1.
#[test]
fn colliding_fronts_no_connectivity_returns_full_potential() {
    let nx = 7usize;
    let speed = make(vec![1.0f32; nx], [1, 1, nx]);
    let f = CollidingFrontsFilter {
        seeds1: vec![[0, 0, 0]],
        seeds2: vec![[0, 0, 6]],
        apply_connectivity: false,
        negative_epsilon: -1e-6,
    };
    let (ov, _) = extract_vec_infallible(&f.apply(&speed));
    assert!(
        (ov[3] + 1.0).abs() < 1e-4,
        "interior potential = {}, want -1",
        ov[3]
    );
    assert!(
        (ov[6] - (-1e-6)).abs() < 1e-9,
        "second seed pinned to negative epsilon"
    );
}

/// Output geometry equals input geometry.
#[test]
fn colliding_fronts_preserves_geometry() {
    let dims = [1usize, 6, 8];
    let n: usize = dims.iter().product();
    let f = CollidingFrontsFilter::new(vec![[0, 1, 1]], vec![[0, 4, 6]]);
    let out = f.apply(&make(vec![1.0; n], dims));
    assert_eq!(out.shape(), dims);
    assert_eq!(out.spacing()[0], 1.0);
}
