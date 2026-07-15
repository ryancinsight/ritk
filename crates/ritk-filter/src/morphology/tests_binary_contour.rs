//! Tests for binary_contour
//! Extracted to keep the 500-line structural limit.
use super::*;
use crate::native_support::LegacyBurnBackend;
use ritk_image::test_support as ts;

type B = LegacyBurnBackend;

fn make_image(data: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
    ts::make_image::<B, 3>(data, shape)
}

fn voxels(img: &Image<B, 3>) -> Vec<f32> {
    img.data_slice().into_owned()
}

/// All-background → all-zero output.
#[test]
fn all_background_zero() {
    let img = make_image(vec![0.0f32; 27], [3, 3, 3]);
    let out = BinaryContourImageFilter::default().apply(&img).unwrap();
    assert!(voxels(&out).iter().all(|&v| v == 0.0));
}

/// All-foreground solid block: only the outer shell is a border.
/// 3×3×3 all-fg: corner/edge/face voxels are borders; center (1,1,1) is interior (6-conn).
#[test]
fn fully_solid_block_has_empty_contour() {
    // A 3×3×3 all-foreground block has NO background voxel anywhere, so no voxel
    // has an in-bounds background neighbour. Out-of-bounds (image-edge) is NOT
    // background, so the contour is empty — matching `sitk.BinaryContour`, which
    // leaves a full-foreground image all-zero.
    let img = make_image(vec![1.0f32; 27], [3, 3, 3]);
    let out = BinaryContourImageFilter::default().apply(&img).unwrap();
    assert!(
        voxels(&out).iter().all(|&x| x == 0.0),
        "full-foreground block must have an empty contour"
    );
}

/// A 3×3×3 foreground block surrounded by a background border (5×5×5) yields the
/// block's outer shell as contour; only its centre is interior.
#[test]
fn block_in_background_yields_outer_shell() {
    let mut data = vec![0.0f32; 125];
    for z in 1..4 {
        for y in 1..4 {
            for x in 1..4 {
                data[z * 25 + y * 5 + x] = 1.0;
            }
        }
    }
    let out = BinaryContourImageFilter::default().apply(&make_image(data, [5, 5, 5]));
    let v = voxels(&out.unwrap());
    // Block centre (2,2,2) = 62 is interior; the 26 surrounding fg voxels are shell.
    assert_eq!(v[62], 0.0, "block centre is interior");
    let shell: f32 = (1..4)
        .flat_map(|z| (1..4).flat_map(move |y| (1..4).map(move |x| z * 25 + y * 5 + x)))
        .filter(|&i| i != 62)
        .map(|i| v[i])
        .sum();
    assert_eq!(shell, 26.0, "all 26 shell voxels are contour");
}

/// 5×5×5 block: all-fg. Center voxel (2,2,2) is interior → 0 (6-connected).
#[test]
fn five_cube_center_is_interior_6conn() {
    let img = make_image(vec![1.0f32; 125], [5, 5, 5]);
    let out = BinaryContourImageFilter::new(Connectivity::Face6, 1.0)
        .apply(&img)
        .unwrap();
    let v = voxels(&out);
    // Center voxel index = 2*25+2*5+2 = 62
    assert_eq!(v[62], 0.0, "center of 5×5×5 should be interior (6-conn)");
}

/// Single foreground voxel in a background image is a border voxel.
#[test]
fn single_fg_voxel_is_border() {
    let mut data = vec![0.0f32; 27];
    data[13] = 1.0; // center of 3×3×3
    let img = make_image(data, [3, 3, 3]);
    let out = BinaryContourImageFilter::default().apply(&img).unwrap();
    let v = voxels(&out);
    assert!((v[13] - 1.0).abs() < 1e-5, "single fg voxel must be border");
    let others: f32 = v
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != 13)
        .map(|(_, &x)| x)
        .sum();
    assert_eq!(others, 0.0);
}

/// Spatial metadata preserved.
#[test]
fn preserves_metadata() {
    let img = make_image(vec![0.0f32; 8], [2, 2, 2]);
    let out = BinaryContourImageFilter::default().apply(&img).unwrap();
    assert_eq!(out.shape(), [2, 2, 2]);
    assert_eq!(*out.origin(), *img.origin());
    assert_eq!(*out.spacing(), *img.spacing());
}

/// 26-connected: center of 5×5×5 all-fg block is also interior.
#[test]
fn five_cube_center_interior_26conn() {
    let img = make_image(vec![1.0f32; 125], [5, 5, 5]);
    let out = BinaryContourImageFilter::new(Connectivity::Vertex26, 1.0)
        .apply(&img)
        .unwrap();
    let v = voxels(&out);
    assert_eq!(v[62], 0.0, "center of 5×5×5 should be interior (26-conn)");
}
