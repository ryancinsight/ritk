use coeus_core::SequentialBackend;
use ritk_filter::{
    Colormap, LabelMapContourOverlayFilter, LabelOverlayFilter, LabelToRGBFilter,
    ScalarToRGBColormapFilter,
};
use ritk_image::{ColorVolume, Image};
use ritk_spatial::{Direction, Point, Spacing};

type B = SequentialBackend;

fn native_image(data: Vec<f32>, dims: [usize; 3]) -> Image<f32, B, 3> {
    Image::from_flat_on(
        data,
        dims,
        Point::origin(),
        Spacing::uniform(1.0),
        Direction::identity(),
        &B::default(),
    )
    .expect("test fixture dimensions match its data length")
}

fn assert_rgb_components(volume: &ColorVolume<f32, B, 3>, expected: [&[f32]; 3]) {
    let actual = volume.data_cow_on(&B::default());
    let voxel_count = expected[0].len();
    assert_eq!(expected[1].len(), voxel_count, "green channel length");
    assert_eq!(expected[2].len(), voxel_count, "blue channel length");
    assert_eq!(actual.len(), voxel_count * 3, "interleaved output length");
    for (index, voxel) in actual.chunks_exact(3).enumerate() {
        assert_eq!(
            voxel,
            &[expected[0][index], expected[1][index], expected[2][index]],
            "voxel {index}"
        );
    }
}

/// Grey colormap on a `[10,20,30,40,50]` ramp → `[0,63,127,191,255]` per channel
/// (normalize by image min/max, ×255, floor — `0.25·255 = 63.75 → 63`).
#[test]
fn grey_ramp_matches_itk_truncation() {
    let img = native_image(vec![10.0, 20.0, 30.0, 40.0, 50.0], [1, 1, 5]);
    let out = ScalarToRGBColormapFilter::new(Colormap::Grey)
        .apply(&img, &B::default())
        .unwrap();
    let expected = [0.0f32, 63.0, 127.0, 191.0, 255.0];
    assert_rgb_components(&out, [&expected, &expected, &expected]);
}

/// Red colormap puts the ramp in channel 0 only.
#[test]
fn red_colormap_channel_selection() {
    let img = native_image(vec![10.0, 20.0, 30.0, 40.0, 50.0], [1, 1, 5]);
    let out = ScalarToRGBColormapFilter::new(Colormap::Red)
        .apply(&img, &B::default())
        .unwrap();
    let red = [0.0, 63.0, 127.0, 191.0, 255.0];
    let black = [0.0; 5];
    assert_rgb_components(&out, [&red, &black, &black]);
}

/// A constant image maps to all-zero (`range = 0 → t = 0`).
#[test]
fn constant_image_maps_to_zero() {
    let img = native_image(vec![7.0; 8], [2, 2, 2]);
    let out = ScalarToRGBColormapFilter::new(Colormap::Grey)
        .apply(&img, &B::default())
        .unwrap();
    let black = [0.0; 8];
    assert_rgb_components(&out, [&black, &black, &black]);
}

/// Unsupported (piecewise) colormaps are rejected, not approximated.
#[test]
fn unsupported_colormap_rejected() {
    assert!(Colormap::from_name("jet").is_err());
    assert!(Colormap::from_name("hot").is_err());
    assert_eq!(Colormap::from_name("Grey").unwrap(), Colormap::Grey);
    assert_eq!(Colormap::from_name("gray").unwrap(), Colormap::Grey);
}

/// LabelToRGB: background→black, labels 1..7 use the ITK table, and the table
/// cycles with period 30 (label 31 == label 1's colour). Pinned by sitk probe.
#[test]
fn label_to_rgb_matches_itk_table_and_cycles() {
    let img = native_image(vec![0.0, 1.0, 2.0, 5.0, 7.0, 30.0, 31.0], [1, 1, 7]);
    let out = LabelToRGBFilter::new(0).apply(&img, &B::default()).unwrap();
    let c = out.data_cow_on(&B::default());
    // (r,g,b) per voxel.
    let rgb = |i: usize| [c[3 * i], c[3 * i + 1], c[3 * i + 2]];
    assert_eq!(rgb(0), [0.0, 0.0, 0.0]); // background
    assert_eq!(rgb(1), [0.0, 205.0, 0.0]); // label 1
    assert_eq!(rgb(2), [0.0, 0.0, 255.0]); // label 2
    assert_eq!(rgb(3), [255.0, 127.0, 0.0]); // label 5
    assert_eq!(rgb(4), [138.0, 43.0, 226.0]); // label 7
    assert_eq!(rgb(5), [255.0, 0.0, 0.0]); // label 30 (table[29])
    assert_eq!(rgb(6), [0.0, 205.0, 0.0]); // label 31 wraps to table[0]
}

/// LabelOverlay: background passes grayscale through; labels alpha-blend with
/// the colour table at `opacity=0.5` (floor). Pinned by sitk probe:
/// gray=[100,100,200,200], label=[0,1,0,2] → [[100,100,100],[50,152,50],
/// [200,200,200],[100,100,227]].
#[test]
fn label_overlay_blends_with_table() {
    let gray = native_image(vec![100.0, 100.0, 200.0, 200.0], [1, 1, 4]);
    let lab = native_image(vec![0.0, 1.0, 0.0, 2.0], [1, 1, 4]);
    let out = LabelOverlayFilter::new(0.5, 0)
        .apply(&gray, &lab, &B::default())
        .unwrap();
    let c = out.data_cow_on(&B::default());
    let rgb = |i: usize| [c[3 * i], c[3 * i + 1], c[3 * i + 2]];
    assert_eq!(rgb(0), [100.0, 100.0, 100.0]); // background
    assert_eq!(rgb(1), [50.0, 152.0, 50.0]); // label1: ½·100+½·[0,205,0]
    assert_eq!(rgb(2), [200.0, 200.0, 200.0]); // background
    assert_eq!(rgb(3), [100.0, 100.0, 227.0]); // label2: ½·200+½·[0,0,255], 227.5→227
}

/// Opacity 1.0 yields the pure label colour over labelled voxels.
#[test]
fn label_overlay_full_opacity_is_label_color() {
    let gray = native_image(vec![100.0, 200.0], [1, 1, 2]);
    let lab = native_image(vec![1.0, 2.0], [1, 1, 2]);
    let out = LabelOverlayFilter::new(1.0, 0)
        .apply(&gray, &lab, &B::default())
        .unwrap();
    let c = out.data_cow_on(&B::default());
    assert_eq!([c[0], c[1], c[2]], [0.0, 205.0, 0.0]);
    assert_eq!([c[3], c[4], c[5]], [0.0, 0.0, 255.0]);
}

/// LabelMapContourOverlay default geometry on an 8×8 (z=1) scene, vs the exact
/// `sitk.LabelMapContourOverlay` RGB output (uint8 feature = arange 0..63).
#[test]
fn label_map_contour_overlay_matches_sitk() {
    let gray: Vec<f32> = (0..64).map(|v| v as f32).collect();
    let mut lab = vec![0.0_f32; 64];
    for y in 1..4 {
        for x in 1..4 {
            lab[y * 8 + x] = 1.0;
        }
    }
    for y in 4..7 {
        for x in 4..7 {
            lab[y * 8 + x] = 2.0;
        }
    }
    let gi = native_image(gray, [1, 8, 8]);
    let li = native_image(lab, [1, 8, 8]);
    let out = LabelMapContourOverlayFilter::new(0.5, 0)
        .apply(&gi, &li, &B::default())
        .unwrap();
    let exp_r: Vec<f32> = [
        0, 1, 2, 3, 2, 5, 6, 7, 8, 9, 10, 11, 6, 13, 14, 15, 16, 17, 18, 19, 10, 21, 22, 23, 24,
        25, 26, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 37, 38, 39, 40, 41, 42, 21, 44, 45, 46, 47,
        48, 49, 50, 25, 52, 53, 54, 55, 56, 57, 58, 29, 60, 61, 62, 63,
    ]
    .iter()
    .map(|&v| v as f32)
    .collect();
    let exp_g: Vec<f32> = [
        0, 1, 2, 3, 104, 5, 6, 7, 8, 9, 10, 11, 108, 13, 14, 15, 16, 17, 18, 19, 112, 21, 22, 23,
        24, 25, 26, 13, 14, 14, 15, 15, 118, 119, 119, 17, 120, 37, 38, 39, 40, 41, 42, 21, 44, 45,
        46, 47, 48, 49, 50, 25, 52, 53, 54, 55, 56, 57, 58, 29, 60, 61, 62, 63,
    ]
    .iter()
    .map(|&v| v as f32)
    .collect();
    let exp_b: Vec<f32> = [
        0, 1, 2, 3, 2, 5, 6, 7, 8, 9, 10, 11, 6, 13, 14, 15, 16, 17, 18, 19, 10, 21, 22, 23, 24,
        25, 26, 141, 141, 142, 142, 143, 16, 16, 17, 145, 18, 37, 38, 39, 40, 41, 42, 149, 44, 45,
        46, 47, 48, 49, 50, 153, 52, 53, 54, 55, 56, 57, 58, 157, 60, 61, 62, 63,
    ]
    .iter()
    .map(|&v| v as f32)
    .collect();
    assert_rgb_components(&out, [&exp_r, &exp_g, &exp_b]);
}
