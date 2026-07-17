use super::*;
use coeus_core::SequentialBackend;
use ritk_image::test_support as ts;

type B = SequentialBackend;

/// Grey colormap on a `[10,20,30,40,50]` ramp → `[0,63,127,191,255]` per channel
/// (normalize by image min/max, ×255, floor — `0.25·255 = 63.75 → 63`).
#[test]
fn grey_ramp_matches_itk_truncation() {
    let img = ts::burn_compat::make_image::<B, 3>(vec![10.0, 20.0, 30.0, 40.0, 50.0], [1, 1, 5]);
    let out = ScalarToRGBColormapFilter::new(Colormap::Grey)
        .apply(&img, &B::default())
        .unwrap();
    let comps = out.into_component_buffers();
    let expected = [0.0f32, 63.0, 127.0, 191.0, 255.0];
    for (c, channel) in comps.iter().enumerate().take(3) {
        assert_eq!(channel.as_slice(), expected.as_slice(), "channel {c}");
    }
}

/// Red colormap puts the ramp in channel 0 only.
#[test]
fn red_colormap_channel_selection() {
    let img = ts::burn_compat::make_image::<B, 3>(vec![10.0, 20.0, 30.0, 40.0, 50.0], [1, 1, 5]);
    let out = ScalarToRGBColormapFilter::new(Colormap::Red)
        .apply(&img, &B::default())
        .unwrap();
    let comps = out.into_component_buffers();
    assert_eq!(comps[0], vec![0.0, 63.0, 127.0, 191.0, 255.0]);
    assert_eq!(comps[1], vec![0.0; 5]);
    assert_eq!(comps[2], vec![0.0; 5]);
}

/// A constant image maps to all-zero (`range = 0 → t = 0`).
#[test]
fn constant_image_maps_to_zero() {
    let img = ts::burn_compat::make_image::<B, 3>(vec![7.0; 8], [2, 2, 2]);
    let out = ScalarToRGBColormapFilter::new(Colormap::Grey)
        .apply(&img, &B::default())
        .unwrap();
    for c in out.into_component_buffers() {
        assert!(c.iter().all(|&x| x == 0.0));
    }
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
    let img =
        ts::burn_compat::make_image::<B, 3>(vec![0.0, 1.0, 2.0, 5.0, 7.0, 30.0, 31.0], [1, 1, 7]);
    let out = LabelToRGBFilter::new(0).apply(&img, &B::default()).unwrap();
    let c = out.into_component_buffers();
    // (r,g,b) per voxel.
    let rgb = |i: usize| [c[0][i], c[1][i], c[2][i]];
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
    let gray = ts::burn_compat::make_image::<B, 3>(vec![100.0, 100.0, 200.0, 200.0], [1, 1, 4]);
    let lab = ts::burn_compat::make_image::<B, 3>(vec![0.0, 1.0, 0.0, 2.0], [1, 1, 4]);
    let out = LabelOverlayFilter::new(0.5, 0)
        .apply(&gray, &lab, &B::default())
        .unwrap();
    let c = out.into_component_buffers();
    let rgb = |i: usize| [c[0][i], c[1][i], c[2][i]];
    assert_eq!(rgb(0), [100.0, 100.0, 100.0]); // background
    assert_eq!(rgb(1), [50.0, 152.0, 50.0]); // label1: ½·100+½·[0,205,0]
    assert_eq!(rgb(2), [200.0, 200.0, 200.0]); // background
    assert_eq!(rgb(3), [100.0, 100.0, 227.0]); // label2: ½·200+½·[0,0,255], 227.5→227
}

/// Opacity 1.0 yields the pure label colour over labelled voxels.
#[test]
fn label_overlay_full_opacity_is_label_color() {
    let gray = ts::burn_compat::make_image::<B, 3>(vec![100.0, 200.0], [1, 1, 2]);
    let lab = ts::burn_compat::make_image::<B, 3>(vec![1.0, 2.0], [1, 1, 2]);
    let out = LabelOverlayFilter::new(1.0, 0)
        .apply(&gray, &lab, &B::default())
        .unwrap();
    let c = out.into_component_buffers();
    assert_eq!([c[0][0], c[1][0], c[2][0]], [0.0, 205.0, 0.0]);
    assert_eq!([c[0][1], c[1][1], c[2][1]], [0.0, 0.0, 255.0]);
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
    let gi = ts::burn_compat::make_image::<B, 3>(gray, [1, 8, 8]);
    let li = ts::burn_compat::make_image::<B, 3>(lab, [1, 8, 8]);
    let out = LabelMapContourOverlayFilter::new(0.5, 0)
        .apply(&gi, &li, &B::default())
        .unwrap();
    let c = out.into_component_buffers();
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
    assert_eq!(c[0], exp_r, "R channel");
    assert_eq!(c[1], exp_g, "G channel");
    assert_eq!(c[2], exp_b, "B channel");
}
