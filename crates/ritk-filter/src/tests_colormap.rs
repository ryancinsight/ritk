use super::*;
use ritk_image::test_support as ts;

type B = burn_ndarray::NdArray<f32>;

/// Grey colormap on a `[10,20,30,40,50]` ramp → `[0,63,127,191,255]` per channel
/// (normalize by image min/max, ×255, floor — `0.25·255 = 63.75 → 63`).
#[test]
fn grey_ramp_matches_itk_truncation() {
    let img = ts::make_image::<B, 3>(vec![10.0, 20.0, 30.0, 40.0, 50.0], [1, 1, 5]);
    let out = ScalarToRGBColormapFilter::new(Colormap::Grey)
        .apply(&img)
        .unwrap();
    let comps = out.into_component_buffers();
    let expected = [0.0f32, 63.0, 127.0, 191.0, 255.0];
    for c in 0..3 {
        assert_eq!(comps[c], expected, "channel {c}");
    }
}

/// Red colormap puts the ramp in channel 0 only.
#[test]
fn red_colormap_channel_selection() {
    let img = ts::make_image::<B, 3>(vec![10.0, 20.0, 30.0, 40.0, 50.0], [1, 1, 5]);
    let out = ScalarToRGBColormapFilter::new(Colormap::Red)
        .apply(&img)
        .unwrap();
    let comps = out.into_component_buffers();
    assert_eq!(comps[0], vec![0.0, 63.0, 127.0, 191.0, 255.0]);
    assert_eq!(comps[1], vec![0.0; 5]);
    assert_eq!(comps[2], vec![0.0; 5]);
}

/// A constant image maps to all-zero (`range = 0 → t = 0`).
#[test]
fn constant_image_maps_to_zero() {
    let img = ts::make_image::<B, 3>(vec![7.0; 8], [2, 2, 2]);
    let out = ScalarToRGBColormapFilter::new(Colormap::Grey)
        .apply(&img)
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
    let img = ts::make_image::<B, 3>(
        vec![0.0, 1.0, 2.0, 5.0, 7.0, 30.0, 31.0],
        [1, 1, 7],
    );
    let out = LabelToRGBFilter::new(0).apply(&img).unwrap();
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
