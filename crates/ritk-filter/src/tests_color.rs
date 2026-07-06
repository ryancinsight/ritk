//! Tests for the per-component color filtering adaptor.

use super::map_color_components;
use crate::MedianFilter;
use ritk_image::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use ritk_image::{ColorVolume, Image};
use ritk_spatial::{Direction, Point, Spacing};

type B = NdArray<f32>;

fn rgb(interleaved: Vec<f32>, spatial: [usize; 3]) -> ColorVolume<B, 3> {
    let [d, r, c] = spatial;
    let dev = Default::default();
    let t = Tensor::<B, 4>::from_data(TensorData::new(interleaved, Shape::new([d, r, c, 3])), &dev);
    ColorVolume::try_new(
        t,
        Point::origin(),
        Spacing::uniform(1.0),
        Direction::identity(),
    )
    .unwrap()
}

#[test]
fn identity_closure_preserves_volume() {
    let interleaved: Vec<f32> = (0..2 * 3 * 4 * 3).map(|i| i as f32).collect();
    let vol = rgb(interleaved.clone(), [2, 3, 4]);
    let out = map_color_components(&vol, |img: &Image<B, 3>| img.clone()).unwrap();
    assert_eq!(out.data_vec(), interleaved);
}

#[test]
fn per_component_median_matches_independent_channel_median() {
    // Build an RGB volume where each channel is a distinct ramp + an outlier, so
    // a median visibly changes each channel; the adaptor must filter each
    // channel exactly as the scalar MedianFilter would on that channel alone.
    let spatial = [1usize, 5, 5];
    let n = spatial.iter().product::<usize>();
    let mut interleaved = vec![0.0_f32; n * 3];
    for i in 0..n {
        interleaved[i * 3] = (i % 7) as f32; // R
        interleaved[i * 3 + 1] = (i % 3) as f32 * 10.0; // G
        interleaved[i * 3 + 2] = (i % 5) as f32 * 100.0; // B
    }
    let vol = rgb(interleaved, spatial);

    let out = map_color_components(&vol, |img: &Image<B, 3>| {
        MedianFilter::new(1).apply(img).unwrap()
    })
    .unwrap();
    let out_comps = out.into_component_buffers();

    // Reference: apply the scalar median to each channel independently.
    for (k, ch) in vol.into_component_buffers().into_iter().enumerate() {
        let dev = Default::default();
        let t = Tensor::<B, 3>::from_data(TensorData::new(ch, Shape::new(spatial)), &dev);
        let img = Image::<B, 3>::new(
            t,
            Point::origin(),
            Spacing::uniform(1.0),
            Direction::identity(),
        );
        let ref_buf = MedianFilter::new(1)
            .apply(&img)
            .unwrap()
            .data()
            .clone()
            .into_data()
            .into_vec::<f32>()
            .unwrap();
        assert_eq!(out_comps[k], ref_buf, "channel {k} median mismatch");
    }
}
