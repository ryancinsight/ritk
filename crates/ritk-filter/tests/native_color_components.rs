//! Tests for the per-component color filtering adaptor.

use coeus_core::SequentialBackend;
use ritk_filter::{map_color_components, MedianFilter};
use ritk_image::{native::Image, ColorVolume};

type B = SequentialBackend;

fn rgb(interleaved: Vec<f32>, spatial: [usize; 3]) -> ColorVolume<f32, B, 3> {
    let [d, r, c] = spatial;
    let backend = B::default();
    let t = coeus_tensor::Tensor::<f32, B>::from_slice_on([d, r, c, 3], &interleaved, &backend);
    ColorVolume::try_new(
        t,
        ritk_spatial::Point::origin(),
        ritk_spatial::Spacing::uniform(1.0),
        ritk_spatial::Direction::identity(),
    )
    .unwrap()
}

#[test]
fn identity_closure_preserves_volume() {
    let interleaved: Vec<f32> = (0..2 * 3 * 4 * 3).map(|i| i as f32).collect();
    let vol = rgb(interleaved.clone(), [2, 3, 4]);
    let out =
        map_color_components(&vol, |img: &Image<f32, B, 3>| img.clone(), &B::default()).unwrap();
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

    let out = map_color_components(
        &vol,
        |img: &Image<f32, B, 3>| MedianFilter::new(1).apply_native(img).unwrap(),
        &B::default(),
    )
    .unwrap();
    let out_comps = out.into_component_buffers();

    // Reference: apply the scalar median to each channel independently.
    for (k, ch) in vol.into_component_buffers().into_iter().enumerate() {
        let backend = B::default();
        let t = coeus_tensor::Tensor::<f32, B>::from_slice_on(spatial, &ch, &backend);
        let img = Image::<f32, B, 3>::new(
            t,
            ritk_spatial::Point::origin(),
            ritk_spatial::Spacing::uniform(1.0),
            ritk_spatial::Direction::identity(),
        )
        .unwrap();
        let ref_buf = MedianFilter::new(1)
            .apply_native(&img)
            .unwrap()
            .data()
            .as_slice()
            .to_vec();
        assert_eq!(out_comps[k], ref_buf, "channel {k} median mismatch");
    }
}
