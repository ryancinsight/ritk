use super::*;

#[test]
fn direct_and_fft_volume_paths_propagate_non_finite_fixed_data() {
    let dims = [1, 1, 3];
    let samples = [1.0_f32, f32::NAN, 3.0];
    let config = BlockMatchingConfig {
        block_radius: [0, 0, 1],
        search_radius: [0, 0, 1],
    };
    let grid = BlockGrid::dense([1, 1, 3]);
    let expected = BlockMatchingError::NonFiniteFixedBlock { centre: [0, 0, 1] };

    let direct = track_volume(
        &samples,
        &samples,
        dims,
        config,
        grid,
        SubpixelRefinement::None,
    )
    .expect_err("non-finite fixed data is not a skipped measurement");
    assert_eq!(direct.downcast_ref::<BlockMatchingError>(), Some(&expected));

    let fft = track_volume_fft(
        &samples,
        &samples,
        dims,
        config,
        grid,
        SubpixelRefinement::None,
    )
    .expect_err("the FFT path preserves the non-finite input failure");
    assert_eq!(fft.downcast_ref::<BlockMatchingError>(), Some(&expected));
}

#[test]
fn featureless_fixed_blocks_keep_the_zero_displacement_field() {
    let dims = [1, 1, 3];
    let samples = [4.0_f32; 3];
    let config = BlockMatchingConfig {
        block_radius: [0, 0, 1],
        search_radius: [0, 0, 1],
    };
    let grid = BlockGrid::dense([1, 1, 3]);

    for field in [
        track_volume(
            &samples,
            &samples,
            dims,
            config,
            grid,
            SubpixelRefinement::None,
        )
        .expect("featureless direct block remains unmeasured"),
        track_volume_fft(
            &samples,
            &samples,
            dims,
            config,
            grid,
            SubpixelRefinement::None,
        )
        .expect("featureless FFT block remains unmeasured"),
    ] {
        assert_eq!(field.centres, vec![[0, 0, 1]]);
        assert_eq!(field.displacements, vec![[0.0; 3]]);
        assert_eq!(field.peak_similarities.len(), 1);
        assert!(field.peak_similarities[0].is_nan());
    }
}
