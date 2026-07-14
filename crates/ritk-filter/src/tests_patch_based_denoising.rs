use super::*;
use burn_ndarray::NdArray;
use ritk_image::test_support as ts;
use ritk_tensor_ops::extract_vec;

type B = NdArray<f32>;

/// The ITK MersenneTwister port must reproduce the canonical MT19937 sequence:
/// seed 0 → first output 2357136044; seed 5489 → 3499211612.
#[test]
fn test_itk_mt19937_canonical() {
    let mut mt0 = ItkMt::new(0);
    assert_eq!(mt0.next_u32(), 2_357_136_044, "seed 0 first output");
    let mut mt5489 = ItkMt::new(5489);
    assert_eq!(mt5489.next_u32(), 3_499_211_612, "seed 5489 first output");
}

#[test]
fn test_itk_patch_reduction_order() {
    assert_eq!(
        itk_reduction_indices(9).collect::<Vec<_>>(),
        [0, 5, 1, 6, 2, 7, 3, 8, 4]
    );
}

#[test]
fn test_itk_pixel_difference_rounds_before_widening() {
    let current = 26_765.939_453_125_f32;
    let selected = 123.456_001_281_738_28_f32;
    let actual = pixel_difference(current, selected);

    assert_eq!(actual, -26_642.484_375_f64);
    assert_ne!(actual, f64::from(selected) - f64::from(current));
}

/// Determinism: the same input yields the same output (seeded RNG, seed 0).
#[test]
fn test_patch_based_denoising_deterministic() {
    let (ny, nx) = (12usize, 12);
    let vals: Vec<f32> = (0..ny * nx).map(|i| ((i * 37) % 90) as f32 + 5.0).collect();
    let img = ts::make_image::<B, 3>(vals, [1, ny, nx]);
    let filt = PatchBasedDenoisingImageFilter {
        patch_radius: 1,
        number_of_iterations: 1,
        ..Default::default()
    };
    let a = extract_vec(&filt.apply(&img).unwrap()).unwrap().0;
    let b = extract_vec(&filt.apply(&img).unwrap()).unwrap().0;
    assert_eq!(a, b, "seeded denoising must be deterministic");
    assert!(a.iter().all(|v| v.is_finite()));
}

/// Batching is an execution strategy only: changing its memory partition must
/// preserve the seeded sample stream and every pixel's reduction order exactly.
#[test]
fn test_patch_based_denoising_batch_partition_invariant() {
    let (ny, nx) = (11usize, 13);
    let data: Vec<f32> = (0..ny * nx).map(|i| ((i * 37) % 90) as f32 + 5.0).collect();
    let filter = PatchBasedDenoisingImageFilter {
        patch_radius: 1,
        number_of_iterations: 1,
        number_of_sample_patches: 32,
        ..Default::default()
    };
    let sample_bytes = size_of::<usize>() * filter.number_of_sample_patches;
    let reference = filter.pass_with_sample_budget(&data, [1, ny, nx], sample_bytes);

    for pixel_capacity in [2, 7, 17, ny * nx - 1, ny * nx, ny * nx + 1] {
        let partitioned =
            filter.pass_with_sample_budget(&data, [1, ny, nx], sample_bytes * pixel_capacity);
        assert_eq!(partitioned, reference, "pixel capacity {pixel_capacity}");
    }
}

#[test]
fn test_sampling_interval_intersects_patch_and_sampler_regions() {
    let size = 64;
    let patch_radius = 2;
    let sample_radius = 50;
    assert_eq!(
        sampling_interval(0, size, patch_radius, sample_radius),
        (0, 50)
    );
    assert_eq!(
        sampling_interval(32, size, patch_radius, sample_radius),
        (2, 61)
    );
    assert_eq!(
        sampling_interval(63, size, patch_radius, sample_radius),
        (13, 63)
    );
    assert_eq!(sampling_interval(0, size, patch_radius, i64::MAX), (0, 61));
}

/// ITK disables the selected patch's boundary condition because the sampling
/// region guarantees that it is at least as in-bounds as the current patch.
/// Exhaust the small integer domain to pin that indexing precondition.
#[test]
fn test_sampling_interval_preserves_patch_offset_bounds() {
    for size in 3..=20 {
        for patch_radius in 1..=(size - 1) / 2 {
            for position in 0..size {
                for sample_radius in (0..=size).chain(std::iter::once(i64::MAX)) {
                    let (lo, hi) = sampling_interval(position, size, patch_radius, sample_radius);
                    for selected in lo..=hi {
                        for offset in -patch_radius..=patch_radius {
                            let current_offset = position + offset;
                            if (0..size).contains(&current_offset) {
                                let selected_offset = selected + offset;
                                assert!(
                                    (0..size).contains(&selected_offset),
                                    "size={size}, radius={patch_radius}, position={position}, \
                                     sample_radius={sample_radius}, selected={selected}, \
                                     offset={offset}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn test_patch_displacements_match_coordinate_indices() {
    for [nz, ny, nx] in [[1, 5, 6], [5, 6, 7]] {
        let ndim = if nz == 1 { 2 } else { 3 };
        let sizes = [nx, ny, nz];
        let radius = 1;
        let z_radius = if ndim == 3 { radius } else { 0 };
        let row_stride = isize::try_from(nx).expect("test shape fits isize");
        let plane_stride = isize::try_from(ny * nx).expect("test shape fits isize");
        let idx = |x: i64, y: i64, z: i64| -> usize {
            usize::try_from(z).expect("test coordinate is nonnegative")
                * usize::try_from(ny * nx).expect("test shape fits usize")
                + usize::try_from(y).expect("test coordinate is nonnegative")
                    * usize::try_from(nx).expect("test shape fits usize")
                + usize::try_from(x).expect("test coordinate is nonnegative")
        };

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let (x_lo, x_hi) = sampling_interval(x, nx, radius, i64::MAX);
                    let (y_lo, y_hi) = sampling_interval(y, ny, radius, i64::MAX);
                    let (z_lo, z_hi) = if ndim == 3 {
                        sampling_interval(z, nz, radius, i64::MAX)
                    } else {
                        (0, 0)
                    };
                    for qz in z_lo..=z_hi {
                        for qy in y_lo..=y_hi {
                            for qx in x_lo..=x_hi {
                                let q_index = idx(qx, qy, qz);
                                for dz in -z_radius..=z_radius {
                                    for dy in -radius..=radius {
                                        for dx in -radius..=radius {
                                            let p_offset = [x + dx, y + dy, z + dz];
                                            if !p_offset.iter().zip(sizes).all(
                                                |(&coordinate, size)| {
                                                    (0..size).contains(&coordinate)
                                                },
                                            ) {
                                                continue;
                                            }
                                            let q_offset = [qx + dx, qy + dy, qz + dz];
                                            assert!(q_offset.iter().zip(sizes).all(
                                                |(&coordinate, size)| {
                                                    (0..size).contains(&coordinate)
                                                }
                                            ));
                                            let displacement = isize::try_from(dz)
                                                .expect("test offset fits isize")
                                                * plane_stride
                                                + isize::try_from(dy)
                                                    .expect("test offset fits isize")
                                                    * row_stride
                                                + isize::try_from(dx)
                                                    .expect("test offset fits isize");
                                            assert_eq!(
                                                q_index.wrapping_add_signed(displacement),
                                                idx(q_offset[0], q_offset[1], q_offset[2])
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn test_patch_based_denoising_rejects_unbounded_sample_storage() {
    let image = ts::make_image::<B, 3>(vec![1.0f32; 9], [1, 3, 3]);
    let max_samples = SAMPLE_BATCH_BYTES / size_of::<usize>();
    let error = PatchBasedDenoisingImageFilter {
        number_of_sample_patches: max_samples + 1,
        ..Default::default()
    }
    .apply(&image)
    .unwrap_err();

    assert_eq!(
        error.to_string(),
        format!(
            "number_of_sample_patches {} exceeds bounded capacity {max_samples}",
            max_samples + 1
        )
    );
}

#[test]
fn test_patch_based_denoising_rejects_invalid_configuration() {
    let image = ts::make_image::<B, 3>(vec![1.0f32; 81], [1, 9, 9]);
    let cases = [
        (
            PatchBasedDenoisingImageFilter {
                number_of_iterations: 0,
                ..Default::default()
            },
            "number_of_iterations must be positive",
        ),
        (
            PatchBasedDenoisingImageFilter {
                number_of_sample_patches: 0,
                ..Default::default()
            },
            "number_of_sample_patches must be positive",
        ),
        (
            PatchBasedDenoisingImageFilter {
                sample_variance: f64::NAN,
                ..Default::default()
            },
            "sample_variance must be finite and nonnegative, got NaN",
        ),
        (
            PatchBasedDenoisingImageFilter {
                sample_variance: -1.0,
                ..Default::default()
            },
            "sample_variance must be finite and nonnegative, got -1",
        ),
        (
            PatchBasedDenoisingImageFilter {
                kernel_sigma: 0.0,
                ..Default::default()
            },
            "kernel_sigma must be finite and positive, got 0",
        ),
        (
            PatchBasedDenoisingImageFilter {
                kernel_sigma: f64::NAN,
                ..Default::default()
            },
            "kernel_sigma must be finite and positive, got NaN",
        ),
    ];

    for (filter, expected) in cases {
        assert_eq!(filter.apply(&image).unwrap_err().to_string(), expected);
    }

    let overflow = PatchBasedDenoisingImageFilter {
        patch_radius: usize::MAX,
        ..Default::default()
    };
    assert_eq!(
        overflow.apply(&image).unwrap_err().to_string(),
        format!("patch_radius {} overflows", usize::MAX)
    );
}

#[test]
fn test_patch_based_denoising_rejects_image_smaller_than_patch() {
    let planar = ts::make_image::<B, 3>(vec![1.0f32; 9], [1, 3, 3]);
    let volumetric = ts::make_image::<B, 3>(vec![1.0f32; 5 * 9 * 9], [5, 9, 9]);
    let filter = PatchBasedDenoisingImageFilter::default();

    assert_eq!(
        filter.apply(&planar).unwrap_err().to_string(),
        "patch diameter 9 exceeds active image dimensions [3, 3]"
    );
    assert_eq!(
        filter.apply(&volumetric).unwrap_err().to_string(),
        "patch diameter 9 exceeds active image dimensions [5, 9, 9]"
    );
}

/// A constant image is a fixed point (every patch distance is 0 → all weights 1
/// → the gradient of (c − c) is exactly 0).
#[test]
fn test_patch_based_denoising_constant_is_fixed_point() {
    let (ny, nx) = (10usize, 10);
    let img = ts::make_image::<B, 3>(vec![42.0f32; ny * nx], [1, ny, nx]);
    let out = PatchBasedDenoisingImageFilter {
        patch_radius: 1,
        ..Default::default()
    }
    .apply(&img)
    .unwrap();
    let r = extract_vec(&out).unwrap().0;
    assert!(
        r.iter().all(|&v| (v - 42.0).abs() < 1e-4),
        "constant image must be preserved"
    );
}

/// Smooth-disc weights: centre weight is 1 (squared), and the edge weight
/// exceeds the corner weight (disc decays with distance).
#[test]
fn test_smooth_disc_weights() {
    let w = smooth_disc_weights_sq(1, 2); // 3x3
    assert_eq!(w.len(), 9);
    assert!((w[4] - 1.0).abs() < 1e-12, "centre weight² = 1");
    assert!(w[1] > w[0], "edge weight > corner weight");
}
