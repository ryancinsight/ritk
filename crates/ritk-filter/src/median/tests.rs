#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]
use super::*;

use coeus_core::SequentialBackend;
use ritk_core::image::Image;
use ritk_image::tensor::Tensor;
use ritk_image::Image as NativeImage;
use ritk_spatial::{Direction, Point, Spacing};

type B = coeus_core::SequentialBackend;

/// Construct a test image from flat values, shape, and optional metadata.
fn make_image(
    vals: Vec<f32>,
    dims: [usize; 3],
    origin: [f64; 3],
    spacing: [f64; 3],
) -> Image<f32, B, 3> {
    let tensor = Tensor::<f32, B>::from_slice(dims, &vals);
    Image::new(
        tensor,
        Point::new(origin),
        Spacing::new(spacing),
        Direction::identity(),
    )
    .expect("invariant: fixture tensor has the declared rank")
}

/// Extract voxel data as `Vec<f32>` from an image.
fn extract_vals(img: &Image<f32, B, 3>) -> Vec<f32> {
    img.data_slice()
        .expect("invariant: contiguous host storage")
        .to_vec()
}

// ── Test 1: Uniform image is unchanged ────────────────────────────────

/// A constant image must be invariant under median filtering for any
/// radius, because the median of identical values equals that value.
///
/// **Proof sketch**: Let I(p) = c for all p. For any neighbourhood N(p),
/// every element of the sorted list is c, so median = c. ∎
#[test]
fn test_uniform_image_unchanged() {
    let dims = [8, 8, 8];
    let val = 42.0_f32;
    let vals = vec![val; dims[0] * dims[1] * dims[2]];
    let img = make_image(vals, dims, [0.0; 3], [1.0; 3]);

    let filter = MedianFilter::new(2);
    let out = filter.apply(&img).unwrap();
    let result = extract_vals(&out);

    assert_eq!(out.shape(), dims);
    for (i, &v) in result.iter().enumerate() {
        assert!((v - val).abs() < 1e-6, "voxel {i}: expected {val}, got {v}");
    }
}

#[test]
fn native_median_removes_an_impulse_and_preserves_metadata() {
    let backend = SequentialBackend;
    let source = NativeImage::from_flat_on(
        vec![0.0, 10.0, 0.0],
        [1, 1, 3],
        Point::new([2.0, 3.0, 4.0]),
        Spacing::new([0.5, 1.0, 2.0]),
        Direction::identity(),
        &backend,
    )
    .unwrap();
    let output = MedianFilter::new(1).apply_native(&source).unwrap();

    assert_eq!(output.data_slice().unwrap(), &[0.0, 0.0, 0.0]);
    assert_eq!(output.shape(), source.shape());
    assert_eq!(output.origin(), source.origin());
    assert_eq!(output.spacing(), source.spacing());
    assert_eq!(output.direction(), source.direction());
}

// ── Test 2: Impulse noise removed ─────────────────────────────────────

/// A single spike voxel (salt-noise) embedded in a constant field must be
/// eliminated by median filtering with radius ≥ 1, because the spike
/// constitutes at most 1 out of (2r+1)³ ≥ 27 samples and therefore
/// cannot be the median of the sorted neighbourhood.
///
/// **Proof**: In a 3×3×3 neighbourhood around the spike, 26 values equal
/// the background `c` and 1 equals `c + spike`. Sorted, the 14th element
/// (index 13) is `c`. ∎
#[test]
fn test_impulse_noise_removed() {
    let dims = [8, 8, 8];
    let bg = 10.0_f32;
    let spike = 1000.0_f32;
    let n = dims[0] * dims[1] * dims[2];
    let mut vals = vec![bg; n];

    // Place spike at centre voxel (4, 4, 4).
    let spike_idx = 4 * dims[1] * dims[2] + 4 * dims[2] + 4;
    vals[spike_idx] = spike;

    let img = make_image(vals, dims, [0.0; 3], [1.0; 3]);
    let filter = MedianFilter::new(1);
    let out = filter.apply(&img).unwrap();
    let result = extract_vals(&out);

    // The spike location must now hold the background value.
    assert!(
        (result[spike_idx] - bg).abs() < 1e-6,
        "spike not removed: expected {bg}, got {}",
        result[spike_idx]
    );

    // All other voxels in the interior (away from boundaries) must remain
    // at the background value since their entire neighbourhood is constant.
    for iz in 1..dims[0] - 1 {
        for iy in 1..dims[1] - 1 {
            for ix in 1..dims[2] - 1 {
                let idx = iz * dims[1] * dims[2] + iy * dims[2] + ix;
                if idx == spike_idx {
                    continue;
                }
                // Voxels adjacent to the spike see at most 1 non-bg value
                // out of 27, so their median is still bg.
                assert!(
                    (result[idx] - bg).abs() < 1e-6,
                    "interior voxel ({iz},{iy},{ix}): expected {bg}, got {}",
                    result[idx]
                );
            }
        }
    }
}

// ── Test 3: Metadata preserved ────────────────────────────────────────

/// Origin, spacing, and direction of the output image must be identical
/// to the input. The median filter operates exclusively on voxel
/// intensities and must not mutate spatial metadata.
#[test]
fn test_metadata_preserved() {
    let dims = [4, 4, 4];
    let origin = [10.0, -5.5, 2.71]; // arbitrary float coordinates
    let spacing = [0.5, 0.75, 1.25];
    let vals = vec![7.0_f32; dims[0] * dims[1] * dims[2]];
    let img = make_image(vals, dims, origin, spacing);

    let filter = MedianFilter::new(1);
    let out = filter.apply(&img).unwrap();

    // Shape.
    assert_eq!(out.shape(), dims);

    // Origin (exact equality; no computation on these values).
    let out_origin = out.origin();
    for d in 0..3 {
        assert!(
            (out_origin[d] - origin[d]).abs() < 1e-12,
            "origin[{d}]: expected {}, got {}",
            origin[d],
            out_origin[d]
        );
    }

    // Spacing.
    let out_spacing = out.spacing();
    for d in 0..3 {
        assert!(
            (out_spacing[d] - spacing[d]).abs() < 1e-12,
            "spacing[{d}]: expected {}, got {}",
            spacing[d],
            out_spacing[d]
        );
    }

    // Direction (identity → identity).
    let out_dir = out.direction();
    let in_dir = img.direction();
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (out_dir[(i, j)] - in_dir[(i, j)]).abs() < 1e-12,
                "direction[{i},{j}] mismatch"
            );
        }
    }
}

// -- Test 4: Identity for radius zero ---------------------------------

/// With radius = 0 the neighbourhood is a single voxel, so the output
/// must be bit-identical to the input.
///
/// **Proof**: |N(p)| = 1³ = 1, so median of {I(p)} = I(p). ∎
#[test]
fn test_identity_for_radius_zero() {
    let dims = [6, 6, 6];
    let n = dims[0] * dims[1] * dims[2];
    // Deterministic non-trivial pattern: voxel value = flat index.
    let vals: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let img = make_image(vals.clone(), dims, [0.0; 3], [1.0; 3]);

    let filter = MedianFilter::new(0);
    let out = filter.apply(&img).unwrap();
    let result = extract_vals(&out);

    assert_eq!(result.len(), vals.len());
    for (i, (&expected, &actual)) in vals.iter().zip(result.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-6,
            "voxel {i}: expected {expected}, got {actual}"
        );
    }
}

// -- Test 5: Brute-force reference agreement -------------------------

/// `median_3d` must produce the lower median over the multiset
/// `{data[clamp(iz + dz), clamp(iy + dy), clamp(ix + dx)] : dz,
/// dy, dx ∈ [-r, r]}` for every voxel. The brute-force reference below
/// gathers the full multiset via clamp arithmetic and applies the
/// same `select_nth_unstable_by(mid)` call; agreement is therefore
/// bit-equal — `assert_eq!` is the correct (not merely bounded)
/// oracle. The clamp arithmetic matches the production code: same
/// `.clamp(0, n - 1)` idiom means identical multisets.
///
/// This test guards against regressions introduced by the
/// PERF-377-01 clamp-hoisting micro-optimisation (clamp indices are
/// pre-baked into `zz_buf` / `yy_buf` instead of computed per inner
/// tick). The hoisting does NOT change the values placed into
/// `neighbors`; it merely re-orders computation.
fn median_3d_brute_force(data: &[f32], dims: [usize; 3], radius: usize) -> Vec<f32> {
    let (nz, ny, nx) = (dims[0], dims[1], dims[2]);
    let r = radius as isize;
    let mut output = vec![0.0_f32; nz * ny * nx];

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let mut nbrs: Vec<f32> = Vec::with_capacity((2 * radius + 1).pow(3));
                for dz in -r..=r {
                    let zz = (iz as isize + dz).clamp(0, nz as isize - 1) as usize;
                    for dy in -r..=r {
                        let yy = (iy as isize + dy).clamp(0, ny as isize - 1) as usize;
                        for dx in -r..=r {
                            let xx = (ix as isize + dx).clamp(0, nx as isize - 1) as usize;
                            nbrs.push(data[zz * ny * nx + yy * nx + xx]);
                        }
                    }
                }
                let mid = nbrs.len() / 2;
                nbrs.select_nth_unstable_by(mid, |a, b| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                });
                output[iz * ny * nx + iy * nx + ix] = nbrs[mid];
            }
        }
    }
    output
}

#[test]
fn test_median_3d_matches_brute_force_reference_r1() {
    // 12×12×12 = 1728 voxels; small but non-trivial boundary handling
    // (r=1 windows always touch the clamp boundary on a 12-D axis).
    let dims = [12, 12, 12];
    let n: usize = dims.iter().product();
    let vals: Vec<f32> = (0..n).map(|i| ((i * 31) % 97) as f32 * 0.13).collect();

    let r1 = median_3d_brute_force(&vals, dims, 1);
    let r2 = super::median_3d(&vals, dims, 1);
    assert_eq!(r1.len(), dims.iter().product::<usize>());
    assert_eq!(r2.len(), dims.iter().product::<usize>());
    for (i, (&a, &b)) in r1.iter().zip(r2.iter()).enumerate() {
        // `select_nth_unstable_by` is deterministic for a given input
        // and partition scheme, so values MUST match exactly.
        assert!(
            a.to_bits() == b.to_bits(),
            "voxel {i} mismatch: brute={a} (bits={:08x}) hoisted={b} (bits={:08x})",
            a.to_bits(),
            b.to_bits()
        );
    }
}

#[test]
fn test_median_3d_matches_brute_force_reference_r3() {
    // r=3 exercises the larger (2r+1) = 7 cube (343 samples) and
    // surfaces any off-by-one in the clamp hoist.
    let dims = [10, 10, 10];
    let n: usize = dims.iter().product();
    let vals: Vec<f32> = (0..n).map(|i| ((i * 13 + 7) % 53) as f32 - 26.0).collect();

    let r1 = median_3d_brute_force(&vals, dims, 3);
    let r2 = super::median_3d(&vals, dims, 3);
    assert_eq!(r1.len(), n);
    assert_eq!(r2.len(), n);
    for (i, (&a, &b)) in r1.iter().zip(r2.iter()).enumerate() {
        assert!(
            a.to_bits() == b.to_bits(),
            "voxel {i} mismatch (r=3): brute={a} hoisted={b}"
        );
    }
}
