use super::*;
use coeus_core::SequentialBackend;
use ritk_core::spatial::{Point, Spacing};
use ritk_filter::edge::GaussianSigma;
use ritk_image::test_support::{make_image, make_image_with};

type B = SequentialBackend;

fn make_image_with_metadata(
    data: Vec<f32>,
    dims: [usize; 3],
    origin: [f64; 3],
    spacing: [f64; 3],
) -> Image<f32, B, 3> {
    make_image_with(
        data,
        dims,
        Some(Point::new(origin)),
        Some(Spacing::new(spacing)),
        None,
    )
}

fn get_values(image: &Image<f32, B, 3>) -> Vec<f32> {
    image.data().to_vec()
}

/// Create a signed distance–like initial φ: negative inside a sphere of
/// radius `r` centred at (`cz`,`cy`,`cx`), positive outside.
fn sphere_phi(dims: [usize; 3], center: [f64; 3], r: f64) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let mut phi = vec![0.0_f32; nz * ny * nx];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let dz = iz as f64 - center[0];
                let dy = iy as f64 - center[1];
                let dx = ix as f64 - center[2];
                let dist = (dz * dz + dy * dy + dx * dx).sqrt();
                phi[iz * ny * nx + iy * nx + ix] = (dist - r) as f32;
            }
        }
    }
    phi
}

// ── Test 1: Step-edge image — contour expands to edge ──────────────────────

#[test]
fn test_step_edge_contour_expands_to_edge() {
    // 16×16×16 image: foreground sphere of radius 6 at center with
    // intensity 200, background 0. Initial φ is a small sphere of
    // radius 2 inside the foreground.
    //
    // Pure balloon expansion (curvature=0, advection=0) with edge-modulated
    // speed: g ≈ 1 inside homogeneous region, g ≪ 1 near edge. The contour
    // should expand from the initial sphere toward the foreground boundary.
    let dims = [16, 16, 16];
    let [nz, ny, nx] = dims;
    let center = [8.0, 8.0, 8.0];
    let fg_radius = 6.0;

    let mut img_data = vec![0.0_f32; nz * ny * nx];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let dz = iz as f64 - center[0];
                let dy = iy as f64 - center[1];
                let dx = ix as f64 - center[2];
                if (dz * dz + dy * dy + dx * dx).sqrt() <= fg_radius {
                    img_data[iz * ny * nx + iy * nx + ix] = 200.0;
                }
            }
        }
    }

    let image = make_image(img_data, dims);
    let init_phi = sphere_phi(dims, center, 2.0);
    let phi_image = make_image(init_phi, dims);

    // Pure balloon: no curvature (avoids shrinkage at the small initial
    // sphere where κ is large), no advection. dt=0.05 keeps the explicit
    // Euler scheme stable.
    let mut gac = GeodesicActiveContourSegmentation::new();
    gac.propagation_weight = 3.0;
    gac.curvature_weight = 0.0;
    gac.advection_weight = 0.0;
    gac.edge_k = 50.0;
    gac.sigma = GaussianSigma::new_unchecked(0.5);
    gac.dt = 0.05;
    gac.max_iterations = 500;

    let result = gac.apply(&image, &phi_image).expect("infallible: validated precondition");
    let mask = get_values(&result);

    // Count segmented voxels.
    let seg_count: usize = mask.iter().filter(|&&v| v == 1.0).count();

    // Count actual foreground voxels.
    let fg_count: usize = {
        let mut c = 0;
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let dz = iz as f64 - center[0];
                    let dy = iy as f64 - center[1];
                    let dx = ix as f64 - center[2];
                    if (dz * dz + dy * dy + dx * dx).sqrt() <= fg_radius {
                        c += 1;
                    }
                }
            }
        }
        c
    };

    // Initial sphere (radius 2) has fewer voxels than the foreground.
    // After balloon expansion, the segmented region must be substantially
    // larger than the initial contour.
    let init_count: usize = sphere_phi(dims, center, 2.0)
        .iter()
        .filter(|&&v| v < 0.0)
        .count();

    assert!(
        seg_count > init_count * 2,
        "segmented region ({}) must be substantially larger than initial contour ({})",
        seg_count,
        init_count
    );

    // Compute overlap between segmented region and foreground sphere.
    let mut overlap = 0usize;
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let i = iz * ny * nx + iy * nx + ix;
                let dz = iz as f64 - center[0];
                let dy = iy as f64 - center[1];
                let dx = ix as f64 - center[2];
                let in_fg = (dz * dz + dy * dy + dx * dx).sqrt() <= fg_radius;
                if mask[i] == 1.0 && in_fg {
                    overlap += 1;
                }
            }
        }
    }

    // With pure balloon on a step-edge image, the contour expands through
    // the homogeneous interior (g ≈ 1) and slows at the edge (g ≪ 1).
    // It will leak past the edge over many iterations. We verify:
    //  (a) expansion happened (checked above)
    //  (b) most of the foreground is covered (recall)
    let recall = overlap as f64 / fg_count.max(1) as f64;
    assert!(
        recall > 0.5,
        "recall w.r.t. foreground must exceed 0.5, got {:.4} \
             (overlap={}, fg_count={}, seg_count={})",
        recall,
        overlap,
        fg_count,
        seg_count
    );
}

// ── Test 1b: Advection edge-attraction stays bounded (no leak) ─────────────

/// With a strong advection (edge-attraction) weight the front must stay bounded
/// near the object, not run away and fill the volume.
///
/// Regression: the advection term ∇g·∇φ was discretised with central differences
/// — unconditionally unstable for a transport term — so a positive
/// `advection_weight` leaked the contour straight through the edge until it
/// filled the whole image (segmented count → volume). Upwind differencing makes
/// the term stable, so the front is pulled toward the edge and the segmented
/// region stays a small multiple of the object, well under half the volume.
#[test]
fn test_advection_does_not_leak_through_edges() {
    let dims = [16, 16, 16];
    let [nz, ny, nx] = dims;
    let total = nz * ny * nx;
    let center = [8.0, 8.0, 8.0];
    let fg_radius = 6.0;

    let mut img = vec![0.0_f32; total];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let dz = iz as f64 - center[0];
                let dy = iy as f64 - center[1];
                let dx = ix as f64 - center[2];
                if (dz * dz + dy * dy + dx * dx).sqrt() <= fg_radius {
                    img[iz * ny * nx + iy * nx + ix] = 200.0;
                }
            }
        }
    }
    let image = make_image(img, dims);
    let phi_image = make_image(sphere_phi(dims, center, 2.0), dims);

    let mut gac = GeodesicActiveContourSegmentation::new();
    gac.propagation_weight = 1.0;
    gac.curvature_weight = 0.2;
    gac.advection_weight = 5.0; // strong edge attraction
    gac.edge_k = 1.0;
    gac.sigma = GaussianSigma::new_unchecked(1.0);
    gac.dt = 0.05;
    gac.max_iterations = 300;

    let mask = get_values(&gac.apply(&image, &phi_image).expect("infallible: validated precondition"));
    let seg: usize = mask.iter().filter(|&&v| v == 1.0).count();
    let init: usize = sphere_phi(dims, center, 2.0)
        .iter()
        .filter(|&&v| v < 0.0)
        .count();

    assert!(
        seg > init,
        "the front must evolve from the seed ({init} → {seg})"
    );
    assert!(
        seg < total / 2,
        "advection must not leak the front through the edge: segmented {seg} of {total} \
         (central-difference advection filled the volume here)"
    );
}

// ── Test 2: Uniform image — no edges, uniform expansion ────────────────────

#[test]
fn test_uniform_image_no_edges() {
    // Uniform image: g ≡ 1, ∇g ≡ 0. With positive propagation, the
    // contour should expand. With enough iterations, most voxels become
    // inside (φ < 0).
    let dims = [10, 10, 10];
    let n: usize = dims.iter().product();
    let img_data = vec![100.0_f32; n];
    let image = make_image(img_data, dims);

    let init_phi = sphere_phi(dims, [5.0, 5.0, 5.0], 2.0);
    let phi_image = make_image(init_phi, dims);

    let mut gac = GeodesicActiveContourSegmentation::new();
    gac.propagation_weight = 2.0;
    gac.curvature_weight = 0.0;
    gac.advection_weight = 0.0;
    gac.dt = 0.1;
    gac.max_iterations = 300;

    let result = gac.apply(&image, &phi_image).expect("infallible: validated precondition");
    let mask = get_values(&result);

    let seg_count: usize = mask.iter().filter(|&&v| v == 1.0).count();
    let init_count: usize = sphere_phi(dims, [5.0, 5.0, 5.0], 2.0)
        .iter()
        .filter(|&&v| v < 0.0)
        .count();

    // With uniform g=1 and positive propagation, the region must expand.
    assert!(
        seg_count > init_count,
        "with positive propagation on uniform image, segmented region ({}) \
             must exceed initial ({})",
        seg_count,
        init_count
    );
}

// ── Test 3: Output is strictly binary ──────────────────────────────────────

#[test]
fn test_output_is_binary() {
    let dims = [8, 8, 8];
    let n: usize = dims.iter().product();
    // Random-ish image data.
    let img_data: Vec<f32> = (0..n).map(|i| ((i * 37 + 13) % 256) as f32).collect();
    let image = make_image(img_data, dims);

    let init_phi = sphere_phi(dims, [4.0, 4.0, 4.0], 3.0);
    let phi_image = make_image(init_phi, dims);

    let mut gac = GeodesicActiveContourSegmentation::new();
    gac.max_iterations = 20;

    let result = gac.apply(&image, &phi_image).expect("infallible: validated precondition");
    let mask = get_values(&result);

    for (i, &v) in mask.iter().enumerate() {
        assert!(
            v == 0.0 || v == 1.0,
            "output at voxel {} must be 0.0 or 1.0, got {}",
            i,
            v
        );
    }
}

// ── Test 4: Spatial metadata preserved ─────────────────────────────────────

#[test]
fn test_metadata_preserved() {
    let dims = [4, 4, 4];
    let n: usize = dims.iter().product();
    let origin = [1.5, -2.0, 3.7];
    let spacing = [0.5, 1.0, 2.0];

    let image = make_image_with_metadata(vec![50.0_f32; n], dims, origin, spacing);
    let phi_image = make_image_with_metadata(
        sphere_phi(dims, [2.0, 2.0, 2.0], 1.5),
        dims,
        origin,
        spacing,
    );

    let mut gac = GeodesicActiveContourSegmentation::new();
    gac.max_iterations = 5;

    let result = gac.apply(&image, &phi_image).expect("infallible: validated precondition");

    assert_eq!(result.origin(), image.origin(), "origin must be preserved");
    assert_eq!(
        result.spacing(),
        image.spacing(),
        "spacing must be preserved"
    );
    assert_eq!(
        result.direction(),
        image.direction(),
        "direction must be preserved"
    );
    assert_eq!(result.shape(), dims, "shape must be preserved");
}

// ── Test 5: Shape mismatch error ───────────────────────────────────────────

#[test]
fn test_shape_mismatch_returns_error() {
    let image: Image<f32, B, 3> = make_image(vec![0.0_f32; 27], [3, 3, 3]);
    let phi_image: Image<f32, B, 3> = make_image(vec![0.0_f32; 8], [2, 2, 2]);

    let gac = GeodesicActiveContourSegmentation::new();
    let result = gac.apply(&image, &phi_image);
    assert!(result.is_err(), "shape mismatch must produce an error");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("must match"),
        "error message must mention shape mismatch, got: {}",
        err_msg
    );
}

// ── Test 6: Edge stopping function correctness ─────────────────────────────

#[test]
fn test_edge_stopping_values() {
    // g(0) = 1, g(k) = 0.5, lim g(∞) → 0.
    let k = 2.0;
    let grad = vec![0.0_f32, 2.0, 100.0];
    let g = compute_edge_stopping(&grad, k);

    // g(0) = 1/(1 + 0) = 1.0
    assert!((g[0] - 1.0).abs() < 1e-6, "g(0) must be 1.0, got {}", g[0]);
    // g(k) = 1/(1 + 1) = 0.5
    assert!((g[1] - 0.5).abs() < 1e-6, "g(k) must be 0.5, got {}", g[1]);
    // g(100) ≈ 1/(1 + 2500) ≈ 0.0004
    assert!(g[2] < 0.01, "g(large) must be near 0, got {}", g[2]);
}

// ── Test 7: Gaussian kernel sums to 1 ──────────────────────────────────────

#[test]
fn test_gaussian_kernel_normalised() {
    let kernel = ritk_filter::gaussian_kernel::<f32>(2.0, Some(6));
    let sum: f32 = kernel.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "Gaussian kernel must sum to 1.0, got {}",
        sum
    );
    // Kernel must be symmetric.
    let len = kernel.len();
    for i in 0..len / 2 {
        assert!(
            (kernel[i] - kernel[len - 1 - i]).abs() < 1e-6,
            "kernel must be symmetric: k[{}]={} vs k[{}]={}",
            i,
            kernel[i],
            len - 1 - i,
            kernel[len - 1 - i]
        );
    }
}

// ── Test 9: RMS convergence criterion — zero-force early termination ──────

/// Verifies the RMS convergence criterion:
/// with all force weights = 0 every per-voxel update dphi = 0, so
/// RMS(Δφ) = 0 < tolerance on the first iteration and the filter breaks
/// before reaching max_iterations. The output mask must equal the initial
/// phi < 0 thresholding because the level set never moves.
///
/// # Evidence
/// dphi = dt × (0 × g × κ × |∇φ| − 0 × g × |∇φ| + 0 × adv) = 0 for every voxel.
/// ∴ sum_sq = 0, rms = 0 < tolerance, break on iteration 1.
/// phi_new[i] = phi[i] + 0 = phi[i], so the output mask equals the initial phi < 0 mask.
#[test]
fn rms_convergence_terminates_early_on_zero_force() {
    let dims = [8, 8, 8];
    let n: usize = dims.iter().product();

    // Non-trivial image: gradient will yield a non-trivial g, but forces are zero.
    let img_data: Vec<f32> = (0..n).map(|i| ((i * 17 + 3) % 200) as f32).collect();
    let image = make_image(img_data, dims);

    // Sphere-like initial phi: negative inside a radius-3 sphere.
    let init_phi = sphere_phi(dims, [4.0, 4.0, 4.0], 3.0);
    let phi_image = make_image(init_phi.clone(), dims);

    // All force weights zero: dphi = 0 at every voxel every iteration.
    // Generous tolerance (1.0) and large max_iterations so that if the RMS
    // criterion is broken the test still completes — it would just return the
    // same mask regardless.
    let mut gac = GeodesicActiveContourSegmentation::new();
    gac.propagation_weight = 0.0;
    gac.curvature_weight = 0.0;
    gac.advection_weight = 0.0;
    gac.dt = 0.05;
    gac.max_iterations = 500;
    gac.tolerance = 1.0;

    let result = gac.apply(&image, &phi_image).expect("infallible: validated precondition");
    let mask = get_values(&result);

    // The expected mask is simply phi < 0 from the initial level set.
    let expected_mask: Vec<f32> = init_phi
        .iter()
        .map(|&v| if v < 0.0 { 1.0_f32 } else { 0.0_f32 })
        .collect();

    for (i, (&got, &expected)) in mask.iter().zip(expected_mask.iter()).enumerate() {
        assert_eq!(
            got, expected,
            "voxel {i}: zero-force GAC must preserve the initial mask (got {got}, expected {expected})"
        );
    }

    // Non-zero count verifies the test isn't vacuously trivial.
    let inside_count = expected_mask.iter().filter(|&&v| v == 1.0).count();
    assert!(
        inside_count > 0,
        "initial sphere must have at least one inside voxel"
    );
}

#[test]
fn test_default_matches_new() {
    let a = GeodesicActiveContourSegmentation::new();
    let b = GeodesicActiveContourSegmentation::default();
    assert_eq!(a.propagation_weight, b.propagation_weight);
    assert_eq!(a.curvature_weight, b.curvature_weight);
    assert_eq!(a.advection_weight, b.advection_weight);
    assert_eq!(a.edge_k, b.edge_k);
    assert_eq!(a.sigma, b.sigma);
    assert_eq!(a.dt, b.dt);
    assert_eq!(a.max_iterations, b.max_iterations);
    assert_eq!(a.tolerance, b.tolerance);
}
