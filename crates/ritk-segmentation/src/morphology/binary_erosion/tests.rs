//! Algebraic and known-value tests for the canonical host-slice erosion kernel.

use super::erode;

/// Helper — count voxels with value strictly above the foreground threshold.
/// Threshold re-asserted here since the legacy constant is `pub(crate)`.
#[inline]
fn count_fg(flat: &[f32]) -> usize {
    flat.iter().filter(|&&v| v > 0.5).count()
}

// ── radius = 0 is identity ────────────────────────────────────────────────

#[test]
fn test_radius0_is_identity_volumetric() {
    // Structuring element {p} → output = input for any binary mask.
    let data: Vec<f32> = (0u8..27)
        .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
        .collect();
    let shape = [3usize, 3, 3];
    let result = erode(&data, &shape, 0);
    assert_eq!(result, data, "radius=0 erosion must be identity");
}

#[test]
fn test_radius0_is_identity_line() {
    let data = vec![1.0, 0.0, 1.0, 1.0, 0.0];
    let shape = [data.len()];
    let result = erode(&data, &shape, 0);
    assert_eq!(result, data, "radius=0 erosion must be identity");
}

// ── All-foreground large image: interior survives ─────────────────────────

#[test]
fn test_all_fg_5x5x5_erosion_r1_keeps_all() {
    // 5×5×5 all-foreground: OOB is treated as foreground (ITK's
    // BoundaryToForeground = true), so the boundary shell is NOT eroded and all
    // 125 voxels survive.
    let flat = vec![1.0_f32; 125];
    let shape = [5usize, 5, 5];
    let result = erode(&flat, &shape, 1);
    assert_eq!(
        count_fg(&result),
        125,
        "5×5×5 all-fg erosion r=1 keeps all 125 voxels (OOB = foreground)"
    );
}

#[test]
fn test_all_fg_7x7x7_erosion_r2_keeps_all() {
    // 7×7×7 all-fg, r=2: OOB = foreground, so nothing erodes; all 343 survive.
    let flat = vec![1.0_f32; 343];
    let shape = [7usize, 7, 7];
    let result = erode(&flat, &shape, 2);
    assert_eq!(
        count_fg(&result),
        343,
        "7×7×7 all-fg erosion r=2 keeps all 343 voxels (OOB = foreground)"
    );
}

// ── z = 1 (2-D promoted) volume erodes in-plane, not to zero ──────────────

#[test]
fn test_z1_square_erodes_in_plane_not_to_zero() {
    // 5×5 foreground square inside a [1, 7, 7] (z=1) volume. With OOB = fg,
    // the degenerate z±1 neighbours do not erode anything, so the square
    // erodes purely in-plane to its 3×3 interior.
    let mut values = vec![0.0_f32; 7 * 7];
    for y in 1..6 {
        for x in 1..6 {
            values[y * 7 + x] = 1.0;
        }
    }
    let shape = [1usize, 7, 7];
    let result = erode(&values, &shape, 1);
    assert_eq!(
        count_fg(&result),
        9,
        "z=1 square must erode to 3×3 = 9, not 0"
    );
    for y in 0..7 {
        for x in 0..7 {
            let want = if (2..5).contains(&y) && (2..5).contains(&x) {
                1.0
            } else {
                0.0
            };
            assert_eq!(result[y * 7 + x], want, "at (y={y}, x={x})");
        }
    }
}

// ── Single isolated voxel is fully eroded ─────────────────────────────────

#[test]
fn test_single_voxel_eroded_to_empty() {
    // Isolated single foreground voxel in 3×3×3 → fully eroded (boundary).
    let mut values = vec![0.0_f32; 27];
    values[13] = 1.0; // center (1, 1, 1)
    let shape = [3usize, 3, 3];
    let result = erode(&values, &shape, 1);
    assert_eq!(
        count_fg(&result),
        0,
        "isolated single voxel must be fully eroded"
    );
}

// ── Anti-extensivity invariant: eroded ⊆ input ───────────────────────────

#[test]
fn test_erosion_is_anti_extensive() {
    let values: Vec<f32> = (0u8..27)
        .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
        .collect();
    let shape = [3usize, 3, 3];
    let result = erode(&values, &shape, 1);
    for (i, (&orig, &out)) in values.iter().zip(result.iter()).enumerate() {
        if out > 0.5 {
            assert!(
                orig > 0.5,
                "erosion introduced foreground at index {} where input was background",
                i
            );
        }
    }
}

// ── All-background stays all-background ───────────────────────────────────

#[test]
fn test_all_background_stays_empty() {
    let flat = vec![0.0_f32; 27];
    let shape = [3usize, 3, 3];
    let result = erode(&flat, &shape, 1);
    assert_eq!(
        count_fg(&result),
        0,
        "all-background mask must remain all-background after erosion"
    );
}

// ── 1D erosion: known cases ───────────────────────────────────────────────

#[test]
fn test_1d_erosion_r1_known_output() {
    // Input: [0, 1, 1, 1, 1, 1, 0]
    // Foreground run [1..=5]. r=1 neighbourhood:
    //   i=1: needs i=0 (bg) → eroded.
    //   i=2: needs i=1 (fg), i=3 (fg) → survives.
    //   i=3: needs i=2 (fg), i=4 (fg) → survives.
    //   i=4: needs i=3 (fg), i=5 (fg) → survives.
    //   i=5: needs i=6 (bg) → eroded.
    // Expected: [0, 0, 1, 1, 1, 0, 0]
    let data = vec![0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0];
    let shape = [data.len()];
    let result = erode(&data, &shape, 1);
    let expected = vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
    assert_eq!(result, expected, "1D r=1 erosion output mismatch");
}

#[test]
fn test_1d_all_foreground_erosion_r1() {
    // [1, 1, 1, 1, 1] all-foreground: OOB = foreground, so the boundary
    // survives and the line erodes to itself.
    let flat = vec![1.0_f32; 5];
    let shape = [flat.len()];
    let result = erode(&flat, &shape, 1);
    assert_eq!(result, vec![1.0, 1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn test_1d_single_voxel_image_survives() {
    // 1-voxel image is entirely foreground; with OOB = foreground its (OOB)
    // neighbours satisfy the erosion, so it survives — the degenerate
    // whole-image-foreground case, matching ITK.
    let flat = vec![1.0_f32; 1];
    let shape = [flat.len()];
    let result = erode(&flat, &shape, 1);
    assert_eq!(result, vec![1.0]);
}

// ── Output strictly binary ────────────────────────────────────────────────

#[test]
fn test_output_strictly_binary_volumetric() {
    let values: Vec<f32> = (0u8..27)
        .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
        .collect();
    let shape = [3usize, 3, 3];
    let result = erode(&values, &shape, 1);
    for &v in result.iter() {
        assert!(
            v == 0.0 || v == 1.0,
            "output must be strictly binary, got {v}"
        );
    }
}

// ── Atlas-side row-major raster-shape preservation ───────────────────────

#[test]
fn test_shape_preserves_voxel_count() {
    // The atlas twin operates on raw slices; the only spatial invariant it can
    // consume and produce is raster shape (length, and per-axis extents via the
    // input `shape` slice). Verify the
    //   output.len() == input.len() == shape.iter().product::<usize>()
    // triangle, the contractual shape invariant for atlas-side callers using
    // `AtlasImage<f32, MoiraiBackend, 3>` rasterized over `coeus_tensor::Tensor`.
    let flat: Vec<f32> = (0u8..27)
        .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
        .collect();
    let shape = [3usize, 3, 3];
    let result = erode(&flat, &shape, 1);
    assert_eq!(
        result.len(),
        flat.len(),
        "output length must equal input length"
    );
    assert_eq!(
        result.len(),
        shape.iter().product::<usize>(),
        "length must match shape product"
    );
}

// ── Idempotency: erode twice ≡ erode once with larger radius (not equal,
// but monotone: second erosion is subset of first erosion) ──────────────

#[test]
fn test_double_erosion_subset_of_single_erosion() {
    // E(E(M)) ⊆ E(M) for any mask M (monotone).
    let flat = vec![1.0_f32; 125];
    let shape = [5usize, 5, 5];
    let once = erode(&flat, &shape, 1);
    let twice = erode(&once, &shape, 1);

    for (i, (&once_v, &twice_v)) in once.iter().zip(twice.iter()).enumerate() {
        if twice_v > 0.5 {
            assert!(
                once_v > 0.5,
                "double erosion result at index {} not subset of single erosion",
                i
            );
        }
    }
}
