use super::*;
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_image::test_support::{make_image, make_image_1d};

type TestBackend = NdArray<f32>;

fn make_mask_1d(data: Vec<f32>) -> Image<TestBackend, 1> {
    make_image_1d(data)
}

fn make_mask_3d(data: Vec<f32>, dims: [usize; 3]) -> Image<TestBackend, 3> {
    make_image(data, dims)
}

fn values_3d(image: &Image<TestBackend, 3>) -> Vec<f32> {
    image
        .data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec()
}

fn values_1d(image: &Image<TestBackend, 1>) -> Vec<f32> {
    image
        .data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec()
}

fn count_fg_3d(image: &Image<TestBackend, 3>) -> usize {
    values_3d(image).iter().filter(|&&v| v > 0.5).count()
}

// ── radius = 0 is identity ────────────────────────────────────────────────

#[test]
fn test_radius0_is_identity_volumetric() {
    // Structuring element {p} → output = input for any binary mask.
    let data: Vec<f32> = (0u8..27)
        .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
        .collect();
    let mask = make_mask_3d(data.clone(), [3, 3, 3]);
    let result = BinaryErosion::new(0).apply(&mask);
    assert_eq!(
        values_3d(&result),
        data,
        "radius=0 erosion must be identity"
    );
}

#[test]
fn test_radius0_is_identity_line() {
    let data = vec![1.0, 0.0, 1.0, 1.0, 0.0];
    let mask = make_mask_1d(data.clone());
    let result = BinaryErosion::new(0).apply(&mask);
    assert_eq!(
        values_1d(&result),
        data,
        "radius=0 erosion must be identity"
    );
}

// ── All-foreground large image: interior survives ─────────────────────────

#[test]
fn test_all_fg_5x5x5_erosion_r1_keeps_all() {
    // 5×5×5 all-foreground: out-of-bounds is treated as foreground (ITK's
    // BoundaryToForeground = true), so the boundary shell is NOT eroded and all
    // 125 voxels survive — a fully-foreground region erodes to itself.
    let mask = make_mask_3d(vec![1.0_f32; 125], [5, 5, 5]);
    let result = BinaryErosion::new(1).apply(&mask);
    assert_eq!(
        count_fg_3d(&result),
        125,
        "5×5×5 all-fg erosion r=1 keeps all 125 voxels (OOB = foreground)"
    );
}

#[test]
fn test_all_fg_7x7x7_erosion_r2_keeps_all() {
    // 7×7×7 all-fg, r=2: OOB = foreground, so nothing erodes; all 343 survive.
    let mask = make_mask_3d(vec![1.0_f32; 343], [7, 7, 7]);
    let result = BinaryErosion::new(2).apply(&mask);
    assert_eq!(
        count_fg_3d(&result),
        343,
        "7×7×7 all-fg erosion r=2 keeps all 343 voxels (OOB = foreground)"
    );
}

// ── z = 1 (2-D promoted) volume erodes in-plane, not to zero ──────────────

#[test]
fn test_z1_square_erodes_in_plane_not_to_zero() {
    // A 5×5 foreground square inside a [1,7,7] (z=1) volume. With OOB treated as
    // foreground, the degenerate z±1 neighbours do not erode anything, so the
    // square erodes purely in-plane to its 3×3 interior — matching a 2-D ITK
    // erosion. (The previous OOB=background rule eroded every voxel via its
    // out-of-bounds z neighbours, collapsing the whole slice to zero.)
    let mut values = vec![0.0_f32; 7 * 7];
    for y in 1..6 {
        for x in 1..6 {
            values[y * 7 + x] = 1.0;
        }
    }
    let mask = make_mask_3d(values, [1, 7, 7]);
    let result = BinaryErosion::new(1).apply(&mask);
    let out = values_3d(&result);
    assert_eq!(
        count_fg_3d(&result),
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
            assert_eq!(out[y * 7 + x], want, "at (y={y}, x={x})");
        }
    }
}

// ── Single isolated voxel is fully eroded ─────────────────────────────────

#[test]
fn test_single_voxel_eroded_to_empty() {
    // Isolated single foreground voxel in 3×3×3 → fully eroded (boundary).
    let mut values = vec![0.0_f32; 27];
    values[13] = 1.0; // center (1,1,1)
    let mask = make_mask_3d(values, [3, 3, 3]);
    let result = BinaryErosion::new(1).apply(&mask);
    assert_eq!(
        count_fg_3d(&result),
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
    let mask = make_mask_3d(values.clone(), [3, 3, 3]);
    let result = BinaryErosion::new(1).apply(&mask);
    let result_vals = values_3d(&result);
    for (i, (&orig, &out)) in values.iter().zip(result_vals.iter()).enumerate() {
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
    let mask = make_mask_3d(vec![0.0_f32; 27], [3, 3, 3]);
    let result = BinaryErosion::new(1).apply(&mask);
    assert_eq!(
        count_fg_3d(&result),
        0,
        "all-background mask must remain all-background after erosion"
    );
}

// ── 1D erosion: known cases ───────────────────────────────────────────────

#[test]
fn test_1d_erosion_r1_known_output() {
    // Input: [0, 1, 1, 1, 1, 1, 0]
    // The foreground run is [1..=5]. r=1 neighbourhood:
    //   i=1: needs i=0 (bg) → eroded.
    //   i=2: needs i=1 (fg), i=3 (fg) → survives.
    //   i=3: needs i=2 (fg), i=4 (fg) → survives.
    //   i=4: needs i=3 (fg), i=5 (fg) → survives.
    //   i=5: needs i=6 (bg) → eroded.
    // Expected: [0, 0, 1, 1, 1, 0, 0]
    let data = vec![0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0];
    let mask = make_mask_1d(data);
    let result = BinaryErosion::new(1).apply(&mask);
    let out = values_1d(&result);
    let expected = vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
    assert_eq!(out, expected, "1D r=1 erosion output mismatch");
}

#[test]
fn test_1d_all_foreground_erosion_r1() {
    // [1,1,1,1,1] all-foreground: OOB = foreground, so the boundary survives and
    // the line erodes to itself.
    let mask = make_mask_1d(vec![1.0_f32; 5]);
    let result = BinaryErosion::new(1).apply(&mask);
    let out = values_1d(&result);
    assert_eq!(out, vec![1.0, 1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn test_1d_single_voxel_image_survives() {
    // A 1-voxel image is entirely foreground; with OOB = foreground its (OOB)
    // neighbours satisfy the erosion, so it survives — the degenerate
    // whole-image-foreground case, matching ITK.
    let mask = make_mask_1d(vec![1.0]);
    let result = BinaryErosion::new(1).apply(&mask);
    assert_eq!(values_1d(&result), vec![1.0]);
}

// ── Output strictly binary ────────────────────────────────────────────────

#[test]
fn test_output_strictly_binary_volumetric() {
    let values: Vec<f32> = (0u8..27)
        .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
        .collect();
    let mask = make_mask_3d(values, [3, 3, 3]);
    let result = BinaryErosion::new(1).apply(&mask);
    for &v in values_3d(&result).iter() {
        assert!(
            v == 0.0 || v == 1.0,
            "output must be strictly binary, got {v}"
        );
    }
}

// ── Metadata preservation ─────────────────────────────────────────────────

#[test]
fn test_preserves_spatial_metadata() {
    let device: <TestBackend as Backend>::Device = Default::default();
    let tensor = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![1.0f32; 27], Shape::new([3, 3, 3])),
        &device,
    );
    let origin = Point::new([1.0, 2.0, 3.0]);
    let spacing = Spacing::new([0.5, 0.5, 0.5]);
    let direction = Direction::identity();
    let mask: Image<TestBackend, 3> = Image::new(tensor, origin, spacing, direction);

    let result = BinaryErosion::new(1).apply(&mask);

    assert_eq!(result.origin(), &origin, "origin must be preserved");
    assert_eq!(result.spacing(), &spacing, "spacing must be preserved");
    assert_eq!(
        result.direction(),
        &direction,
        "direction must be preserved"
    );
    assert_eq!(result.shape(), [3, 3, 3], "shape must be preserved");
}

// ── Idempotency: erode twice ≡ erode once with larger radius (not equal,
// but monotone: second erosion is subset of first erosion) ──────────────

#[test]
fn test_double_erosion_subset_of_single_erosion() {
    // E(E(M)) ⊆ E(M) for any mask M (monotone).
    let mask = make_mask_3d(vec![1.0_f32; 125], [5, 5, 5]);
    let once = BinaryErosion::new(1).apply(&mask);
    let twice = BinaryErosion::new(1).apply(&once);

    let once_vals = values_3d(&once);
    let twice_vals = values_3d(&twice);

    for (i, (&once_v, &twice_v)) in once_vals.iter().zip(twice_vals.iter()).enumerate() {
        if twice_v > 0.5 {
            assert!(
                once_v > 0.5,
                "double erosion result at index {} not subset of single erosion",
                i
            );
        }
    }
}
