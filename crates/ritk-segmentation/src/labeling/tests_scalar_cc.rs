use super::scalar_connected_components;

/// `[10,11,50,51,52]` with threshold 5 → two components `[1,1,2,2,2]`
/// (10–11 and 50–51–52 each within Δ≤5; the 11→50 jump of 39 splits them).
/// Pinned against `sitk.ScalarConnectedComponent`.
#[test]
fn splits_on_value_gap() {
    let v = vec![10.0, 11.0, 50.0, 51.0, 52.0];
    let out = scalar_connected_components(&v, [1, 1, 5], 5.0, 6);
    assert_eq!(out, vec![1.0, 1.0, 2.0, 2.0, 2.0]);
}

/// Threshold 0 (the ITK default) splits every distinct-valued voxel.
#[test]
fn zero_threshold_splits_distinct_values() {
    let v = vec![1.0, 1.0, 2.0, 3.0];
    let out = scalar_connected_components(&v, [1, 1, 4], 0.0, 6);
    assert_eq!(out, vec![1.0, 1.0, 2.0, 3.0]);
}

/// A constant image is a single component regardless of threshold.
#[test]
fn constant_is_one_component() {
    let v = vec![7.0; 8];
    let out = scalar_connected_components(&v, [2, 2, 2], 0.0, 26);
    assert_eq!(out, vec![1.0; 8]);
}

/// 2-D diagonal touch: under 6-connectivity (face) two diagonal same-value cells
/// are separate; under 26-connectivity they merge.
#[test]
fn connectivity_controls_diagonal_merge() {
    // (y,x): [5 . / . 5] — two equal cells touching only at a corner.
    let v = vec![5.0, 0.0, 0.0, 5.0];
    let face = scalar_connected_components(&v, [1, 2, 2], 0.5, 6);
    let full = scalar_connected_components(&v, [1, 2, 2], 0.5, 26);
    // Face: the two 5s are different labels; Full: same label.
    assert_ne!(face[0], face[3]);
    assert_eq!(full[0], full[3]);
}
