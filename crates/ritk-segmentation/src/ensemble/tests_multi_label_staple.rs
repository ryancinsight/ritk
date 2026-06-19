use super::multi_label_staple;

const TOL: f64 = 1e-5;

/// Unanimous raters: every voxel's consensus is the agreed label.
#[test]
fn multi_label_staple_unanimous_returns_input() {
    let pattern = vec![0.0f32, 1.0, 2.0, 1.0, 0.0];
    let raters = vec![pattern.clone(), pattern.clone(), pattern.clone()];
    let r = multi_label_staple(&raters, None, TOL, None);
    assert_eq!(
        r.labels, pattern,
        "unanimous consensus must equal the input"
    );
    assert_eq!(r.label_for_undecided, 3.0); // L = max(2) + 1
}

/// A 2-of-3 majority wins each voxel.
#[test]
fn multi_label_staple_majority_wins() {
    let raters = vec![vec![0.0f32, 0.0], vec![0.0, 0.0], vec![1.0, 1.0]];
    let r = multi_label_staple(&raters, None, TOL, None);
    assert_eq!(r.labels, vec![0.0, 0.0], "2-of-3 majority for label 0");
}

/// A perfect tie between two raters yields the undecided label everywhere.
#[test]
fn multi_label_staple_tie_is_undecided() {
    let raters = vec![vec![0.0f32, 0.0], vec![1.0, 1.0]];
    let r = multi_label_staple(&raters, None, TOL, None);
    // L = 2, undecided = 2.
    assert_eq!(r.label_for_undecided, 2.0);
    assert_eq!(r.labels, vec![2.0, 2.0], "ties resolve to undecided");
}

/// A custom undecided label is honored.
#[test]
fn multi_label_staple_custom_undecided_label() {
    let raters = vec![vec![0.0f32], vec![1.0]];
    let r = multi_label_staple(&raters, None, TOL, Some(99.0));
    assert_eq!(r.labels, vec![99.0]);
}
