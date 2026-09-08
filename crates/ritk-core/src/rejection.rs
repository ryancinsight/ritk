//! Rejection assertions for the workspace's test modules.
//!
//! A bare `is_err` assertion passes whenever a call fails for *any* reason, so
//! a test written for one precondition also passes when an unrelated guard
//! fires first — an assertion that cannot fail on the defect it was written
//! for. Measured in this workspace: a DICOM loader checks path existence
//! before format, so a test named for the unsupported-format branch that
//! passed a nonexistent path never reached it, and was indistinguishable from
//! its missing-file sibling.
//!
//! These make the claim value-semantic: the rejection must name its cause.
//! Most of the tree returns `anyhow::Error`, so the fragment is matched
//! against the rendered chain rather than a variant.
//!
//! Gated behind `test-helpers`, the feature this crate already uses for
//! test-only surfaces, so nothing enters a default build.

/// Assert a call was rejected and that the rejection names `fragment`.
///
/// The whole error chain is rendered (`{:#}`), so a fragment contributed by a
/// wrapped source still matches — which is what makes the assertion usable on
/// the `anyhow` contexts this workspace layers.
///
/// # Panics
///
/// Panics when `result` is `Ok`, or when the rendered chain does not contain
/// `fragment`.
#[track_caller]
pub fn assert_rejects<T, E: std::fmt::Display>(result: Result<T, E>, fragment: &str) {
    match result {
        Err(error) => {
            let rendered = format!("{error:#}");
            assert!(
                rendered.contains(fragment),
                "rejection must name its cause: expected a message containing \
                 {fragment:?}, got {rendered:?}"
            );
        }
        Ok(_) => panic!("expected a rejection naming {fragment:?}, got Ok"),
    }
}
