use crate::native_support::LegacyBurnBackend;
use super::*;
use ritk_image::test_support as ts;

type B = LegacyBurnBackend;

/// Default labels (fg=1, bg=0): pixels equal to 1 become 0, all others become 1.
#[test]
fn binary_not_default_labels_flip_mask() {
    let img = ts::burn_compat::make_image::<B, 3>(vec![0.0, 1.0, 2.0, 0.0, 5.0], [1, 1, 5]);
    let out = BinaryNotImageFilter::new().apply(&img);
    assert_eq!(out.data_slice().into_owned(), vec![1.0, 0.0, 1.0, 1.0, 1.0]);
}

/// Custom labels (fg=5, bg=9): only the foreground value maps to background.
#[test]
fn binary_not_custom_labels() {
    let img = ts::burn_compat::make_image::<B, 3>(vec![0.0, 1.0, 2.0, 0.0, 5.0], [1, 1, 5]);
    let out = BinaryNotImageFilter::with_labels(5.0, 9.0).apply(&img);
    assert_eq!(out.data_slice().into_owned(), vec![5.0, 5.0, 5.0, 5.0, 9.0]);
}
