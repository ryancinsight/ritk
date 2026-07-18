//! Value-semantic tests for [`merge_label_maps`] against sitk reference outputs.
//!
//! Expected vectors are the exact `sitk.LabelMapToLabel(sitk.MergeLabelMap(...))`
//! results for two z=1 label images (see module docs); they are not derived from
//! ritk and so constitute a genuine differential oracle.

use super::{merge_label_maps, MergeLabelError, MergeLabelMethod};
use ritk_image::test_support as ts;
use ritk_tensor_ops::extract_vec_infallible;

type B = coeus_core::SequentialBackend;

/// Build a z=1 label image from a 4×5 row-major slice.
fn img(rows: [[f32; 5]; 4]) -> ritk_image::Image<f32, B, 3> {
    let data: Vec<f32> = rows.iter().flatten().copied().collect();
    ts::make_image::<f32, B, 3>(data, [1, 4, 5])
}

fn inputs() -> (ritk_image::Image<f32, B, 3>, ritk_image::Image<f32, B, 3>) {
    let a = img([
        [1.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 3.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 5.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]);
    let b = img([
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 3.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 7.0],
        [0.0, 0.0, 0.0, 2.0, 0.0],
    ]);
    (a, b)
}

fn run(method: MergeLabelMethod) -> Vec<f32> {
    let (a, b) = inputs();
    let out = merge_label_maps(&[&a, &b], method).expect("merge succeeds");
    extract_vec_infallible(&out).0
}

#[test]
fn keep_matches_sitk() {
    let expect = vec![
        1.0, 8.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 7.0, 0.0, 0.0, 0.0,
        2.0, 0.0,
    ];
    assert_eq!(run(MergeLabelMethod::Keep), expect);
}

#[test]
fn aggregate_matches_sitk() {
    let expect = vec![
        1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 7.0, 0.0, 0.0, 0.0,
        2.0, 0.0,
    ];
    assert_eq!(run(MergeLabelMethod::Aggregate), expect);
}

#[test]
fn pack_matches_sitk() {
    let expect = vec![
        1.0, 4.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 7.0, 0.0, 0.0, 0.0,
        5.0, 0.0,
    ];
    assert_eq!(run(MergeLabelMethod::Pack), expect);
}

#[test]
fn strict_errors_on_collision() {
    let (a, b) = inputs();
    // labels 1 and 3 collide between a and b.
    let err = merge_label_maps(&[&a, &b], MergeLabelMethod::Strict).unwrap_err();
    assert_eq!(err, MergeLabelError::StrictConflict { label: 1, input: 1 });
}

#[test]
fn strict_disjoint_keeps_labels() {
    let a = img([
        [1.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 3.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 5.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]);
    let d = img([
        [0.0, 0.0, 0.0, 0.0, 10.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 11.0, 0.0, 0.0],
    ]);
    let out = merge_label_maps(&[&a, &d], MergeLabelMethod::Strict).expect("disjoint ok");
    let v = extract_vec_infallible(&out).0;
    assert_eq!(v[0], 1.0);
    assert_eq!(v[4], 10.0);
    assert_eq!(v[17], 11.0);
}

#[test]
fn no_inputs_errors() {
    let empty: [&ritk_image::Image<f32, B, 3>; 0] = [];
    assert_eq!(
        merge_label_maps(&empty, MergeLabelMethod::Keep).unwrap_err(),
        MergeLabelError::NoInputs
    );
}
