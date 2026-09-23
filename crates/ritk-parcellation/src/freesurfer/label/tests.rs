use super::*;

const SAMPLE: &str = "\
#!ascii label  , from subject bert vox2ras=TkReg
3
1025  -35.125  -18.5  51.25  0.0
7  1.5  2.0  -3.0  0.75
-1  10.0  20.0  30.0  1.0
";

#[test]
fn a_label_reads_to_exactly_its_records() {
    let label = SurfaceLabel::read(SAMPLE.as_bytes()).expect("valid");

    assert_eq!(
        label.comment(),
        "!ascii label  , from subject bert vox2ras=TkReg"
    );
    assert_eq!(
        label.vertices(),
        &[
            LabelVertex {
                vertex: Some(1025),
                position: [-35.125, -18.5, 51.25],
                value: 0.0,
            },
            LabelVertex {
                vertex: Some(7),
                position: [1.5, 2.0, -3.0],
                value: 0.75,
            },
            LabelVertex {
                vertex: None,
                position: [10.0, 20.0, 30.0],
                value: 1.0,
            },
        ]
    );
}

/// `read_label.m` reads records with `fscanf`, which ignores line structure.
#[test]
fn records_are_whitespace_separated_regardless_of_lines() {
    let text = "#c\r\n2\r\n1 0 0 0 0 2\n0.5 0.5 0.5 9\n";
    let label = SurfaceLabel::read(text.as_bytes()).expect("valid");
    assert_eq!(label.comment(), "c");
    assert_eq!(label.vertices()[1].vertex, Some(2));
    assert_eq!(label.vertices()[1].value, 9.0);
}

#[test]
fn a_written_label_reads_back_identically() {
    let label = SurfaceLabel::new(
        "written".to_owned(),
        vec![LabelVertex {
            vertex: Some(3),
            position: [0.1, -1.0e-7, 123_456.789],
            value: -2.5,
        }]
        .into_boxed_slice(),
    )
    .expect("valid");
    let mut text = Vec::new();
    label.write(&mut text).expect("writes");
    assert_eq!(SurfaceLabel::read(text.as_slice()).expect("reads"), label);
}

fn malformed_record(text: &str) -> usize {
    match SurfaceLabel::read(text.as_bytes()) {
        Err(FreeSurferError::Malformed {
            field: "record",
            index,
            ..
        }) => index,
        other => panic!("expected a malformed record, got {other:?}"),
    }
}

#[test]
fn fewer_records_than_the_count_are_rejected() {
    assert_eq!(malformed_record("#c\n3\n1 0 0 0 0\n2 0 0 0 0\n"), 2);
    assert_eq!(malformed_record("#c\n1\n1 0 0\n"), 0);
}

#[test]
fn content_beyond_the_count_is_rejected() {
    assert_eq!(malformed_record("#c\n1\n1 0 0 0 0\n2 0 0 0 0\n"), 1);
}

#[test]
fn a_vertex_number_below_minus_one_is_rejected() {
    assert_eq!(malformed_record("#c\n1\n-2 0 0 0 0\n"), 0);
}

#[test]
fn a_non_numeric_or_non_finite_field_is_rejected() {
    assert_eq!(malformed_record("#c\n1\n1 0 x 0 0\n"), 0);
    assert_eq!(malformed_record("#c\n1\n1 0 0 0 NaN\n"), 0);
    assert_eq!(malformed_record("#c\n1\n1.5 0 0 0 0\n"), 0);
}

#[test]
fn an_unreasonable_count_is_rejected() {
    for (text, count) in [("#c\n-3\n", -3), ("#c\n99999999999\n", 99_999_999_999)] {
        let error =
            SurfaceLabel::read(text.as_bytes()).expect_err("invalid input must be rejected");
        assert!(
            matches!(error, FreeSurferError::InvalidCount { count: got, .. } if got == count),
            "got {error}"
        );
    }
}

#[test]
fn a_missing_header_or_count_is_rejected() {
    for (text, line) in [
        ("", 1),
        ("#only a comment", 1),
        ("#c\n", 2),
        ("#c\nthree\n", 2),
    ] {
        let error =
            SurfaceLabel::read(text.as_bytes()).expect_err("invalid input must be rejected");
        assert!(
            matches!(error, FreeSurferError::Malformed { field: "line", index, .. } if index == line),
            "{text:?}: got {error}"
        );
    }
}
