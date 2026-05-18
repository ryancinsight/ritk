use super::*;
use crate::tools::interaction::Annotation;

// ── CSV determinism ───────────────────────────────────────────────────────

#[test]
fn csv_empty_annotations_returns_header_only() {
    let annotations: &[Annotation] = &[];
    let csv = csv_for(annotations);
    assert_eq!(
        csv, "index,type,primary_value,unit,extra\n",
        "empty annotation list must produce only the header row"
    );
}

#[test]
fn csv_header_row_is_canonical() {
    let annotations: &[Annotation] = &[];
    let csv = csv_for(annotations);
    let first_line = csv.lines().next().expect("header line must exist");
    assert_eq!(
        first_line, "index,type,primary_value,unit,extra",
        "CSV header must match canonical schema"
    );
}

#[test]
fn csv_row_count_equals_annotation_count_plus_header() {
    let annotations = vec![
        Annotation::Length {
            p1: [0.0, 0.0],
            p2: [3.0, 4.0],
            length_mm: 5.0,
        },
        Annotation::HuPoint {
            pos: [10.0, 20.0],
            value: 42.0,
        },
    ];
    let csv = csv_for(&annotations);
    let line_count = csv.lines().count();
    assert_eq!(
        line_count,
        annotations.len() + 1,
        "CSV must have one header row + one data row per annotation"
    );
}

#[test]
fn csv_length_row_analytical() {
    // Analytical ground truth: length_mm = 5.0, index = 0
    // Expected: "0,Length,5.00,mm,"
    let ann = Annotation::Length {
        p1: [0.0, 0.0],
        p2: [3.0, 4.0],
        length_mm: 5.0,
    };
    let csv = csv_for(std::slice::from_ref(&ann));
    let data_row = csv.lines().nth(1).expect("data row must exist");
    assert_eq!(
        data_row, "0,Length,5.00,mm,",
        "Length annotation CSV row must match analytical format"
    );
}

#[test]
fn csv_angle_row_analytical() {
    // Analytical: angle_deg = 90.0, index = 0
    // Expected: "0,Angle,90.00,deg,"
    let ann = Annotation::Angle {
        p1: [0.0, 0.0],
        p2: [0.0, 1.0],
        p3: [1.0, 1.0],
        angle_deg: 90.0,
    };
    let csv = csv_for(std::slice::from_ref(&ann));
    let data_row = csv.lines().nth(1).expect("data row must exist");
    assert_eq!(
        data_row, "0,Angle,90.00,deg,",
        "Angle annotation CSV row must match analytical format"
    );
}

#[test]
fn csv_roi_rect_row_analytical() {
    // Analytical: mean=100.0, σ=10.0, area=50.0, index=0
    // Expected: "0,ROI Rect,100.00,HU,σ=10.00 area=50.00mm²"
    let ann = Annotation::RoiRect {
        top_left: [0.0, 0.0],
        bottom_right: [5.0, 10.0],
        mean: 100.0,
        std_dev: 10.0,
        min: 80.0,
        max: 120.0,
        area_mm2: 50.0,
    };
    let csv = csv_for(std::slice::from_ref(&ann));
    let data_row = csv.lines().nth(1).expect("data row must exist");
    assert_eq!(
        data_row, "0,ROI Rect,100.00,HU,σ=10.00 area=50.00mm²",
        "ROI Rect CSV row must match analytical format"
    );
}

#[test]
fn csv_roi_ellipse_row_analytical() {
    // Analytical: mean=80.0, σ=5.0, area=25.0, index=0
    // Expected: "0,ROI Ellipse,80.00,HU,σ=5.00 area=25.00mm²"
    let ann = Annotation::RoiEllipse {
        center: [5.0, 5.0],
        radii: [3.0, 2.0],
        mean: 80.0,
        std_dev: 5.0,
        min: 60.0,
        max: 100.0,
        area_mm2: 25.0,
    };
    let csv = csv_for(std::slice::from_ref(&ann));
    let data_row = csv.lines().nth(1).expect("data row must exist");
    assert_eq!(
        data_row, "0,ROI Ellipse,80.00,HU,σ=5.00 area=25.00mm²",
        "ROI Ellipse CSV row must match analytical format"
    );
}

#[test]
fn csv_hu_point_row_analytical() {
    // Analytical: value=200.0, index=0
    // Expected: "0,HU Point,200.00,HU,"
    let ann = Annotation::HuPoint {
        pos: [10.0, 20.0],
        value: 200.0,
    };
    let csv = csv_for(std::slice::from_ref(&ann));
    let data_row = csv.lines().nth(1).expect("data row must exist");
    assert_eq!(
        data_row, "0,HU Point,200.00,HU,",
        "HU Point CSV row must match analytical format"
    );
}

#[test]
fn csv_multi_annotation_indices_are_sequential() {
    // Analytical: three annotations → data rows must carry indices 0, 1, 2
    let annotations = vec![
        Annotation::Length {
            p1: [0.0, 0.0],
            p2: [1.0, 0.0],
            length_mm: 1.0,
        },
        Annotation::HuPoint {
            pos: [0.0, 0.0],
            value: 0.0,
        },
        Annotation::Angle {
            p1: [0.0, 0.0],
            p2: [1.0, 0.0],
            p3: [1.0, 1.0],
            angle_deg: 90.0,
        },
    ];
    let csv = csv_for(&annotations);
    let rows: Vec<&str> = csv.lines().skip(1).collect();
    assert_eq!(rows.len(), 3, "expected 3 data rows");
    assert!(
        rows[0].starts_with("0,"),
        "first data row must have index 0"
    );
    assert!(
        rows[1].starts_with("1,"),
        "second data row must have index 1"
    );
    assert!(
        rows[2].starts_with("2,"),
        "third data row must have index 2"
    );
}

// ── AnnotationPanelAction equality ────────────────────────────────────────

#[test]
fn action_none_equals_none() {
    assert_eq!(AnnotationPanelAction::None, AnnotationPanelAction::None);
}

#[test]
fn action_delete_carries_index() {
    let action = AnnotationPanelAction::Delete(3);
    match action {
        AnnotationPanelAction::Delete(i) => {
            assert_eq!(i, 3, "Delete must carry the exact index")
        }
        _ => panic!("expected Delete variant"),
    }
}

#[test]
fn action_clear_all_is_distinct_from_delete() {
    assert_ne!(
        AnnotationPanelAction::ClearAll,
        AnnotationPanelAction::Delete(0),
        "ClearAll and Delete(0) must be distinct actions"
    );
}

#[test]
fn action_export_csv_payload_matches_csv_for() {
    let annotations = vec![Annotation::Length {
        p1: [0.0, 0.0],
        p2: [3.0, 4.0],
        length_mm: 5.0,
    }];
    let expected_csv = csv_for(&annotations);
    let action = AnnotationPanelAction::ExportCsv(expected_csv.clone());
    match action {
        AnnotationPanelAction::ExportCsv(payload) => {
            assert_eq!(
                payload, expected_csv,
                "ExportCsv payload must equal csv_for output"
            );
        }
        _ => panic!("expected ExportCsv variant"),
    }
}

// ── annotation_label determinism ──────────────────────────────────────────

#[test]
fn label_length_format() {
    let ann = Annotation::Length {
        p1: [0.0, 0.0],
        p2: [3.0, 4.0],
        length_mm: 5.0,
    };
    let label = annotation_label(0, &ann);
    assert!(
        label.contains("5.0 mm"),
        "length label must contain formatted length: got '{label}'"
    );
    assert!(label.starts_with("#0"), "label must begin with index");
}

#[test]
fn label_angle_format() {
    let ann = Annotation::Angle {
        p1: [0.0, 0.0],
        p2: [1.0, 0.0],
        p3: [1.0, 1.0],
        angle_deg: 90.0,
    };
    let label = annotation_label(1, &ann);
    assert!(
        label.contains("90.00°"),
        "angle label must contain formatted angle: got '{label}'"
    );
    assert!(label.starts_with("#1"), "label must begin with index");
}

#[test]
fn label_hu_point_format() {
    let ann = Annotation::HuPoint {
        pos: [10.0, 20.0],
        value: 150.0,
    };
    let label = annotation_label(2, &ann);
    assert!(
        label.contains("150"),
        "HU point label must contain intensity value: got '{label}'"
    );
    assert!(label.starts_with("#2"), "label must begin with index");
}
