use super::*;

#[test]
fn direct_annotations_preserve_wire_identity_order_and_values() {
    let expected = [
        r#"{"Length":{"plane":0,"location":null,"p1":[0.0,0.0],"p2":[3.0,4.0],"length_mm":123.25}}"#,
        r#"{"PatientLength":{"plane":2,"location":null,"length":{"start_mm":[0.0,0.0,0.0],"end_mm":[3.0,4.0,0.0]}}}"#,
        r#"{"Angle":{"location":null,"p1":[0.0,1.0],"p2":[0.0,0.0],"p3":[1.0,0.0],"angle_deg":90.0}}"#,
        r#"{"RoiRect":{"location":null,"top_left":[10.0,10.0],"bottom_right":[20.0,30.0],"mean":42.5,"std_dev":1.25,"min":40.0,"max":45.0,"area_mm2":200.0}}"#,
        r#"{"RoiEllipse":{"location":null,"center":[15.0,20.0],"radii":[5.0,10.0],"mean":37.0,"std_dev":2.0,"min":33.0,"max":41.0,"area_mm2":314.15927}}"#,
        r#"{"HuPoint":{"location":null,"pos":[5.0,8.0],"value":-150.0}}"#,
        r#"{"Deviation":{"location":{"plane":1,"slice":4},"start":[2.0,3.0],"end":[5.0,7.0],"reference":[5.0,3.0],"distance_mm":12.5,"angle_deg":53.0}}"#,
        r#"{"ClosedPolygon":{"location":{"plane":2,"slice":6},"points":[[1.0,1.0],[1.0,6.0],[4.0,1.0]],"area_mm2":77.25,"perimeter_mm":44.5}}"#,
        r#"{"OpenPolygon":{"location":null,"points":[[2.0,2.0],[4.0,7.0],[9.0,7.0]],"length_mm":31.75}}"#,
        r#"{"Arrow":{"location":{"plane":0,"slice":8},"tail":[3.0,4.0],"head":[8.0,12.0]}}"#,
        r#"{"Pencil":{"location":null,"points":[[0.5,1.5],[2.5,3.5],[4.5,5.5]]}}"#,
        r#"{"CobbAngle":{"location":{"plane":1,"slice":10},"points":[[0.0,0.0],[3.0,1.0],[5.0,8.0],[9.0,9.0]],"angle_deg":42.25}}"#,
    ];
    let annotations = all_annotation_variants();

    assert_eq!(annotations.len(), expected.len());
    for (annotation, expected_json) in annotations.iter().zip(expected) {
        let actual_json = serde_json::to_string(annotation).expect("serialize direct annotation");
        assert_eq!(actual_json, expected_json);
        assert_eq!(
            serde_json::from_str::<Annotation>(&actual_json)
                .expect("deserialize direct annotation"),
            *annotation
        );
    }
}

#[test]
fn direct_annotation_defaults_and_cached_values_remain_authoritative() {
    let json = serde_json::json!({
        "Length": {"p1": [0.0, 0.0], "p2": [3.0, 4.0], "length_mm": 123.25}
    });
    let annotation = serde_json::from_value::<Annotation>(json).expect("legacy length loads");
    assert_eq!(
        annotation,
        Annotation::Length {
            plane: 0,
            location: None,
            p1: [0.0, 0.0],
            p2: [3.0, 4.0],
            length_mm: 123.25,
        }
    );

    let original_json = serde_json::to_string(&annotation).expect("serialize cached length");
    let mut changed = annotation;
    let Annotation::Length { length_mm, .. } = &mut changed else {
        panic!("fixture is a length annotation");
    };
    *length_mm = 321.5;
    let changed_json = serde_json::to_string(&changed).expect("serialize changed cached length");
    assert_ne!(changed_json, original_json);
    assert_eq!(
        serde_json::from_str::<Annotation>(&changed_json).expect("deserialize changed length"),
        changed
    );
}

#[test]
fn malformed_annotation_wire_values_are_rejected() {
    let missing = serde_json::from_value::<Annotation>(serde_json::json!({
        "Length": {"p2": [3.0, 4.0], "length_mm": 123.25}
    }))
    .expect_err("required p1 is absent");
    assert_eq!(missing.to_string(), "missing field `p1`");

    let unknown = serde_json::from_value::<Annotation>(serde_json::json!({"Unknown": {}}))
        .expect_err("unknown annotation variant rejects");
    assert_eq!(
        unknown.to_string(),
        "unknown variant `Unknown`, expected one of `Length`, `PatientLength`, `Angle`, `RoiRect`, `RoiEllipse`, `Deviation`, `HuPoint`, `ClosedPolygon`, `OpenPolygon`, `Arrow`, `Pencil`, `CobbAngle`"
    );

    let invalid_patient = serde_json::from_value::<Annotation>(serde_json::json!({
        "PatientLength": {
            "length": {
                "start_mm": [-1.0e308, 0.0, 0.0],
                "end_mm": [1.0e308, 0.0, 0.0]
            }
        }
    }))
    .expect_err("overflowing patient displacement rejects");
    assert_eq!(
        invalid_patient.to_string(),
        "patient length measurement result is not finite"
    );
}
