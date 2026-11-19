use crate::session::ViewerSessionSnapshot;

#[test]
fn default_snapshot_retains_wire_field_names_order_and_values() {
    let json = serde_json::to_string(&ViewerSessionSnapshot::default())
        .expect("default session snapshot is serializable");

    assert_eq!(
        json,
        concat!(
            r#"{"format":3,"source":null,"viewer_state":{"slice_index":0,"window_center":null,"window_width":null},"#,
            r#""colormap":"Grayscale","axis":0,"active_tool":"WindowLevel","multi_planar":false,"show_overlay":true,"#,
            r#""show_crosshair":false,"show_rt_struct_overlay":true,"show_rt_dose_overlay":false,"rt_dose_opacity":0.5,"#,
            r#""show_series_browser":true,"sidebar_tab":"Series","coronal_slice":0,"sagittal_slice":0,"pan_offset":[0.0,0.0],"#,
            r#""zoom":1.0,"cine_enabled":false,"cine_fps":12.0,"annotations":[]}"#,
        )
    );
}

#[test]
fn deserializer_retains_public_struct_name() {
    let error = serde_json::from_str::<ViewerSessionSnapshot>("null")
        .expect_err("null is not a viewer session snapshot");

    assert!(
        error.to_string().contains("struct ViewerSessionSnapshot"),
        "serde formats observe the public struct name: {error}"
    );
}
