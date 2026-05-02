use super::*;

#[test]
fn session_snapshot_default_matches_viewer_defaults() {
    let snapshot = ViewerSessionSnapshot::default();

    assert_eq!(snapshot.source, None);
    assert_eq!(snapshot.viewer_state.slice_index, 0);
    assert_eq!(snapshot.viewer_state.window_center, None);
    assert_eq!(snapshot.viewer_state.window_width, None);
    assert_eq!(snapshot.colormap, Colormap::Grayscale);
    assert_eq!(snapshot.axis, 0);
    assert_eq!(snapshot.active_tool, ToolKind::WindowLevel);
    assert!(!snapshot.multi_planar);
    assert!(snapshot.show_overlay);
    assert!(!snapshot.show_crosshair);
    assert!(snapshot.show_series_browser);
    assert_eq!(snapshot.sidebar_tab, SidebarTab::Series);
    assert_eq!(snapshot.coronal_slice, 0);
    assert_eq!(snapshot.sagittal_slice, 0);
    assert_eq!(snapshot.pan_offset, [0.0, 0.0]);
    assert_eq!(snapshot.zoom, 1.0);
}

#[test]
fn session_snapshot_json_round_trip_preserves_values() {
    let snapshot = ViewerSessionSnapshot {
        source: Some(PathBuf::from("C:/studies/DICOMDIR")),
        viewer_state: ViewerState {
            slice_index: 12,
            window_center: Some(40.0),
            window_width: Some(400.0),
        },
        colormap: Colormap::Bone,
        axis: 2,
        active_tool: ToolKind::MeasureAngle,
        multi_planar: true,
        show_overlay: false,
        show_crosshair: true,
        show_series_browser: false,
        sidebar_tab: SidebarTab::Metadata,
        coronal_slice: 7,
        sagittal_slice: 9,
        pan_offset: [11.5, -3.25],
        zoom: 2.5,
    };

    let json = serde_json::to_string_pretty(&snapshot).expect("serialize snapshot");
    let recovered: ViewerSessionSnapshot =
        serde_json::from_str(&json).expect("deserialize snapshot");

    assert_eq!(recovered, snapshot);
}
