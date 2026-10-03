use super::*;

#[test]
fn resize_and_f4_in_one_batch_present_the_picker() {
    let (mut batched, _batched_root) = session();
    let (mut sequential, _sequential_root) = session();
    let study = replacement_study();
    batched
        .open_study_path(study.path())
        .expect("open the study for coalesced input");
    sequential
        .open_study_path(study.path())
        .expect("open the study for sequential input");
    let resize = || WindowEvent::Resized {
        width: 1_024,
        height: 720,
    };
    let open_picker = || WindowEvent::KeyDown {
        virtual_key: 0x73,
        repeated: false,
        modifiers: ModifierState::NONE,
    };

    batched
        .handle_events(&[resize(), open_picker()])
        .expect("resize and open the multiple-series picker in one batch");
    sequential
        .handle_events(&[resize()])
        .expect("resize before opening the multiple-series picker");
    sequential
        .handle_events(&[open_picker()])
        .expect("open the multiple-series picker after resizing");

    assert_eq!(
        (
            batched.window_chrome.multi_series_dialog_is_open(),
            sequential.window_chrome.multi_series_dialog_is_open(),
        ),
        (true, true),
        "both event delivery shapes must open the picker"
    );
    assert_eq!(batched.framebuffer.width(), sequential.framebuffer.width());
    assert_eq!(
        batched.framebuffer.height(),
        sequential.framebuffer.height()
    );
    assert_eq!(
        batched.framebuffer.pixels(),
        sequential.framebuffer.pixels(),
        "coalesced resize and input must present the post-input frame"
    );
}

#[test]
fn queued_input_keeps_its_original_panel_owner_after_maximize() {
    let (mut viewer, _initial_root) = session();
    let study = replacement_study();
    viewer
        .open_study_path(study.path())
        .expect("open two-series study");
    viewer
        .set_workspace_layout(WorkspaceLayout::Panels(
            PanelGrid::new(2, 1).expect("side-by-side grid is valid"),
        ))
        .expect("select side-by-side layout");
    viewer
        .assign_series_to_panel(1, 1)
        .expect("load the second series into panel two");
    viewer.refresh_frame().expect("render both series");

    let hidden_panel = viewer.viewports[0];
    let hidden_slice = viewer.app.viewer_state.slice_index;
    let visible_slice = viewer.compare_panels[0].app.viewer_state.slice_index;
    let maximize_x = i32::try_from(
        viewer.viewports[1]
            .panel_x
            .saturating_add(viewer.viewports[1].panel_width)
            .saturating_sub(23),
    )
    .expect("panel maximize x fits i32");
    let maximize_y = i32::try_from(viewer.viewports[1].panel_y.saturating_sub(13))
        .expect("panel maximize y fits i32");
    let (hidden_x, hidden_y) = hidden_panel.center();

    viewer
        .handle_events(&[
            WindowEvent::PointerDown {
                x: maximize_x,
                y: maximize_y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: maximize_x,
                y: maximize_y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerWheel {
                x: hidden_x,
                y: hidden_y,
                delta_x: 0,
                delta_y: -120,
                modifiers: ModifierState::NONE,
            },
        ])
        .expect("ignore input to the panel hidden by the maximize action");

    assert_eq!(
        viewer.workspace_layout.grid().map(PanelGrid::panel_count),
        Some(1)
    );
    assert_eq!(loaded_series_uid(&viewer.app), Some(SECOND_SERIES_UID));
    assert_eq!(viewer.app.viewer_state.slice_index, visible_slice);
    assert_eq!(
        loaded_series_uid(&viewer.compare_panels[0].app),
        Some(fixtures::SERIES_UID)
    );
    assert_eq!(
        viewer.compare_panels[0].app.viewer_state.slice_index,
        hidden_slice.saturating_add(1),
        "the event is routed against the image layout visible at batch start"
    );
}

#[test]
fn queued_series_browse_replaces_its_original_panel_after_maximize() {
    let (mut viewer, _initial_root) = session();
    let study = four_series_study();
    viewer
        .open_study_path(study.path())
        .expect("open four-series study");
    viewer
        .set_workspace_layout(WorkspaceLayout::Panels(
            PanelGrid::new(2, 1).expect("side-by-side grid is valid"),
        ))
        .expect("select side-by-side layout");
    viewer
        .assign_series_to_panel(2, 1)
        .expect("load the third series into panel two");
    viewer.refresh_frame().expect("render both series");

    let original_first_panel = viewer.viewports[0];
    let maximized = viewer.viewports[1];
    let maximize_x = i32::try_from(
        maximized
            .panel_x
            .saturating_add(maximized.panel_width)
            .saturating_sub(23),
    )
    .expect("panel maximize x fits i32");
    let maximize_y =
        i32::try_from(maximized.panel_y.saturating_sub(13)).expect("panel maximize y fits i32");
    let (browse_x, browse_y) = original_first_panel.center();

    viewer
        .handle_events(&[
            WindowEvent::PointerDown {
                x: maximize_x,
                y: maximize_y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: maximize_x,
                y: maximize_y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerWheel {
                x: browse_x,
                y: browse_y,
                delta_x: 120,
                delta_y: 0,
                modifiers: ModifierState::NONE,
            },
        ])
        .expect("route queued series browsing to its original panel");

    assert_eq!(
        viewer.workspace_layout.grid().map(PanelGrid::panel_count),
        Some(1)
    );
    assert_eq!(viewer.primary_series_index, Some(2));
    assert_eq!(viewer.compare_panels[0].series_index, Some(1));
    assert_eq!(loaded_series_uid(&viewer.app), Some(THIRD_SERIES_UID));
    assert_eq!(
        loaded_series_uid(&viewer.compare_panels[0].app),
        Some(SECOND_SERIES_UID)
    );
    assert_eq!(viewer.active_panel, 0);
}

#[test]
fn queued_drag_keeps_its_original_viewport_mapping_after_maximize() {
    let (mut viewer, _initial_root) = session();
    let study = replacement_study();
    viewer
        .open_study_path(study.path())
        .expect("open two-series study");
    viewer
        .set_workspace_layout(WorkspaceLayout::Panels(
            PanelGrid::new(2, 1).expect("side-by-side grid is valid"),
        ))
        .expect("select side-by-side layout");
    viewer
        .assign_series_to_panel(1, 1)
        .expect("load the second series into panel two");
    viewer.refresh_frame().expect("render both series");
    viewer.app.active_tool = crate::tools::kind::ToolKind::Pan;

    let original_viewport = viewer.viewports[0];
    let primary_offset = viewer.app.pan_offset;
    let maximized_offset = viewer.compare_panels[0].app.pan_offset;
    let original_mapping = original_viewport.mapping();
    let (start_x, start_y) = original_viewport.center();
    let end_x = start_x.saturating_add(24);
    let end_y = start_y.saturating_add(12);
    let expected = pan_from_drag_delta(
        primary_offset,
        original_mapping
            .map(ViewportPoint::new(f64::from(start_x), f64::from(start_y)))
            .expect("drag starts over the original image"),
        original_mapping
            .map(ViewportPoint::new(f64::from(end_x), f64::from(end_y)))
            .expect("drag ends over the original image"),
    );
    let maximized = viewer.viewports[1];
    let maximize_x = i32::try_from(
        maximized
            .panel_x
            .saturating_add(maximized.panel_width)
            .saturating_sub(23),
    )
    .expect("panel maximize x fits i32");
    let maximize_y =
        i32::try_from(maximized.panel_y.saturating_sub(13)).expect("panel maximize y fits i32");

    viewer
        .handle_events(&[
            WindowEvent::PointerDown {
                x: maximize_x,
                y: maximize_y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: maximize_x,
                y: maximize_y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerDown {
                x: start_x,
                y: start_y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerMove { x: end_x, y: end_y },
            WindowEvent::PointerUp {
                x: end_x,
                y: end_y,
                button: MouseButton::Left,
            },
        ])
        .expect("map queued drag coordinates against the visible frame");

    assert_eq!(viewer.compare_panels[0].app.pan_offset, expected);
    assert_eq!(
        viewer.app.pan_offset, maximized_offset,
        "the maximized panel keeps its independent view state"
    );
    assert_ne!(expected, primary_offset);
}
