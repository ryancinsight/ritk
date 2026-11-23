use super::held_gestures::{key, show_two_panels};
use super::*;
use crate::presentation::native_session::layout::{screen_coordinate, CROSSHAIR_COLOR};
use crate::presentation::native_session::NativeViewerSession;
use metis_platform::native::MouseButton;

fn trace() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .with_test_writer()
            .finish(),
    )
}

fn source(root: &std::path::Path, uid: &str) {
    fixtures::write_positioned_study(
        root,
        "CT",
        uid,
        &fixtures::SAMPLES,
        (Some([1.5, 0.5]), true),
        [10, 12, 14],
        Some("2.25.259"),
    )
    .expect("write sagittal stack");
}

#[test]
fn native_linked_display_preserves_offsets_and_optional_window() {
    let _trace = trace();
    let (mut viewer, _initial) = tracing::info_span!("arrange").in_scope(session);
    let root = tempfile::tempdir().expect("study directory");
    for uid in [fixtures::SERIES_UID, SECOND_SERIES_UID] {
        source(root.path(), uid);
    }
    show_two_panels(&mut viewer, root.path());
    viewer.active_panel = 0;
    let (x, y) = viewer.viewports[0].center();
    let drag = |viewer: &mut NativeViewerSession, tool, dx, dy| {
        viewer
            .handle_events(&[
                key(tool),
                WindowEvent::PointerDown {
                    x,
                    y,
                    button: MouseButton::Left,
                },
                WindowEvent::PointerMove {
                    x: x + dx,
                    y: y + dy,
                },
                WindowEvent::PointerUp {
                    x: x + dx,
                    y: y + dy,
                    button: MouseButton::Left,
                },
            ])
            .expect("drag selected tool");
    };
    let _act = tracing::info_span!("act").entered();
    viewer
        .handle_events(&[key(0x75)])
        .expect("disable linked navigation");
    drag(&mut viewer, 0x4d, 8, -4);
    viewer
        .handle_events(&[key(0x75)])
        .expect("enable without moving either pane");
    drag(&mut viewer, 0x4d, 16, 12);
    drag(&mut viewer, 0x5a, 0, -25);
    let _check = tracing::info_span!("assert").entered();
    assert_eq!(viewer.app.pan_offset.x(), 24.);
    let target = &viewer.compare_panels[0].app;
    assert_eq!(
        (target.pan_offset.x(), target.pan_offset.y(), target.zoom),
        (16., 12., 1.25)
    );
    let window = target.browser_window_level_values();
    viewer
        .handle_events(&[key(0x32)])
        .expect("first preset independent");
    assert_ne!(viewer.app.browser_window_level_values(), window);
    assert_eq!(
        viewer.compare_panels[0].app.browser_window_level_values(),
        window
    );
    viewer
        .handle_events(&[
            WindowEvent::KeyDown {
                virtual_key: 0x75,
                repeated: false,
                modifiers: ModifierState::SHIFT,
            },
            key(0x31),
        ])
        .expect("link and apply a new preset");
    assert_eq!(
        viewer.compare_panels[0].app.browser_window_level_values(),
        viewer.app.browser_window_level_values()
    );
}

#[test]
fn native_grid_cursor_tracks_perpendicular_stack_until_release() {
    let _trace = trace();
    let (mut viewer, _initial) = tracing::info_span!("arrange").in_scope(session);
    let root = tempfile::tempdir().expect("directory");
    source(root.path(), fixtures::SERIES_UID);
    fixtures::write_oriented_study(
        root.path(),
        "CT",
        SECOND_SERIES_UID,
        &fixtures::SAMPLES,
        (Some([1.5, 2.0]), true),
        fixtures::PositionedSlices {
            origins: [[10, 20, 29], [10, 20, 30], [10, 20, 31]],
            orientation: "1\\0\\0\\0\\1\\0",
            frame: Some("2.25.259"),
        },
    )
    .expect("write axial comparison");
    show_two_panels(&mut viewer, root.path());
    viewer.active_panel = 0;
    let project = |panel: usize, row, col| {
        let [x, y] = viewer.viewports[panel].mapping().project([row, col]);
        (
            screen_coordinate(x, "x").expect("screen x"),
            screen_coordinate(y, "y").expect("screen y"),
        )
    };
    let (x, y) = project(0, 1.15, 0.5); // (12,20,30.975), closest target z=31.
    let (cx, cy) = project(1, 0.9, 1.5); // vertical mark through x=12
    let modifiers = ModifierState::SHIFT;
    let _act = tracing::info_span!("act").entered();
    viewer
        .handle_events(&[
            WindowEvent::KeyDown {
                virtual_key: 0x11,
                repeated: false,
                modifiers,
            },
            WindowEvent::PointerDown {
                x,
                y,
                button: MouseButton::Left,
            },
        ])
        .expect("momentary native cursor");
    let _check = tracing::info_span!("assert").entered();
    assert_eq!(viewer.compare_panels[0].app.axis_slice_info(0).0, 2);
    assert_eq!(viewer.framebuffer.get_pixel(cx, cy), CROSSHAIR_COLOR);
    viewer
        .handle_events(&[WindowEvent::PointerUp {
            x,
            y,
            button: MouseButton::Left,
        }])
        .expect("release");
    assert_ne!(viewer.framebuffer.get_pixel(cx, cy), CROSSHAIR_COLOR);
}
