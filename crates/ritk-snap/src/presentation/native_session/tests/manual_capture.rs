use super::super::composition::save_capture;
use super::super::layout::{screen_coordinate, PanelGrid, WorkspaceLayout};
use super::super::startup::prepare_initial_study;
use super::super::window_controls::Menu;
use super::measurement_overlay::axial_panel;
use super::support::control_center;
use super::*;
use crate::tools::interaction::Annotation;
use metis_platform::native::MouseButton;
use std::path::PathBuf;

fn key(virtual_key: u32) -> WindowEvent {
    WindowEvent::KeyDown {
        virtual_key,
        repeated: false,
        modifiers: ModifierState::NONE,
    }
}

fn down((x, y): (i32, i32)) -> WindowEvent {
    WindowEvent::PointerDown {
        x,
        y,
        button: MouseButton::Left,
    }
}

fn up((x, y): (i32, i32)) -> WindowEvent {
    WindowEvent::PointerUp {
        x,
        y,
        button: MouseButton::Left,
    }
}

fn click(point: (i32, i32)) -> [WindowEvent; 2] {
    [down(point), up(point)]
}

fn drag(start: (i32, i32), end: (i32, i32)) -> [WindowEvent; 3] {
    [
        down(start),
        WindowEvent::PointerMove { x: end.0, y: end.1 },
        up(end),
    ]
}

#[test]
fn native_manual_annotation_frames() {
    let fixture = tempfile::tempdir().expect("manual fixture directory");
    let study = if let Some(path) = std::env::var_os("RITK_MANUAL_STUDY") {
        PathBuf::from(path)
    } else {
        fixtures::write_study(
            fixture.path(),
            "CT",
            fixtures::SERIES_UID,
            &fixtures::SAMPLES,
        )
        .expect("write manual fixture");
        fixture.path().to_path_buf()
    };
    let mut app = SnapApp::default();
    let browser = prepare_initial_study(&mut app, &study, None, true).expect("open manual study");
    let mut viewer = NativeViewerSession::new_with_browser(
        app,
        Arc::new(NativeViewerObservation::default()),
        false,
        NativePresentationSelection::Fixed(NativePresentationMode::Orthogonal),
        false,
        browser,
        None,
    )
    .expect("show the study and series preview");
    let output = std::env::var_os("RITK_MANUAL_CAPTURE_DIR").map(PathBuf::from);
    if let Some(output) = &output {
        save_capture(&viewer.framebuffer, &output.join("dicom-native-slices.png"))
            .expect("capture slice scrollbars in the MPR layout");
    }
    let grid = PanelGrid::new(1, 1).expect("single panel layout");
    viewer
        .set_workspace_layout(WorkspaceLayout::Panels(grid))
        .expect("show a single annotation pane");
    viewer.refresh_frame().expect("display the annotation pane");
    let (pane, ..) = axial_panel(&viewer);
    let image = viewer.viewports[pane].image;
    let point = |x: f64, y: f64| {
        (
            screen_coordinate(x.mul_add(image.width, image.x), "manual x").expect("screen x"),
            screen_coordinate(y.mul_add(image.height, image.y), "manual y").expect("screen y"),
        )
    };
    viewer.handle_events(&[key(0x4c)]).expect("select Length");
    for y in [0.23, 0.39] {
        viewer
            .handle_events(&drag(point(0.16, y), point(0.42, y)))
            .expect("measure length");
    }
    let tools = control_center(&viewer, None, WindowAction::OpenMenu(Menu::Tools));
    let calibrate = control_center(&viewer, Some(Menu::Tools), WindowAction::LengthCalibration);
    viewer.handle_events(&click(tools)).expect("open Tools");
    viewer
        .handle_events(&click(calibrate))
        .expect("open calibration");
    for character in ['1', '0', '0'] {
        viewer
            .handle_events(&[WindowEvent::TextInput { character }])
            .expect("enter calibration length");
    }
    if let Some(output) = &output {
        save_capture(
            &viewer.framebuffer,
            &output.join("dicom-native-calibration.png"),
        )
        .expect("capture calibration input");
    }
    viewer
        .handle_events(&[key(0x0d)])
        .expect("apply calibration");
    let Some(Annotation::Length {
        length_mm,
        location,
        ..
    }) = viewer.app.annotations.last()
    else {
        panic!("the reference length is recorded");
    };
    assert_eq!(
        viewer.app.length_scale(*location),
        100.0 / f64::from(*length_mm)
    );

    viewer
        .handle_events(&[key(0x43)])
        .expect("select Cobb Angle");
    for (start, end) in [
        (point(0.58, 0.25), point(0.81, 0.25)),
        (point(0.68, 0.35), point(0.68, 0.55)),
    ] {
        viewer
            .handle_events(&drag(start, end))
            .expect("draw Cobb segment");
    }
    viewer
        .handle_events(&[key(0x4f)])
        .expect("select Open Polygon");
    for vertex in [point(0.15, 0.67), point(0.37, 0.67), point(0.35, 0.84)] {
        viewer
            .handle_events(&click(vertex))
            .expect("place polygon vertex");
    }
    viewer
        .handle_events(&[key(0x0d), key(0x59)])
        .expect("finish path and select Arrow");
    viewer
        .handle_events(&drag(point(0.60, 0.73), point(0.80, 0.63)))
        .expect("draw Arrow");
    viewer.handle_events(&[key(0x50)]).expect("select Pencil");
    let sample = point(0.65, 0.80);
    viewer
        .handle_events(&[
            down(point(0.57, 0.76)),
            WindowEvent::PointerMove {
                x: sample.0,
                y: sample.1,
            },
            up(point(0.75, 0.75)),
        ])
        .expect("draw Pencil");
    viewer.handle_events(&[key(0x4d)]).expect("select Pan");
    viewer
        .handle_events(&click(point(0.26, 0.67)))
        .expect("select path");
    viewer
        .handle_events(&drag(point(0.26, 0.67), point(0.25, 0.71)))
        .expect("insert and move a vertex");

    let [Annotation::Length { .. }, Annotation::Length { .. }, Annotation::CobbAngle { angle_deg, .. }, Annotation::OpenPolygon { points, .. }, Annotation::Arrow { .. }, Annotation::Pencil {
        points: freehand, ..
    }] = viewer.app.annotations.as_slice()
    else {
        panic!("the capture records two lengths, Cobb angle, open polygon, arrow and pencil");
    };
    assert_eq!(*angle_deg, 90.0);
    assert_eq!(points.len(), 4);
    assert_eq!(freehand.len(), 3);
    if let Some(output) = output {
        save_capture(
            &viewer.framebuffer,
            &output.join("dicom-native-annotations.png"),
        )
        .expect("capture annotation overlays");
    }
}
