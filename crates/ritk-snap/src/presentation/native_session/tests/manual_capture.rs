use super::super::composition::save_capture;
use super::super::layout::{screen_coordinate, PanelGrid, WorkspaceLayout};
use super::super::startup::prepare_initial_study;
use super::super::window_controls::Menu;
use super::measurement_overlay::{axial_panel, composed};
use super::support::control_center;
use super::*;
use crate::tools::interaction::{Annotation, AnnotationLocation, ImagePoint, ToolState};
use eunomia::FloatElement;
use metis_platform::{native::MouseButton, Color, RasterLine, Rect};
use metis_ui_lang::DisplayCommand;
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

fn rectangles_overlap(a: Rect, b: Rect) -> bool {
    i64::from(a.x) < i64::from(b.x) + i64::from(b.width)
        && i64::from(b.x) < i64::from(a.x) + i64::from(a.width)
        && i64::from(a.y) < i64::from(b.y) + i64::from(b.height)
        && i64::from(b.y) < i64::from(a.y) + i64::from(a.height)
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
    let (pane, origin_x, origin_y, sx, sy) = axial_panel(&viewer);
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
    // The field ends at y=408 and unit buttons at y=448; padding has no ink.
    for x in 509..771 {
        for y in [409, 450] {
            assert_eq!(viewer.framebuffer.get_pixel(x, y), Color::rgb(35, 40, 47));
        }
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
    let calibration = 100.0 / f64::from(*length_mm);
    assert_eq!(viewer.app.length_scale(*location), calibration);

    let original_measurements = viewer.app.annotations.clone();
    viewer.handle_events(&[key(0x4d)]).expect("select Pan");
    viewer
        .handle_events(&drag(point(0.42, 0.23), point(0.48, 0.23)))
        .expect("edit the first length endpoint");
    viewer
        .handle_events(&drag(point(0.32, 0.23), point(0.32, 0.29)))
        .expect("translate the edited length");
    assert_eq!(viewer.app.pointer.selected_annotation, Some(0));
    let Some(Annotation::Length {
        p1, p2, location, ..
    }) = viewer.app.annotations.first()
    else {
        panic!("the edited length is recorded");
    };
    let stored = |(x, y): (i32, i32)| {
        [
            (f64::from(y) - origin_y) / sy,
            (f64::from(x) - origin_x) / sx,
        ]
        .map(<f32 as FloatElement>::from_f64)
    };
    let start = stored(point(0.32, 0.23));
    let end = stored(point(0.32, 0.29));
    let delta = [end[0] - start[0], end[1] - start[1]];
    let expected = [stored(point(0.16, 0.23)), stored(point(0.48, 0.23))]
        .map(|point| [point[0] + delta[0], point[1] + delta[1]]);
    assert_eq!([*p1, *p2], expected);
    assert_eq!(viewer.app.annotations[1], original_measurements[1]);
    assert_eq!(viewer.app.length_scale(*location), calibration);
    if let Some(output) = &output {
        save_capture(
            &viewer.framebuffer,
            &output.join("dicom-native-length-edit.png"),
        )
        .expect("capture selected edited length");
    }

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
    let target = composed(&viewer, None)
        .iter()
        .find_map(|command| match command {
            DisplayCommand::DrawText { text, x, y, style } => style.extent(*x, *y, text),
            _ => None,
        })
        .expect("measurement label");
    let y = target.y + target.height / 2;
    let [row0, col0] = stored((target.x - 2, y));
    let [row1, col1] = stored((target.x + target.width + 2, y));
    viewer.app.tool_state = ToolState::LengthDrag {
        start: ImagePoint::new(col0, row0),
        current: ImagePoint::new(col1, row1),
    };
    viewer.app.pointer.drag_measurement_location = Some(AnnotationLocation {
        plane: viewer.views[pane].axis,
        slice: viewer.views[pane].slice_index,
    });
    let commands = composed(&viewer, None);
    let labels: Vec<_> = commands
        .iter()
        .filter_map(|command| {
            let DisplayCommand::DrawText { text, x, y, style } = command else {
                return None;
            };
            (style.color == Color::rgba(255, 255, 0, 255))
                .then(|| style.extent(*x, *y, text))
                .flatten()
        })
        .collect();
    assert_eq!(labels.len(), 4);
    let strokes: Vec<_> = commands
        .iter()
        .filter_map(|command| {
            let DisplayCommand::DrawLine { start, end, color } = command else {
                return None;
            };
            [
                Color::rgba(255, 255, 0, 255),
                Color::rgba(0, 255, 0, 255),
                Color::rgba(0, 255, 255, 255),
            ]
            .contains(color)
            .then_some((*start, *end))
        })
        .collect();
    assert_eq!(strokes.len(), 21);
    assert!(RasterLine::new(strokes[20].0, strokes[20].1).intersects(target));
    for (start, end) in strokes {
        let stroke = RasterLine::new(start, end);
        for label in &labels {
            assert!(!stroke.intersects(*label));
        }
    }
    for (index, label) in labels.iter().enumerate() {
        for other in labels.iter().skip(index + 1) {
            assert!(!rectangles_overlap(*label, *other));
        }
    }
    if let Some(output) = output {
        save_capture(
            &viewer.framebuffer,
            &output.join("dicom-native-annotations.png"),
        )
        .expect("capture annotation overlays");
    }
}
