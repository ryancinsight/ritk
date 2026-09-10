//! Metadata popup input and lifecycle regressions.

use super::*;

fn details_frame(
    app: &mut crate::app::SnapApp,
    context: &egui::Context,
    text: &str,
    events: Vec<egui::Event>,
) -> egui::FullOutput {
    context.run(
        egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, egui::vec2(800.0, 600.0))),
            events,
            ..egui::RawInput::default()
        },
        |context| {
            app.consume_global_shortcuts(context);
            egui::CentralPanel::default().show(context, |ui| {
                OverlayRenderer::show_details(
                    ui,
                    Rect::from_min_size(Pos2::new(20.0, 20.0), egui::vec2(176.0, 88.0)),
                    text,
                );
            });
        },
    )
}

#[test]
fn details_activation_displays_metadata_and_escape_or_outside_click_closes() {
    let mut app = crate::app::SnapApp::default();
    let context = egui::Context::default();
    let text = "Example Patient\nID: PATIENT-123\nSeries acquisition\nCT\nDate: 20260906\nAxial: 2/3\nSpacing: 0.50 x 1.50 x 2.00 mm\nDims: 4x2x3\nW:400 C:60\nZoom: 100%\nCursor value: 260\nPointer value: 40\nOrientation: left R, right L, top A, bottom P";
    let first = details_frame(&mut app, &context, text, Vec::new());
    let button = first
        .shapes
        .iter()
        .find_map(|shape| match &shape.shape {
            egui::Shape::Text(label) if label.galley.text() == "Details" => {
                Some(Rect::from_min_size(label.pos, label.galley.size()).center())
            }
            _ => None,
        })
        .expect("visible Details control");
    let contains_metadata = |output: &egui::FullOutput| {
        output.shapes.iter().any(
            |shape| matches!(&shape.shape, egui::Shape::Text(label) if label.galley.text() == text),
        )
    };
    assert!(!contains_metadata(&first));
    for close_with_escape in [true, false] {
        details_frame(
            &mut app,
            &context,
            text,
            vec![
                egui::Event::PointerMoved(button),
                egui::Event::PointerButton {
                    pos: button,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        details_frame(
            &mut app,
            &context,
            text,
            vec![egui::Event::PointerButton {
                pos: button,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            }],
        );
        let opened = details_frame(&mut app, &context, text, Vec::new());
        assert!(
            contains_metadata(&opened),
            "click must render all metadata in popup"
        );
        if close_with_escape {
            details_frame(
                &mut app,
                &context,
                text,
                vec![egui::Event::Key {
                    key: egui::Key::Escape,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: egui::Modifiers::NONE,
                }],
            );
        } else {
            let outside = Pos2::new(700.0, 500.0);
            for pressed in [true, false] {
                details_frame(
                    &mut app,
                    &context,
                    text,
                    vec![
                        egui::Event::PointerMoved(outside),
                        egui::Event::PointerButton {
                            pos: outside,
                            button: egui::PointerButton::Primary,
                            pressed,
                            modifiers: egui::Modifiers::NONE,
                        },
                    ],
                );
            }
        }
        let closed = details_frame(&mut app, &context, text, Vec::new());
        assert!(
            !contains_metadata(&closed),
            "dismissed popup must stop rendering metadata"
        );
    }
}

#[test]
fn zoom_clipped_overlay_discloses_details_inside_visible_image() {
    use crate::dicom::loader::{load_volume_from_path, tests::fixtures};
    let root = tempfile::tempdir().expect("study root");
    fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID).expect("write study");
    let volume = load_volume_from_path(root.path()).expect("load study");
    let context = egui::Context::default();
    let visible = Rect::from_min_size(Pos2::new(20.0, 20.0), egui::vec2(176.0, 88.0));
    let image = Rect::from_min_size(Pos2::new(-400.0, -400.0), egui::vec2(1200.0, 1200.0));
    let output = context.run(egui::RawInput::default(), |context| {
        egui::CentralPanel::default().show(context, |ui| {
            ui.set_clip_rect(visible);
            let details = OverlayRenderer::draw(
                ui.painter(),
                image,
                &volume,
                OverlayContext {
                    axis: 0,
                    slice_index: 1,
                    wl: WindowLevel::new(60.0, 400.0),
                    zoom: 4.0,
                    cursor_value: Some(260.0),
                    pointer_intensity: 40.0,
                    pointer_suv: None,
                    cursor_suv: None,
                    view_transform: ViewTransform::default(),
                },
            )
            .expect("visible region cannot fit full metadata");
            assert!(details.contains("Zoom: 400%"));
            OverlayRenderer::show_details(ui, image, &details);
        });
    });
    let labels: Vec<_> = output
        .shapes
        .iter()
        .filter_map(|shape| match &shape.shape {
            egui::Shape::Text(label) => Some(label),
            _ => None,
        })
        .collect();
    assert_eq!(labels.len(), 1);
    assert_eq!(labels[0].galley.text(), "Details");
    assert!(visible.contains_rect(Rect::from_min_size(labels[0].pos, labels[0].galley.size())));
}

#[test]
fn keyboard_details_scroll_reaches_boundaries_without_changing_study_slice() {
    use crate::dicom::loader::{load_volume_from_path, tests::fixtures};
    let root = tempfile::tempdir().expect("study root");
    fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID).expect("write study");
    let mut app = crate::app::SnapApp::default();
    app.load_volume(
        load_volume_from_path(root.path()).expect("load study"),
        "Loaded".to_owned(),
    );
    app.viewer_state.slice_index = 1;
    let context = egui::Context::default();
    let text = format!(
        "First metadata line\n{}Last metadata line",
        "Metadata field: complete content\n".repeat(80)
    );
    let key = |key| egui::Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    };
    details_frame(&mut app, &context, &text, Vec::new());
    details_frame(&mut app, &context, &text, vec![key(egui::Key::Tab)]);
    details_frame(&mut app, &context, &text, vec![key(egui::Key::Enter)]);
    let opened = details_frame(&mut app, &context, &text, Vec::new());
    let metadata_bounds = |output: &egui::FullOutput| {
        output.shapes.iter().find_map(|shape| match &shape.shape {
            egui::Shape::Text(label) if label.galley.text() == text => Some((
                Rect::from_min_size(label.pos, label.galley.size()),
                shape.clip_rect,
            )),
            _ => None,
        })
    };
    let (first, _) = metadata_bounds(&opened).expect("keyboard activation opens metadata");
    details_frame(&mut app, &context, &text, vec![key(egui::Key::End)]);
    let end_frame = details_frame(&mut app, &context, &text, Vec::new());
    let (last, clip) = metadata_bounds(&end_frame).expect("metadata remains open");
    assert!(last.top() < first.top(), "End must scroll content");
    assert!(
        last.bottom() <= clip.bottom(),
        "final metadata line must be reachable"
    );
    assert_eq!(app.viewer_state.slice_index, 1);
    details_frame(&mut app, &context, &text, vec![key(egui::Key::Home)]);
    let home_frame = details_frame(&mut app, &context, &text, Vec::new());
    let (home, _) = metadata_bounds(&home_frame).expect("metadata remains open");
    assert_eq!(home.top(), first.top());
    assert_eq!(app.viewer_state.slice_index, 1);
    details_frame(&mut app, &context, &text, vec![key(egui::Key::PageDown)]);
    assert_eq!(app.viewer_state.slice_index, 1);
    details_frame(&mut app, &context, &text, vec![key(egui::Key::Escape)]);
    let closed = details_frame(&mut app, &context, &text, Vec::new());
    assert_eq!(metadata_bounds(&closed), None);
    assert_eq!(app.viewer_state.slice_index, 1);
}

#[test]
fn omitted_details_owner_closes_popup_and_restores_viewer_navigation() {
    use crate::dicom::loader::{load_volume_from_path, tests::fixtures};

    #[derive(Clone, Copy)]
    enum Layout {
        Narrow,
        Fits,
        Hidden,
        Removed,
    }
    fn frame(
        app: &mut crate::app::SnapApp,
        context: &egui::Context,
        volume: &LoadedVolume,
        layout: Layout,
        events: Vec<egui::Event>,
    ) -> egui::FullOutput {
        context.run(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, egui::vec2(800.0, 600.0))),
                events,
                ..egui::RawInput::default()
            },
            |context| {
                app.consume_global_shortcuts(context);
                if matches!(layout, Layout::Removed) {
                    return;
                }
                egui::CentralPanel::default().show(context, |ui| {
                    if matches!(layout, Layout::Hidden) {
                        return;
                    }
                    let size = if matches!(layout, Layout::Fits) {
                        egui::vec2(352.0, 264.0)
                    } else {
                        egui::vec2(176.0, 88.0)
                    };
                    let rect = Rect::from_min_size(Pos2::new(20.0, 20.0), size);
                    let details = OverlayRenderer::draw(
                        ui.painter(),
                        rect,
                        volume,
                        OverlayContext {
                            axis: 0,
                            slice_index: 1,
                            wl: WindowLevel::new(60.0, 400.0),
                            zoom: 1.0,
                            cursor_value: Some(260.0),
                            pointer_intensity: 40.0,
                            pointer_suv: None,
                            cursor_suv: None,
                            view_transform: ViewTransform::default(),
                        },
                    );
                    match layout {
                        Layout::Narrow => OverlayRenderer::show_details(
                            ui,
                            rect,
                            &details.expect("narrow viewport overflows"),
                        ),
                        Layout::Fits => assert_eq!(details, None),
                        Layout::Hidden | Layout::Removed => {
                            unreachable!("excluded before rendering")
                        }
                    }
                });
            },
        )
    }
    let root = tempfile::tempdir().expect("study root");
    fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID).expect("write study");
    let volume = load_volume_from_path(root.path()).expect("load study");
    let key = |key| egui::Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    };
    for absent in [Layout::Fits, Layout::Hidden, Layout::Removed] {
        let context = egui::Context::default();
        let mut app = crate::app::SnapApp::default();
        app.load_volume(volume.clone(), "Loaded".to_owned());
        app.viewer_state.slice_index = 1;
        frame(&mut app, &context, &volume, Layout::Narrow, Vec::new());
        frame(
            &mut app,
            &context,
            &volume,
            Layout::Narrow,
            vec![key(egui::Key::Tab)],
        );
        frame(
            &mut app,
            &context,
            &volume,
            Layout::Narrow,
            vec![key(egui::Key::Enter)],
        );
        assert!(context.memory(egui::Memory::any_popup_open));
        frame(&mut app, &context, &volume, absent, Vec::new());
        assert!(!context.memory(egui::Memory::any_popup_open));
        frame(
            &mut app,
            &context,
            &volume,
            absent,
            vec![key(egui::Key::ArrowDown)],
        );
        assert_eq!(
            app.viewer_state.slice_index, 2,
            "dismissed owner releases navigation"
        );

        // Reopen metadata, then let another UI popup take ownership. Omission
        // must not close a popup belonging to another component.
        frame(&mut app, &context, &volume, Layout::Narrow, Vec::new());
        frame(
            &mut app,
            &context,
            &volume,
            Layout::Narrow,
            vec![key(egui::Key::Tab)],
        );
        frame(
            &mut app,
            &context,
            &volume,
            Layout::Narrow,
            vec![key(egui::Key::Enter)],
        );
        let unrelated = egui::Id::new("unrelated-popup");
        context.memory_mut(|memory| memory.open_popup(unrelated));
        frame(&mut app, &context, &volume, absent, Vec::new());
        assert!(context.memory(|memory| memory.is_popup_open(unrelated)));
    }
}
