//! Physical aspect ratios observed in actual textured viewport shapes.

use super::*;
use crate::dicom::loader::{load_volume_from_path, tests::fixtures};
use crate::ui::{RotationSteps, ViewTransform};

fn study_app() -> SnapApp {
    let root = tempfile::tempdir().expect("study root");
    fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID).expect("write Part 10 study");
    let volume = load_volume_from_path(root.path()).expect("decode anisotropic study");
    assert_eq!(volume.spacing, [2.0, 1.5, 0.5]);
    let mut app = SnapApp::default();
    app.loaded_secondary = Some(volume.clone());
    app.load_volume(volume, "Loaded".to_owned());
    app.show_overlay = false;
    app.show_crosshair = false;
    app
}

#[test]
fn dicom_window_metadata_drives_initial_viewer_state() {
    let root = tempfile::tempdir().expect("window metadata fixture root");
    let (filename, _) =
        fixtures::write_grayscale_presentation(root.path(), "MONOCHROME2", Some("LINEAR"))
            .expect("write window metadata fixture");
    let volume = load_volume_from_path(root.path().join(filename).as_path())
        .expect("decode window metadata fixture");
    let mut app = SnapApp::default();
    app.load_volume(volume, "Loaded".to_owned());
    assert_eq!(app.viewer_state.window_center, Some(0.0));
    assert_eq!(app.viewer_state.window_width, Some(40.0));
}

fn image_bounds(output: &egui::FullOutput, id: egui::TextureId) -> egui::Rect {
    output
        .shapes
        .iter()
        .find_map(|shape| match &shape.shape {
            egui::Shape::Mesh(mesh) if mesh.texture_id == id => Some(mesh.calc_bounds()),
            egui::Shape::Rect(rect) if rect.fill_texture_id == id => Some(rect.rect),
            _ => None,
        })
        .expect("actual viewport image shape")
}

fn assert_physical_ratio(bounds: egui::Rect, axis: usize, rotation: RotationSteps) {
    // Fixture physical extents derive independently from its 3x2x4 voxels and
    // [2, 1.5, 0.5] mm sampling, in [width, height] order.
    let [width, height]: [f32; 2] = match axis {
        0 => [2.0, 3.0],
        1 => [2.0, 6.0],
        _ => [3.0, 6.0],
    };
    let [width, height] = match rotation {
        RotationSteps::Ninety | RotationSteps::TwoSeventy => [height, width],
        RotationSteps::Zero | RotationSteps::OneEighty => [width, height],
    };
    // Cross multiplication avoids dividing by a rounded displayed height.
    // Eight f32 rounding steps bound fit division, extent multiplication,
    // endpoint translation/subtraction, and the two oracle products.
    let tolerance = 8.0 * f32::EPSILON * bounds.max.to_vec2().max_elem() * width.max(height);
    let residual = bounds.width() * height - bounds.height() * width;
    assert!(residual.abs() <= tolerance,
        "axis {axis} rotation {rotation:?}: rendered {bounds:?}, expected physical {width}x{height}, residual {residual}, bound {tolerance}");
}

#[test]
fn primary_physical_aspect_survives_all_layouts_and_texture_transforms() {
    let mut app = study_app();
    let context = egui::Context::default();
    for rotation in [
        RotationSteps::Zero,
        RotationSteps::Ninety,
        RotationSteps::OneEighty,
        RotationSteps::TwoSeventy,
    ] {
        for layout in 0..4 {
            app.multi_planar = layout == 1;
            app.dual_plane = layout == 2;
            app.compare_side_by_side = layout == 3;
            for (flip_h, flip_v) in [(false, false), (true, false), (false, true), (true, true)] {
                app.view_transform = ViewTransform {
                    flip_h,
                    flip_v,
                    rotation,
                };
                app.mark_all_textures_dirty();
                for axis in 0..3 {
                    let output = context.run(
                        egui::RawInput {
                            screen_rect: Some(egui::Rect::from_min_size(
                                egui::Pos2::ZERO,
                                egui::vec2(640.0, 480.0),
                            )),
                            ..egui::RawInput::default()
                        },
                        |context| {
                            egui::CentralPanel::default()
                                .show(context, |ui| app.render_axis_viewport(ui, context, axis));
                        },
                    );
                    let texture = match axis {
                        0 => &app.texture,
                        1 => &app.coronal_tex,
                        _ => &app.sagittal_tex,
                    }
                    .as_ref()
                    .expect("rendered texture");
                    assert_physical_ratio(image_bounds(&output, texture.id()), axis, rotation);
                }
            }
        }
    }
}

#[test]
fn comparison_physical_aspect_uses_the_output_sampling_grid() {
    let mut app = study_app();
    app.compare_side_by_side = true;
    let context = egui::Context::default();
    for fused in [false, true] {
        app.compare_fused_overlay = fused;
        for rotation in [
            RotationSteps::Zero,
            RotationSteps::Ninety,
            RotationSteps::OneEighty,
            RotationSteps::TwoSeventy,
        ] {
            app.view_transform = ViewTransform {
                flip_h: true,
                flip_v: true,
                rotation,
            };
            for primary_axis in 0..3 {
                for secondary_axis in 0..3 {
                    app.secondary_texture_dirty = true;
                    let output = context.run(
                        egui::RawInput {
                            screen_rect: Some(egui::Rect::from_min_size(
                                egui::Pos2::ZERO,
                                egui::vec2(640.0, 480.0),
                            )),
                            ..egui::RawInput::default()
                        },
                        |context| {
                            egui::CentralPanel::default().show(context, |ui| {
                                app.render_secondary_compare_viewport(
                                    ui,
                                    context,
                                    primary_axis,
                                    secondary_axis,
                                )
                            });
                        },
                    );
                    if fused && primary_axis != secondary_axis {
                        assert!(
                            app.secondary_texture.is_none(),
                            "non-parallel fused planes must not retain a stale texture"
                        );
                        assert!(
                            app.status_message.contains("not parallel"),
                            "non-parallel fused planes must report the physical failure"
                        );
                        continue;
                    }
                    let texture = app.secondary_texture.as_ref().expect("comparison texture");
                    assert_physical_ratio(
                        image_bounds(&output, texture.id()),
                        if fused { primary_axis } else { secondary_axis },
                        rotation,
                    );
                }
            }
        }
    }
}

#[test]
fn common_distance_scale_cannot_change_display_aspect() {
    let mut app = study_app();
    let context = egui::Context::default();
    for exponent in [-1000, -30, 0, 1000] {
        // A common binary unit change is exact in the f64 geometry. The outer
        // exponents deliberately exceed f32's range; only ratios reach egui.
        app.loaded.as_mut().expect("volume").spacing =
            fixtures::SPACING.map(|spacing| spacing * 2.0_f64.powi(exponent));
        for axis in 0..3 {
            let output = context.run(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(640.0, 480.0),
                    )),
                    ..egui::RawInput::default()
                },
                |context| {
                    egui::CentralPanel::default()
                        .show(context, |ui| app.render_axis_viewport(ui, context, axis));
                },
            );
            let texture = match axis {
                0 => &app.texture,
                1 => &app.coronal_tex,
                _ => &app.sagittal_tex,
            }
            .as_ref()
            .expect("texture");
            assert_physical_ratio(
                image_bounds(&output, texture.id()),
                axis,
                RotationSteps::Zero,
            );
        }
    }
}

#[test]
fn cursor_click_uses_the_physical_image_rectangle_in_every_layout() {
    let mut app = study_app();
    let context = egui::Context::default();
    for layout in 0..4 {
        app.multi_planar = layout == 1;
        app.dual_plane = layout == 2;
        app.compare_side_by_side = layout == 3;
        for axis in 0..3 {
            app.viewer_state.slice_index = 1;
            app.coronal_slice = 1;
            app.sagittal_slice = 2;
            let output = context.run(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(640.0, 480.0),
                    )),
                    ..egui::RawInput::default()
                },
                |context| {
                    egui::CentralPanel::default()
                        .show(context, |ui| app.render_axis_viewport(ui, context, axis));
                },
            );
            let texture = match axis {
                0 => &app.texture,
                1 => &app.coronal_tex,
                _ => &app.sagittal_tex,
            }
            .as_ref()
            .expect("texture");
            let bounds = image_bounds(&output, texture.id());
            let pointer = bounds.min + bounds.size() * egui::vec2(0.625, 0.75);
            for pressed in [true, false] {
                drop(context.run(
                    egui::RawInput {
                        screen_rect: Some(egui::Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(640.0, 480.0),
                        )),
                        events: vec![
                            egui::Event::PointerMoved(pointer),
                            egui::Event::PointerButton {
                                pos: pointer,
                                button: egui::PointerButton::Primary,
                                pressed,
                                modifiers: egui::Modifiers::NONE,
                            },
                        ],
                        ..egui::RawInput::default()
                    },
                    |context| {
                        egui::CentralPanel::default()
                            .show(context, |ui| app.render_axis_viewport(ui, context, axis));
                    },
                ));
            }
            let expected = if axis == 0 { [1, 1, 2] } else { [2, 1, 2] };
            assert_eq!(app.linked_cursor.expect("selected voxel").voxel(), expected);
            assert_eq!(
                app.current_cursor_value(),
                Some(if axis == 0 { 260.0 } else { 420.0 })
            );
        }
    }
}

#[test]
fn unrepresentable_display_geometry_reports_failure_before_painting_image() {
    let mut app = study_app();
    app.loaded.as_mut().expect("volume").spacing = [2.0, f64::MAX, f64::MIN_POSITIVE];
    let context = egui::Context::default();
    let output = context.run(egui::RawInput::default(), |context| {
        egui::CentralPanel::default().show(context, |ui| app.render_axis_viewport(ui, context, 0));
    });
    assert_eq!(app.status_message, "Image placement failed: physical slice geometry cannot be represented in viewport coordinates");
    assert!(output.shapes.iter().any(|shape| matches!(&shape.shape,
        egui::Shape::Text(text) if text.galley.text() == app.status_message)));
    let texture_id = app.texture.as_ref().expect("decoded texture").id();
    assert!(!output.shapes.iter().any(|shape| match &shape.shape {
        egui::Shape::Rect(rect) => rect.fill_texture_id == texture_id,
        egui::Shape::Mesh(mesh) => mesh.texture_id == texture_id,
        _ => false,
    }));
}

#[test]
fn translated_image_extent_that_rounds_to_zero_is_rejected_before_paint() {
    let mut app = study_app();
    app.loaded.as_mut().expect("primary").spacing = [2.0, 1.0, 1e-10];
    app.loaded_secondary.as_mut().expect("secondary").spacing = [2.0, 1.0, 1e-10];
    let context = egui::Context::default();
    for secondary in [false, true] {
        let output = context.run(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(640.0, 480.0),
                )),
                ..egui::RawInput::default()
            },
            |context| {
                egui::CentralPanel::default().show(context, |ui| {
                    if secondary {
                        app.render_secondary_compare_viewport(ui, context, 0, 0);
                    } else {
                        app.render_axis_viewport(ui, context, 0);
                    }
                });
            },
        );
        assert_eq!(
            app.status_message,
            "Image placement failed: physical slice rectangle collapses at screen coordinates"
        );
        let id = if secondary {
            &app.secondary_texture
        } else {
            &app.texture
        }
        .as_ref()
        .expect("decoded texture")
        .id();
        assert!(!output.shapes.iter().any(|shape| match &shape.shape {
            egui::Shape::Rect(rect) => rect.fill_texture_id == id,
            egui::Shape::Mesh(mesh) => mesh.texture_id == id,
            _ => false,
        }));
        assert!(output.shapes.iter().any(|shape| matches!(&shape.shape,
            egui::Shape::Text(text) if text.galley.text() == app.status_message)));
    }
}

#[test]
fn physical_images_keep_requested_zoom_extents_and_viewport_clip() {
    let mut app = study_app();
    let context = egui::Context::default();
    for secondary in [false, true] {
        for axis in 0..3 {
            for zoom in [crate::ui::MIN_ZOOM, 1.0, crate::ui::MAX_ZOOM] {
                app.zoom = zoom;
                let mut available = egui::Vec2::ZERO;
                let mut clip = egui::Rect::NOTHING;
                let output = context.run(
                    egui::RawInput {
                        screen_rect: Some(egui::Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(640.0, 480.0),
                        )),
                        ..egui::RawInput::default()
                    },
                    |context| {
                        egui::CentralPanel::default().show(context, |ui| {
                            available = ui.available_size();
                            clip = ui.clip_rect();
                            if secondary {
                                app.render_secondary_compare_viewport(ui, context, axis, axis);
                            } else {
                                app.render_axis_viewport(ui, context, axis);
                            }
                        });
                    },
                );
                let texture = if secondary {
                    &app.secondary_texture
                } else {
                    match axis {
                        0 => &app.texture,
                        1 => &app.coronal_tex,
                        _ => &app.sagittal_tex,
                    }
                }
                .as_ref()
                .expect("rendered texture");
                let bounds = image_bounds(&output, texture.id());
                assert_physical_ratio(bounds, axis, RotationSteps::Zero);
                // Physical fit touches at least one available edge before zoom.
                // Eight f32 rounding steps cover fitted extents, translated
                // endpoints, and the normalized extent divisions.
                let fitted_fraction = (bounds.size() / (available * zoom)).max_elem();
                assert!((fitted_fraction - 1.0).abs() <= 8.0 * f32::EPSILON);
                let image_clip = output
                    .shapes
                    .iter()
                    .find_map(|shape| {
                        let is_image = match &shape.shape {
                            egui::Shape::Rect(rect) => rect.fill_texture_id == texture.id(),
                            egui::Shape::Mesh(mesh) => mesh.texture_id == texture.id(),
                            _ => false,
                        };
                        is_image.then_some(shape.clip_rect)
                    })
                    .expect("image clip rectangle");
                assert_eq!(image_clip, clip);
            }
        }
    }
}
