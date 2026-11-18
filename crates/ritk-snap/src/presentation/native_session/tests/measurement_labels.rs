use super::super::layout::overlay_display::COLORBAR_WIDTH;
use super::super::slice_scrollbar::RIGHT_GUTTER;
use super::measurement_overlay::{axial_panel, composed};
use super::support::session;
use crate::presentation::ViewportPoint;
use crate::tools::interaction::{Annotation, AnnotationLocation};
use metis_ui_lang::DisplayCommand;

#[test]
fn measurement_labels_fit_before_the_colorbar() {
    let (mut viewer, _study) = session();
    let (pane, ..) = axial_panel(&viewer);
    let viewport = viewer.viewports[pane];
    let image = viewport.image;
    let vertex = viewport
        .mapping()
        .map(ViewportPoint::new(
            image.x + image.width - 1.0,
            image.y + image.height / 2.0,
        ))
        .expect("fixture endpoint is within the image");
    let location = Some(AnnotationLocation {
        plane: 0,
        slice: viewer.views[pane].slice_index,
    });
    let right = [vertex.y(), vertex.x()];
    let left = [vertex.y(), 0.0];
    viewer.app.annotations.extend([
        Annotation::Angle {
            location,
            p1: left,
            p2: right,
            p3: [0.0, vertex.x()],
            angle_deg: 90.0,
        },
        Annotation::Length {
            plane: 0,
            location,
            p1: [0.0, vertex.x()],
            p2: right,
            length_mm: 12.0,
        },
    ]);
    let commands = composed(&viewer, None);
    let panel = viewport.panel_rect().expect("pane bounds");
    let ramp_left = panel.x + panel.width - RIGHT_GUTTER - COLORBAR_WIDTH;
    let extents: Vec<_> = commands
        .iter()
        .filter_map(|command| match command {
            DisplayCommand::DrawText { text, x, y, style } => Some((
                text.as_ref(),
                style.extent(*x, *y, text).expect("measured text bounds"),
            )),
            _ => None,
        })
        .collect();
    assert_eq!(
        extents.iter().map(|(text, _)| *text).collect::<Vec<_>>(),
        ["90.0 deg", "12.0 mm"]
    );
    for (_, glyph) in extents {
        assert!(glyph.x >= panel.x);
        assert!(glyph.x + glyph.width <= ramp_left);
    }
}
