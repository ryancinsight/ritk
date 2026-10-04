use super::*;
use crate::ui::ViewTransform;
use metis_platform::{Framebuffer, Rect};

fn projection() -> ImageToPanel {
    ImageToPanel {
        transform: ViewTransform::default(),
        source_size: [100, 100],
        output_size: [100.0, 100.0],
        origin: [30.0, 0.0],
        texel: [1.4, 1.4],
        clip: PanelBounds {
            x: 0,
            y: 0,
            width: 200,
            height: 140,
        }
        .clip()
        .expect("non-empty panel"),
    }
}

#[test]
fn completed_measurements_render_geometry_and_actual_values() {
    let projection = projection();
    let annotations = [
        Annotation::Length {
            p1: [20.0, 20.0],
            p2: [60.0, 50.0],
            length_mm: 50.0,
        },
        Annotation::Angle {
            p1: [85.0, 15.0],
            p2: [65.0, 15.0],
            p3: [65.0, 35.0],
            angle_deg: 90.0,
        },
    ];
    let overlay =
        overlay_for(&annotations, &ToolState::Idle, &projection).expect("measurement display list");
    assert!(overlay.commands.iter().any(|command| matches!(
        command,
        DisplayCommand::DrawText { text, .. } if text == "50.0 mm"
    )));
    assert!(overlay.commands.iter().any(|command| matches!(
        command,
        DisplayCommand::DrawText { text, .. } if text == "90.0°"
    )));

    let mut framebuffer = Framebuffer::new(200, 140).expect("measurement framebuffer");
    framebuffer.clear(Color::BLACK);
    overlay.render_to(&mut framebuffer);
    let length_midpoint = projection
        .project([40.0, 35.0])
        .expect("length midpoint projects");
    assert_eq!(
        framebuffer.get_pixel(
            screen_coordinate(length_midpoint[0], "test x").expect("test x"),
            screen_coordinate(length_midpoint[1], "test y").expect("test y")
        ),
        MEASUREMENT_COLOR,
        "the 3-4-5 length must paint its projected segment"
    );
    assert!(
        overlay
            .commands
            .iter()
            .filter_map(text_extent)
            .all(|extent| { pixels_in(&framebuffer, extent).any(|pixel| pixel != Color::BLACK) }),
        "each measured value must change pixels inside its label extent"
    );

    let mut changed_value = annotations.clone();
    let Annotation::Length { length_mm, .. } = &mut changed_value[0] else {
        panic!("invariant: first fixture annotation is a length");
    };
    *length_mm = 51.0;
    let changed_overlay = overlay_for(&changed_value, &ToolState::Idle, &projection)
        .expect("changed measurement display list");
    let mut changed_framebuffer =
        Framebuffer::new(200, 140).expect("changed measurement framebuffer");
    changed_framebuffer.clear(Color::BLACK);
    changed_overlay.render_to(&mut changed_framebuffer);
    assert_ne!(
        framebuffer.pixels(),
        changed_framebuffer.pixels(),
        "changing the stored physical value must change visible label pixels"
    );
}

#[test]
fn active_measurements_render_their_committed_geometry() {
    let projection = projection();
    let first = ImagePoint::new(20.0, 20.0);
    let vertex = ImagePoint::new(60.0, 20.0);
    let overlay = overlay_for(
        &[],
        &ToolState::MeasureAngle2 {
            p1: first,
            p2: vertex,
        },
        &projection,
    )
    .expect("active angle display list");
    let mut framebuffer = Framebuffer::new(200, 140).expect("active measurement framebuffer");
    framebuffer.clear(Color::BLACK);
    overlay.render_to(&mut framebuffer);
    let midpoint = projection
        .project([20.0, 40.0])
        .expect("active ray midpoint projects");
    assert_eq!(
        framebuffer.get_pixel(
            screen_coordinate(midpoint[0], "test x").expect("test x"),
            screen_coordinate(midpoint[1], "test y").expect("test y")
        ),
        MEASUREMENT_COLOR,
        "the active angle's first ray must be visible"
    );

    let anchor_overlay = overlay_for(&[], &ToolState::MeasureLength1 { p1: first }, &projection)
        .expect("active length display list");
    let mut anchor_framebuffer = Framebuffer::new(200, 140).expect("active anchor framebuffer");
    anchor_framebuffer.clear(Color::BLACK);
    anchor_overlay.render_to(&mut anchor_framebuffer);
    let anchor = projection
        .project([first.y(), first.x()])
        .expect("active anchor projects");
    assert_eq!(
        anchor_framebuffer.get_pixel(
            screen_coordinate(anchor[0], "test x").expect("test x"),
            screen_coordinate(anchor[1], "test y").expect("test y")
        ),
        MEASUREMENT_COLOR,
        "the active length anchor must be visible"
    );
}

fn text_extent(command: &DisplayCommand) -> Option<Rect> {
    match command {
        DisplayCommand::DrawText { text, x, y, style } => style.extent(*x, *y, text),
        _ => None,
    }
}

fn pixels_in(framebuffer: &Framebuffer, rect: Rect) -> impl Iterator<Item = Color> + '_ {
    let left = rect.x.max(0);
    let top = rect.y.max(0);
    let right = rect
        .x
        .saturating_add(rect.width)
        .min(i32::try_from(framebuffer.width()).expect("test width fits i32"));
    let bottom = rect
        .y
        .saturating_add(rect.height)
        .min(i32::try_from(framebuffer.height()).expect("test height fits i32"));
    (top..bottom).flat_map(move |y| (left..right).map(move |x| framebuffer.get_pixel(x, y)))
}
