use super::super::layout::application_overlay;
use super::support::session;
use metis_platform::typeface::GlyphWeight;
use metis_platform::{Color, Framebuffer, Rect};
use metis_ui_lang::DisplayCommand;

#[test]
fn orthogonal_overlay_uses_embedded_typeface_and_composites_exact_rgba() {
    let (session, _root) = session();
    assert_eq!(
        (session.framebuffer.width(), session.framebuffer.height()),
        (1_280, 800)
    );

    let overlay = application_overlay(&session.views, &session.viewports, false, 30.0)
        .expect("compose the RITK overlay commands");
    assert_eq!(overlay.commands.len(), 12);

    let expected_fills = session
        .viewports
        .iter()
        .flat_map(|viewport| {
            let panel = viewport.panel_rect().expect("bounded native panel");
            [
                Rect::new(panel.x, panel.y, panel.width, 20),
                Rect::new(panel.x, panel.y + panel.height - 20, panel.width, 20),
            ]
        })
        .collect::<Vec<_>>();
    assert_eq!(expected_fills.len(), 6);
    let mut fill_index = 0;
    for (view_index, view) in session.views.iter().enumerate() {
        let command_index = view_index * 4;
        for (command, expected_rect) in overlay.commands[command_index..command_index + 2]
            .iter()
            .zip(&expected_fills[fill_index..fill_index + 2])
        {
            match command {
                DisplayCommand::FillRect { rect, color, .. } => {
                    assert_eq!(*rect, *expected_rect);
                    assert_eq!(*color, Color::rgba(0, 0, 0, 224));
                }
                other => panic!("expected a panel bar fill, got {other:?}"),
            }
        }
        fill_index += 2;

        let expected_title = format!("METIS  RITK-SNAP  {}", view.plane_name);
        let expected_footer = format!(
            "Slice {}/{}  {}x{}  W:{:.0} C:{:.0}",
            view.slice_index.saturating_add(1),
            view.slice_count,
            view.frame.width(),
            view.frame.height(),
            view.window_level.width,
            view.window_level.center
        );
        for (command, expected_text, expected_size) in [
            (&overlay.commands[command_index + 2], expected_title, 16_u32),
            (
                &overlay.commands[command_index + 3],
                expected_footer,
                13_u32,
            ),
        ] {
            match command {
                DisplayCommand::DrawText { text, style, .. } => {
                    assert_eq!(text.as_ref(), expected_text.as_str());
                    assert_eq!(style.color, Color::rgba(255, 255, 160, 255));
                    assert_eq!(style.size.pixels(), f64::from(expected_size));
                    assert_eq!(style.weight, GlyphWeight::Regular);
                }
                other => panic!("expected a styled text command, got {other:?}"),
            }
        }
    }
    assert_eq!(fill_index, expected_fills.len());

    let mut framebuffer = Framebuffer::new(1_280, 800).expect("bounded test framebuffer");
    framebuffer.clear(Color::rgb(37, 53, 71));
    overlay.render_to(&mut framebuffer);
    // Rounded source-over gives (37,53,71) * (255-224) / 255 = (4,6,9).
    let first_viewport = session
        .viewports
        .first()
        .expect("three orthogonal viewport slots");
    let first_panel = first_viewport.panel_rect().expect("bounded native panel");
    let sample_x = first_panel.x + 1;
    let sample_y = first_panel.y + 1;
    assert_eq!(
        framebuffer.get_pixel(sample_x, sample_y),
        Color::rgb(4, 6, 9)
    );
}
