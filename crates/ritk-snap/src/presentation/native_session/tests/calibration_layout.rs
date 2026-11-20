use super::calibration_scope::calibration_clicks;
use super::support::session;
use super::*;
use crate::presentation::native_session::layout::{draw_text, text_style};
use crate::tools::interaction::ImagePoint;
use crate::tools::kind::ToolKind;
use metis_platform::rasterizer::{fill_rect, CornerRadius};
use metis_platform::{Color, Framebuffer, Rect};

const PANEL_COLOR: Color = Color::rgb(35, 40, 47);
const FIELD_COLOR: Color = Color::rgb(56, 63, 72);
const TEXT_COLOR: Color = Color::rgb(221, 228, 235);
const MAX_INPUT: &str = "00000000000000000000000000000007";

// The embedded regular face declares 1,240 line-height units and 1,000
// units/em. At 13 pixels the containing line box is ceil(16.12) = 17 pixels.
const LINE_BOX_HEIGHT: i32 = (1_240 * 13 + 999) / 1_000;
// Zero and seven advance 648 and 510 units in that face. Thirty-one zeros
// followed by seven occupy 267.774 pixels, so the 246-pixel input lane
// end-scrolls by ceil(21.774) = 22 pixels and exposes a distinct tail glyph.
const MAX_INPUT_OVERFLOW: i32 = ((31 * 648 + 510) * 13 - 246 * 1_000 + 999) / 1_000;

fn dialog_geometry(framebuffer: &Framebuffer) -> anyhow::Result<(Rect, Rect, Rect)> {
    let width = i32::try_from(framebuffer.width())?;
    let height = i32::try_from(framebuffer.height())?;
    let x = (width - 286).div_euclid(2).max(0);
    let y = (height - 86).div_euclid(2).max(0);
    let panel = Rect::new(x, y, 286, 86);
    let caption = Rect::new(x + 12, y + 8, 262, 26);
    let field = Rect::new(x + 12, y + 38, 262, 32);
    Ok((panel, caption, field))
}

fn render_reference_dialog(
    framebuffer: &mut Framebuffer,
    input: &str,
    scroll: i32,
) -> anyhow::Result<()> {
    let (panel, caption, _) = dialog_geometry(framebuffer)?;
    fill_rect(framebuffer, panel, CornerRadius::SQUARE, PANEL_COLOR);
    let text = text_style(TEXT_COLOR, 13)?;
    let caption_y = caption.y + (caption.height - LINE_BOX_HEIGHT).div_euclid(2);
    draw_text(
        framebuffer,
        caption.x,
        caption_y,
        "Known distance (mm) · Enter to apply · Esc",
        text,
    );
    render_reference_input(framebuffer, input, scroll)
}

fn render_reference_input(
    framebuffer: &mut Framebuffer,
    input: &str,
    scroll: i32,
) -> anyhow::Result<()> {
    let (_, _, field) = dialog_geometry(framebuffer)?;
    let text = text_style(TEXT_COLOR, 13)?;
    fill_rect(framebuffer, field, CornerRadius::SQUARE, FIELD_COLOR);
    let input_bounds = Rect::new(field.x + 8, field.y, field.width - 16, field.height);
    let input_y = input_bounds.y + (input_bounds.height - LINE_BOX_HEIGHT).div_euclid(2);
    framebuffer.render_clipped(input_bounds, |surface| {
        draw_text(surface, input_bounds.x - scroll, input_y, input, text);
    });
    Ok(())
}

fn type_text(session: &mut NativeViewerSession, input: &str) -> anyhow::Result<()> {
    let events = input
        .chars()
        .map(|character| WindowEvent::TextInput { character })
        .collect::<Vec<_>>();
    session.handle_events(&events)?;
    Ok(())
}

fn prepare_reference(session: &mut NativeViewerSession) -> anyhow::Result<()> {
    session.app.axis = 0;
    session
        .app
        .on_drag_start(ToolKind::MeasureLength, Some(ImagePoint::new(0.0, 0.0)));
    session.app.on_drag_end(Some(ImagePoint::new(3.0, 4.0)));
    session.refresh_frame()?;
    Ok(())
}

fn open_dialog(session: &mut NativeViewerSession) -> anyhow::Result<()> {
    session.handle_events(&calibration_clicks(session))?;
    Ok(())
}

fn backspace() -> WindowEvent {
    WindowEvent::KeyDown {
        virtual_key: 0x08,
        repeated: false,
        modifiers: ModifierState::NONE,
    }
}

fn assert_frame_eq(actual: &Framebuffer, expected: &Framebuffer) {
    assert_eq!(
        (actual.width(), actual.height()),
        (expected.width(), expected.height())
    );
    let width = usize::try_from(actual.width()).expect("framebuffer width fits usize");
    let mut mismatch_count = 0_usize;
    let mut first_mismatch = None;
    for (index, (actual, expected)) in actual.pixels().iter().zip(expected.pixels()).enumerate() {
        if actual != expected {
            mismatch_count += 1;
            first_mismatch.get_or_insert((index % width, index / width, *actual, *expected));
        }
    }
    assert_eq!(
        mismatch_count, 0,
        "{mismatch_count} pixels differ; first mismatch is {first_mismatch:?}"
    );
}

#[test]
fn calibration_dialog_line_boxes_stay_in_controls() -> anyhow::Result<()> {
    let (mut short, _root) = session();
    prepare_reference(&mut short)?;
    assert_eq!(
        (short.framebuffer.width(), short.framebuffer.height()),
        (1_280, 800)
    );
    let (_, _, field) = dialog_geometry(&short.framebuffer)?;
    assert_eq!(
        (field.x, field.y, field.width, field.height),
        (509, 395, 262, 32)
    );
    let mut empty_reference = short.framebuffer.clone();
    render_reference_dialog(&mut empty_reference, "", 0)?;
    open_dialog(&mut short)?;
    assert_frame_eq(&short.framebuffer, &empty_reference);

    let mut short_reference = empty_reference;
    render_reference_dialog(&mut short_reference, "100", 0)?;
    type_text(&mut short, "100")?;
    assert_frame_eq(&short.framebuffer, &short_reference);
    Ok(())
}

#[test]
fn calibration_dialog_input_end_scrolls_and_backspace_resets() -> anyhow::Result<()> {
    let (mut long, _root) = session();
    prepare_reference(&mut long)?;
    assert_eq!(MAX_INPUT.len(), 32);
    assert_eq!(MAX_INPUT_OVERFLOW, 22);
    open_dialog(&mut long)?;
    let mut empty_reference = long.framebuffer.clone();
    let mut long_reference = empty_reference.clone();
    render_reference_input(&mut long_reference, MAX_INPUT, MAX_INPUT_OVERFLOW)?;
    type_text(&mut long, MAX_INPUT)?;
    assert_frame_eq(&long.framebuffer, &long_reference);

    render_reference_input(&mut empty_reference, "000", 0)?;
    let backspaces = (0..MAX_INPUT.len() - 3)
        .map(|_| backspace())
        .collect::<Vec<_>>();
    long.handle_events(&backspaces)?;
    assert_frame_eq(&long.framebuffer, &empty_reference);
    Ok(())
}
