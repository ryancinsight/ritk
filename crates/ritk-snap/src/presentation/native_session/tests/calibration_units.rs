use super::calibration_scope::{calibration_clicks, key};
use super::measurement_overlay::{composed, draw_texts};
use super::pointer_capture::{down, up};
use super::support::session;
use super::*;
use crate::tools::interaction::ImagePoint;
use crate::tools::kind::ToolKind;
use metis_platform::native::MouseButton;

#[test]
fn reference_unit_selection_preserves_annotations_and_reset() -> anyhow::Result<()> {
    for (entered, tabs, button_x) in [
        ("25.4", 0, 53),
        ("1", 1, 141),
        ("76.2", 2, 231),
        ("76.2", -1_i32, 231),
    ] {
        for pointer_selection in [false, true] {
            let (mut session, _root) = session();
            session.app.axis = 0;
            session
                .app
                .on_drag_start(ToolKind::MeasureLength, Some(ImagePoint::new(0.0, 0.0)));
            session.app.on_drag_end(Some(ImagePoint::new(3.0, 4.0)));
            let raw = session.app.annotations.clone();
            let original = draw_texts(&composed(&session, None));
            let mut events = calibration_clicks(&session).to_vec();
            if pointer_selection {
                // Unit buttons share the centred 286 by 124 dialog. These
                // independently specified interior points exercise hit routing.
                let x =
                    i32::try_from(session.framebuffer.width().saturating_sub(286) / 2)? + button_x;
                let y = i32::try_from(session.framebuffer.height().saturating_sub(124) / 2)? + 95;
                events.extend([down(x, y, MouseButton::Left), up(x, y, MouseButton::Left)]);
            } else {
                events.extend((0..tabs.unsigned_abs()).map(|_| WindowEvent::KeyDown {
                    virtual_key: 0x09,
                    repeated: false,
                    modifiers: if tabs < 0 {
                        ModifierState::SHIFT
                    } else {
                        ModifierState::NONE
                    },
                }));
            }
            events.extend(
                entered
                    .chars()
                    .map(|character| WindowEvent::TextInput { character }),
            );
            events.push(key(0x0d));
            session.handle_events(&events)?;
            assert_eq!(draw_texts(&composed(&session, None)), ["25.4 mm"]);
            assert_eq!(session.app.annotations, raw);
            session.handle_events(&calibration_clicks(&session))?;
            assert_eq!(draw_texts(&composed(&session, None)), original);
            assert_eq!(session.app.annotations, raw);
        }
    }
    Ok(())
}

#[test]
fn unit_dialog_rejects_overflow_and_cancels_without_calibration() -> anyhow::Result<()> {
    for (entered, initial_tabs, final_tabs) in
        [("1e38", 1, 0), ("111111111111111111111111111111111", 0, 1)]
    {
        for pointer_selection in [false, true] {
            let (mut session, _root) = session();
            session.app.axis = 0;
            session
                .app
                .on_drag_start(ToolKind::MeasureLength, Some(ImagePoint::new(0.0, 0.0)));
            session.app.on_drag_end(Some(ImagePoint::new(3.0, 4.0)));
            let raw = session.app.annotations.clone();
            let original = draw_texts(&composed(&session, None));
            let mut events = calibration_clicks(&session).to_vec();
            events.extend((0..initial_tabs).map(|_| key(0x09)));
            // The first input converts outside f32; the second exceeds the editor budget.
            events.extend(
                entered
                    .chars()
                    .map(|character| WindowEvent::TextInput { character }),
            );
            if pointer_selection {
                let x = i32::try_from(session.framebuffer.width().saturating_sub(286) / 2)? + 141;
                let y = i32::try_from(session.framebuffer.height().saturating_sub(124) / 2)? + 95;
                events.extend([down(x, y, MouseButton::Left), up(x, y, MouseButton::Left)]);
            } else {
                events.extend((0..final_tabs).map(|_| key(0x09)));
            }
            events.push(key(0x0d));
            session.handle_events(&events)?;
            assert!(!session.app.has_length_calibration());
            assert_eq!(draw_texts(&composed(&session, None)), original);
            session.handle_events(&[key(0x1b)])?;
            assert_eq!(session.app.annotations, raw);
            assert_eq!(draw_texts(&composed(&session, None)), original);
        }
    }
    Ok(())
}
