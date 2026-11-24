use super::super::layout::{draw_text, text_style};
use super::{WindowAction, WindowChrome, WindowChromeEvent};
use crate::app::{CalibrationInputError, CalibrationUnit, LengthCalibrationReference, SnapApp};
use crate::presentation::{PointerButton, PresentationEvent};
use anyhow::Result;
use metis_platform::rasterizer::{fill_rect, CornerRadius};
use metis_platform::{whole_pixels, Color, Framebuffer, Rect};
use metis_ui_lang::{DecimalEdit, DecimalEntry};

#[derive(Clone, Copy, PartialEq, Eq)]
enum InputState {
    Editing,
    Rejected,
    Full,
}

#[derive(Clone)]
pub(super) struct CalibrationDialog {
    panel: usize,
    reference: LengthCalibrationReference,
    input: DecimalEntry<32>,
    input_state: InputState,
    unit: CalibrationUnit,
}

impl WindowChrome {
    pub(super) fn open_calibration(&mut self, app: &SnapApp, panel: usize) -> WindowChromeEvent {
        self.open_menu = None;
        let reference = app.length_calibration_reference();
        if app.has_length_calibration() {
            return WindowChromeEvent {
                consumed: true,
                repaint: true,
                action: Some(WindowAction::ApplyLengthCalibration {
                    panel,
                    reference,
                    known_mm: None,
                }),
            };
        }
        self.cancel_pointer_capture();
        self.calibration = Some(CalibrationDialog {
            panel,
            reference,
            input: DecimalEntry::new(),
            input_state: InputState::Editing,
            unit: CalibrationUnit::Millimetres,
        });
        WindowChromeEvent::consumed(true)
    }

    pub(super) fn edit_calibration(
        &mut self,
        event: &PresentationEvent,
        width: u32,
        height: u32,
    ) -> Result<Option<WindowChromeEvent>> {
        let Some(dialog) = self.calibration.as_mut() else {
            return Ok(None);
        };
        match event {
            PresentationEvent::FocusLost => {
                self.calibration = None;
                self.cancel_pointer_capture();
                return Ok(Some(WindowChromeEvent {
                    consumed: false,
                    repaint: true,
                    action: None,
                }));
            }
            PresentationEvent::PointerDown {
                x,
                y,
                button: PointerButton::Left,
                ..
            } => {
                for (unit, rect) in CalibrationDialog::unit_rects(width, height)? {
                    if super::series::rect_contains(rect, *x, *y) {
                        dialog.unit = unit;
                        if dialog.input_state == InputState::Rejected {
                            dialog.input_state = InputState::Editing;
                        }
                        break;
                    }
                }
            }
            PresentationEvent::TextInput { character } => match dialog.input.insert(*character) {
                DecimalEdit::Appended => dialog.input_state = InputState::Editing,
                DecimalEdit::Full => dialog.input_state = InputState::Full,
                _ => {}
            },
            PresentationEvent::KeyDown {
                virtual_key,
                repeated: false,
                modifiers,
            } => match *virtual_key {
                0x09 => {
                    dialog.unit = if modifiers.shift() {
                        dialog.unit.previous()
                    } else {
                        dialog.unit.next()
                    };
                    if dialog.input_state == InputState::Rejected {
                        dialog.input_state = InputState::Editing;
                    }
                }
                0x08 => {
                    if dialog.input.backspace() {
                        dialog.input_state = InputState::Editing;
                    }
                }
                0x0d => match dialog
                    .input
                    .as_str()
                    .parse::<f32>()
                    .map_err(|_| CalibrationInputError::InvalidDistance)
                    .and_then(|value| dialog.unit.millimetres(value))
                {
                    Ok(mm) if dialog.input_state != InputState::Full => {
                        let action = WindowAction::ApplyLengthCalibration {
                            panel: dialog.panel,
                            reference: dialog.reference,
                            known_mm: Some(mm.to_bits()),
                        };
                        self.calibration = None;
                        return Ok(Some(WindowChromeEvent {
                            consumed: true,
                            repaint: true,
                            action: Some(action),
                        }));
                    }
                    _ => {
                        if dialog.input_state != InputState::Full {
                            dialog.input_state = InputState::Rejected;
                        }
                    }
                },
                0x1b => self.calibration = None,
                _ => {}
            },
            _ => return Ok(Some(WindowChromeEvent::consumed(false))),
        }
        Ok(Some(WindowChromeEvent::consumed(true)))
    }
}

impl CalibrationDialog {
    fn panel_origin(width: u32, height: u32) -> Result<(i32, i32)> {
        let width = i32::try_from(width)?;
        let height = i32::try_from(height)?;
        Ok((
            (width - 286).div_euclid(2).max(0),
            (height - 124).div_euclid(2).max(0),
        ))
    }

    fn unit_rects(width: u32, height: u32) -> Result<[(CalibrationUnit, Rect); 3]> {
        let (x, y) = Self::panel_origin(width, height)?;
        Ok([
            (
                CalibrationUnit::Millimetres,
                Rect::new(x + 12, y + 80, 82, 30),
            ),
            (CalibrationUnit::Inches, Rect::new(x + 100, y + 80, 82, 30)),
            (CalibrationUnit::French, Rect::new(x + 188, y + 80, 86, 30)),
        ])
    }

    pub(super) fn render(&self, framebuffer: &mut Framebuffer) -> Result<()> {
        let (x, y) = Self::panel_origin(framebuffer.width(), framebuffer.height())?;
        let rect = Rect::new(x, y, 286, 124);
        fill_rect(
            framebuffer,
            rect,
            CornerRadius::SQUARE,
            Color::rgb(35, 40, 47),
        );
        let text = text_style(Color::rgb(221, 228, 235), 13)?;
        let caption = match self.input_state {
            InputState::Editing => "Known distance · Tab: units · Enter",
            InputState::Rejected => "Invalid distance or unit range",
            InputState::Full => "Distance is too long",
        };
        let caption_bounds = Rect::new(x + 12, y + 8, 262, 26);
        draw_text(
            framebuffer,
            caption_bounds.x,
            text.centered_line_box_top(caption_bounds)?,
            caption,
            text,
        );
        let field = Rect::new(x + 12, y + 38, 262, 32);
        fill_rect(
            framebuffer,
            field,
            CornerRadius::SQUARE,
            Color::rgb(56, 63, 72),
        );
        let input_bounds = Rect::new(field.x + 8, field.y, field.width - 16, field.height);
        let overflow = whole_pixels(
            (text.advance(self.input.as_str()) - f64::from(input_bounds.width)).max(0.0),
        )?;
        let input_x = i32::try_from(i64::from(input_bounds.x) - i64::from(overflow))?;
        let input_y = text.centered_line_box_top(input_bounds)?;
        framebuffer.render_clipped(input_bounds, |surface| {
            draw_text(surface, input_x, input_y, self.input.as_str(), text);
        });
        for (unit, rect) in Self::unit_rects(framebuffer.width(), framebuffer.height())? {
            let color = if self.unit == unit {
                Color::rgb(53, 103, 151)
            } else {
                Color::rgb(56, 63, 72)
            };
            fill_rect(framebuffer, rect, CornerRadius::SQUARE, color);
            draw_text(
                framebuffer,
                rect.x + 8,
                text.centered_line_box_top(rect)?,
                unit.label(),
                text,
            );
        }
        Ok(())
    }
}
