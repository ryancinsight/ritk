use super::super::layout::{draw_text, text_style};
use super::{WindowAction, WindowChrome, WindowChromeEvent};
use crate::app::{LengthCalibrationReference, SnapApp};
use crate::presentation::PresentationEvent;
use anyhow::Result;
use metis_platform::rasterizer::{fill_rect, CornerRadius};
use metis_platform::{Color, Framebuffer, Rect};
use metis_ui_lang::{DecimalEdit, DecimalEntry};

#[derive(Clone)]
pub(super) struct CalibrationDialog {
    panel: usize,
    reference: LengthCalibrationReference,
    input: DecimalEntry<32>,
    invalid: bool,
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
            invalid: false,
        });
        WindowChromeEvent::consumed(true)
    }

    pub(super) fn edit_calibration(
        &mut self,
        event: &PresentationEvent,
    ) -> Option<WindowChromeEvent> {
        let dialog = self.calibration.as_mut()?;
        match event {
            PresentationEvent::FocusLost => {
                self.calibration = None;
                self.control_down = false;
                self.cancel_pointer_capture();
                return Some(WindowChromeEvent {
                    consumed: false,
                    repaint: true,
                    action: None,
                });
            }
            PresentationEvent::TextInput { character } => match dialog.input.insert(*character) {
                DecimalEdit::Appended => dialog.invalid = false,
                DecimalEdit::Full => dialog.invalid = true,
                _ => {}
            },
            PresentationEvent::KeyDown {
                virtual_key,
                repeated: false,
                ..
            } => match *virtual_key {
                0x08 => {
                    if dialog.input.backspace() {
                        dialog.invalid = false;
                    }
                }
                0x0d => match dialog.input.as_str().parse::<f32>() {
                    Ok(mm) if !dialog.invalid && mm.is_finite() && mm > 0.0 => {
                        let action = WindowAction::ApplyLengthCalibration {
                            panel: dialog.panel,
                            reference: dialog.reference,
                            known_mm: Some(mm.to_bits()),
                        };
                        self.calibration = None;
                        return Some(WindowChromeEvent {
                            consumed: true,
                            repaint: true,
                            action: Some(action),
                        });
                    }
                    _ => dialog.invalid = true,
                },
                0x1b => self.calibration = None,
                _ => {}
            },
            _ => return Some(WindowChromeEvent::consumed(false)),
        }
        Some(WindowChromeEvent::consumed(true))
    }
}

impl CalibrationDialog {
    pub(super) fn render(&self, framebuffer: &mut Framebuffer) -> Result<()> {
        let width = i32::try_from(framebuffer.width())?;
        let height = i32::try_from(framebuffer.height())?;
        let x = (width - 286).div_euclid(2).max(0);
        let y = (height - 86).div_euclid(2).max(0);
        let rect = Rect::new(x, y, 286, 86);
        fill_rect(
            framebuffer,
            rect,
            CornerRadius::SQUARE,
            Color::rgb(35, 40, 47),
        );
        let text = text_style(Color::rgb(221, 228, 235), 13)?;
        let caption = if self.invalid {
            "Enter a positive distance in millimetres"
        } else {
            "Known distance (mm) · Enter to apply · Esc"
        };
        draw_text(framebuffer, x + 12, y + 24, caption, text);
        let field = Rect::new(x + 12, y + 38, 262, 32);
        fill_rect(
            framebuffer,
            field,
            CornerRadius::SQUARE,
            Color::rgb(56, 63, 72),
        );
        draw_text(
            framebuffer,
            field.x + 8,
            field.y + 22,
            self.input.as_str(),
            text,
        );
        Ok(())
    }
}
