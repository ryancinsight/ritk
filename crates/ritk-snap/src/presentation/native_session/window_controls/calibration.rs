use crate::presentation::PresentationEvent;
use anyhow::Result;
use metis_platform::rasterizer::{fill_rect, CornerRadius};
use metis_platform::{Color, Framebuffer, Rect};
use metis_ui_lang::{DecimalEdit, DecimalEntry};

use super::super::layout::{draw_text, text_style};
use super::{WindowAction, WindowChrome, WindowChromeEvent};

const PANEL: Color = Color::rgb(43, 49, 57);
const FIELD: Color = Color::rgb(31, 36, 42);
const MUTED: Color = Color::rgb(183, 194, 204);
const ERROR: Color = Color::rgb(239, 130, 120);

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum DialogAction {
    None,
    Cancel,
    Confirm(f32),
}

#[derive(Debug, Clone)]
pub(super) struct CalibrationDialog {
    input: DecimalEntry<32>,
    error: Option<&'static str>,
}

impl CalibrationDialog {
    pub(super) fn new() -> Self {
        Self {
            input: DecimalEntry::new(),
            error: None,
        }
    }

    pub(super) fn handle_event(&mut self, event: &PresentationEvent) -> DialogAction {
        match event {
            PresentationEvent::TextInput { character } => {
                match self.input.insert(*character) {
                    DecimalEdit::Appended => self.error = None,
                    DecimalEdit::Full => self.error = Some("Distance is too long"),
                    _ => {}
                }
                DialogAction::None
            }
            PresentationEvent::KeyDown {
                virtual_key,
                repeated: false,
                ..
            } => match *virtual_key {
                0x08 => {
                    if self.input.backspace() {
                        self.error = None;
                    }
                    DialogAction::None
                }
                0x0d => match self.input.as_str().parse::<f32>() {
                    Ok(value) if value.is_finite() && value > 0.0 => DialogAction::Confirm(value),
                    _ => {
                        self.error = Some("Enter a positive distance in millimetres");
                        DialogAction::None
                    }
                },
                0x1b => DialogAction::Cancel,
                _ => DialogAction::None,
            },
            _ => DialogAction::None,
        }
    }

    pub(super) fn render(&self, framebuffer: &mut Framebuffer) -> Result<()> {
        let width = i32::try_from(framebuffer.width())
            .map_err(|_| anyhow::anyhow!("framebuffer width exceeds i32"))?;
        let height = i32::try_from(framebuffer.height())
            .map_err(|_| anyhow::anyhow!("framebuffer height exceeds i32"))?;
        let panel = Rect::new(
            (width - 360).div_euclid(2),
            (height - 200).div_euclid(2),
            360,
            200,
        );
        fill_rect(framebuffer, panel, CornerRadius::SQUARE, PANEL);
        let body = text_style(MUTED, 13)?;
        draw_text(
            framebuffer,
            panel.x + 18,
            panel.y + 24,
            "Calibrate length (mm)",
            text_style(Color::WHITE, 15)?,
        );
        draw_text(
            framebuffer,
            panel.x + 18,
            panel.y + 44,
            "Enter known distance; Enter confirms, Esc cancels",
            text_style(MUTED, 11)?,
        );
        let field = Rect::new(panel.x + 18, panel.y + 62, 324, 28);
        fill_rect(framebuffer, field, CornerRadius::SQUARE, FIELD);
        draw_text(
            framebuffer,
            field.x + 8,
            field.y + 20,
            self.input.as_str(),
            body,
        );
        if let Some(error) = self.error {
            draw_text(
                framebuffer,
                panel.x + 18,
                panel.y + 144,
                error,
                text_style(ERROR, 12)?,
            );
        }
        Ok(())
    }
}

impl WindowChrome {
    pub(super) fn handle_calibration_event(
        &mut self,
        event: &PresentationEvent,
    ) -> Option<WindowChromeEvent> {
        let dialog = self.calibration_dialog.as_mut()?;
        match dialog.handle_event(event) {
            DialogAction::None => Some(WindowChromeEvent::consumed(true)),
            DialogAction::Cancel => {
                self.calibration_dialog = None;
                self.calibration_target = None;
                Some(WindowChromeEvent::consumed(true))
            }
            DialogAction::Confirm(value) => {
                self.calibration_dialog = None;
                self.calibration_value = Some(value);
                let Some(target) = self.calibration_target.take() else {
                    return Some(WindowChromeEvent::consumed(true));
                };
                Some(WindowChromeEvent {
                    consumed: true,
                    repaint: true,
                    action: Some(WindowAction::ApplyCalibration(target)),
                })
            }
        }
    }
}
