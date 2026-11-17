//! Native manual calibration dialog for a selected length annotation.

use crate::presentation::PresentationEvent;
use anyhow::Result;
use metis_platform::rasterizer::{fill_rect, CornerRadius};
use metis_platform::{Color, Framebuffer, Rect};

use super::super::layout::{draw_text, text_style};
use super::{WindowAction, WindowChrome, WindowChromeEvent};
use arrayvec::ArrayString;

const PANEL: Color = Color::rgb(43, 49, 57);
const FIELD: Color = Color::rgb(31, 36, 42);
const TEXT: Color = Color::rgb(221, 228, 235);
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
    input: ArrayString<32>,
    error: Option<&'static str>,
}

impl CalibrationDialog {
    pub(super) fn new() -> Self {
        Self {
            input: ArrayString::new(),
            error: None,
        }
    }

    pub(super) fn handle_event(&mut self, event: &PresentationEvent) -> DialogAction {
        match event {
            PresentationEvent::TextInput { character } if !character.is_control() => {
                if character.is_ascii_digit() || matches!(character, '.' | '-' | '+' | 'e' | 'E') {
                    self.error = self
                        .input
                        .try_push(*character)
                        .err()
                        .map(|_| "Distance is too long");
                }
                DialogAction::None
            }
            PresentationEvent::KeyDown {
                virtual_key,
                repeated: false,
                ..
            } => match *virtual_key {
                0x08 => {
                    self.input.pop();
                    self.error = None;
                    DialogAction::None
                }
                0x0d => match self.input.trim().parse::<f32>() {
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
            (height - 176).div_euclid(2),
            360,
            176,
        );
        fill_rect(framebuffer, panel, CornerRadius::SQUARE, PANEL);
        let title = text_style(TEXT, 16)?;
        let body = text_style(MUTED, 13)?;
        draw_text(
            framebuffer,
            panel.x + 18,
            panel.y + 26,
            "Length calibration",
            title,
        );
        draw_text(
            framebuffer,
            panel.x + 18,
            panel.y + 53,
            "Known distance (mm)",
            body,
        );
        let field = Rect::new(panel.x + 18, panel.y + 62, 324, 28);
        fill_rect(framebuffer, field, CornerRadius::SQUARE, FIELD);
        draw_text(framebuffer, field.x + 8, field.y + 20, &self.input, body);
        if let Some(error) = &self.error {
            let error_style = text_style(ERROR, 12)?;
            draw_text(framebuffer, panel.x + 18, panel.y + 112, error, error_style);
        } else {
            draw_text(
                framebuffer,
                panel.x + 18,
                panel.y + 112,
                "Enter confirms; Escape cancels",
                body,
            );
        }
        draw_text(
            framebuffer,
            panel.x + 18,
            panel.y + 145,
            "Calibration applies to this image",
            body,
        );
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
                Some(WindowChromeEvent::consumed(true))
            }
            DialogAction::Confirm(value) => {
                self.calibration_dialog = None;
                self.calibration_value = Some(value);
                Some(WindowChromeEvent {
                    consumed: true,
                    repaint: true,
                    action: Some(WindowAction::ApplyCalibration),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CalibrationDialog;
    use crate::presentation::PresentationEvent;

    #[test]
    fn typed_distance_confirms_only_positive_finite_values() {
        let mut dialog = CalibrationDialog::new();
        for character in ['7', '.', '5'] {
            assert!(matches!(
                dialog.handle_event(&PresentationEvent::TextInput { character }),
                super::DialogAction::None
            ));
        }
        assert_eq!(
            dialog.handle_event(&PresentationEvent::KeyDown {
                virtual_key: 0x0d,
                repeated: false,
                modifiers: crate::presentation::PresentationModifiers::NONE,
            }),
            super::DialogAction::Confirm(7.5)
        );
    }
}
