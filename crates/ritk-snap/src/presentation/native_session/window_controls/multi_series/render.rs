use super::super::series::{draw_fit, offset, patient_label, study_label};
use super::{
    DialogGeometry, MultiSeriesDialog, BORDER, FOCUSED_ROW, FOOTER_HEIGHT, LIST_HEADER_HEIGHT,
    MAX_SELECTION, MUTED, OVERLAY, PANEL, ROW, ROW_HEIGHT, SELECTED, TEXT, WARNING,
};
use crate::presentation::native_session::layout::{draw_text, text_style, TextStyle};
use crate::presentation::native_session::series_browser::SeriesBrowser;
use anyhow::{anyhow, Result};
use arrayvec::ArrayString;
use metis_platform::rasterizer::{fill_rect, CornerRadius};
use metis_platform::{Color, Framebuffer, Rect};
use std::fmt::Write as _;

pub(super) const TABLE_HEADER_BACKGROUND: Color = Color::rgb(25, 30, 36);
const TABLE_HEADER_TEXT: Color = Color::rgb(167, 182, 196);

impl MultiSeriesDialog {
    pub(in crate::presentation::native_session::window_controls) fn render(
        &self,
        framebuffer: &mut Framebuffer,
        browser: &SeriesBrowser,
    ) -> Result<()> {
        let Some(geometry) = DialogGeometry::new(framebuffer.width(), framebuffer.height())? else {
            return Ok(());
        };
        fill_rect(
            framebuffer,
            Rect::new(
                0,
                0,
                i32::try_from(framebuffer.width())
                    .map_err(|_| anyhow!("dialog surface width exceeds i32"))?,
                i32::try_from(framebuffer.height())
                    .map_err(|_| anyhow!("dialog surface height exceeds i32"))?,
            ),
            CornerRadius::SQUARE,
            OVERLAY,
        );
        fill_rect(framebuffer, geometry.dialog, CornerRadius::SQUARE, PANEL);
        draw_border(framebuffer, geometry.dialog, BORDER);

        let title = text_style(TEXT, 18)?;
        let regular = text_style(TEXT, 12)?;
        let detail = text_style(MUTED, 11)?;
        draw_text(
            framebuffer,
            offset(geometry.dialog.x, 20)?,
            offset(geometry.dialog.y, 18)?,
            "Open multiple series",
            title,
        );
        draw_text(
            framebuffer,
            offset(geometry.dialog.x, 20)?,
            offset(geometry.dialog.y, 48)?,
            "Ctrl-click or Space selects series; Enter opens the selection or first match",
            detail,
        );

        fill_rect(framebuffer, geometry.filter, CornerRadius::SQUARE, ROW);
        draw_text(
            framebuffer,
            offset(geometry.filter.x, 10)?,
            offset(geometry.filter.y, 10)?,
            "Find text…",
            detail,
        );
        let filter_text = if self.filter.is_empty() {
            "Description, modality, patient or study"
        } else {
            self.filter.as_str()
        };
        draw_fit(
            framebuffer,
            offset(geometry.filter.x, 88)?,
            offset(geometry.filter.y, 10)?,
            filter_text,
            regular,
            geometry.filter.width.saturating_sub(100),
        );

        let list_header = Rect::new(
            geometry.list.x,
            geometry.list.y.saturating_sub(LIST_HEADER_HEIGHT),
            geometry.list.width,
            LIST_HEADER_HEIGHT,
        );
        fill_rect(
            framebuffer,
            list_header,
            CornerRadius::SQUARE,
            TABLE_HEADER_BACKGROUND,
        );
        fill_rect(
            framebuffer,
            geometry.list,
            CornerRadius::SQUARE,
            Color::rgb(22, 27, 33),
        );
        let column_style = text_style(TABLE_HEADER_TEXT, 9)?;
        let patient_column = geometry.list.x.saturating_add(34);
        let study_column = geometry.list.x.saturating_add(160);
        let modality_column = geometry.list.x.saturating_add(318);
        let series_column = geometry.list.x.saturating_add(368);
        let images_column = geometry
            .list
            .x
            .saturating_add(geometry.list.width)
            .saturating_sub(56);
        for (x, label) in [
            (patient_column, "PATIENT"),
            (study_column, "STUDY"),
            (modality_column, "MODALITY"),
            (series_column, "SERIES"),
            (images_column, "IMAGES"),
        ] {
            draw_text(
                framebuffer,
                x,
                list_header.y.saturating_add(8),
                label,
                column_style,
            );
        }
        for x in [
            geometry.list.x.saturating_add(150),
            geometry.list.x.saturating_add(308),
            geometry.list.x.saturating_add(358),
        ] {
            fill_rect(
                framebuffer,
                Rect::new(
                    x,
                    list_header.y.saturating_add(4),
                    1,
                    LIST_HEADER_HEIGHT - 8,
                ),
                CornerRadius::SQUARE,
                BORDER,
            );
        }
        let count = self.visible_rows(geometry);
        for row in 0..count {
            let visible_index = self.first_visible.saturating_add(row);
            let Some(series_index) = self.matches.get(visible_index).copied() else {
                break;
            };
            let choice = browser
                .choice(series_index)
                .ok_or_else(|| anyhow!("filtered series is absent from the study catalog"))?;
            let row_index = i32::try_from(row).map_err(|_| anyhow!("dialog row exceeds i32"))?;
            let row_y = geometry
                .list
                .y
                .checked_add(row_index.saturating_mul(ROW_HEIGHT))
                .ok_or_else(|| anyhow!("dialog row y overflows"))?;
            let bounds = Rect::new(
                geometry.list.x.saturating_add(1),
                row_y,
                geometry.list.width.saturating_sub(2),
                ROW_HEIGHT.saturating_sub(1),
            );
            fill_rect(
                framebuffer,
                bounds,
                CornerRadius::SQUARE,
                if visible_index == self.cursor {
                    FOCUSED_ROW
                } else {
                    ROW
                },
            );
            let checked = self.selected.contains(&series_index);
            let check = Rect::new(
                bounds.x.saturating_add(9),
                bounds.y.saturating_add(8),
                14,
                14,
            );
            fill_rect(
                framebuffer,
                check,
                CornerRadius::SQUARE,
                if checked { SELECTED } else { PANEL },
            );
            if checked {
                fill_rect(
                    framebuffer,
                    Rect::new(check.x + 3, check.y + 6, 3, 5),
                    CornerRadius::SQUARE,
                    TEXT,
                );
                fill_rect(
                    framebuffer,
                    Rect::new(check.x + 6, check.y + 8, 6, 3),
                    CornerRadius::SQUARE,
                    TEXT,
                );
            }
            let patient = patient_label(choice)?;
            draw_fit(
                framebuffer,
                patient_column,
                bounds.y.saturating_add(10),
                patient.as_str(),
                detail,
                112,
            );
            draw_fit(
                framebuffer,
                study_column,
                bounds.y.saturating_add(10),
                study_label(choice)?.as_str(),
                detail,
                142,
            );
            draw_fit(
                framebuffer,
                modality_column,
                bounds.y.saturating_add(10),
                choice.modality.as_ref(),
                regular,
                38,
            );
            let mut image_count = ArrayString::<32>::new();
            write!(&mut image_count, "{} images", choice.image_count)
                .map_err(|_| anyhow!("series image count exceeds its display buffer"))?;
            let count_width = detail
                .extent(0, 0, image_count.as_str())
                .map_or(0, |extent| extent.width);
            let count_x = bounds
                .x
                .saturating_add(bounds.width)
                .saturating_sub(count_width.saturating_add(12));
            draw_fit(
                framebuffer,
                series_column,
                bounds.y.saturating_add(10),
                choice.description.as_ref(),
                regular,
                count_x.saturating_sub(series_column).saturating_sub(8),
            );
            draw_text(
                framebuffer,
                count_x,
                bounds.y.saturating_add(10),
                image_count.as_str(),
                detail,
            );
        }
        if self.matches.is_empty() {
            draw_text(
                framebuffer,
                geometry.list.x.saturating_add(18),
                geometry.list.y.saturating_add(12),
                "No series match this filter",
                detail,
            );
        }

        let footer_y = geometry.dialog.y + geometry.dialog.height - FOOTER_HEIGHT;
        let mut status = ArrayString::<80>::new();
        write!(
            &mut status,
            "{} selected  |  maximum {} panels",
            self.selected.len(),
            MAX_SELECTION
        )
        .map_err(|_| anyhow!("dialog selection status exceeds its display buffer"))?;
        if self.selection_limit_reached {
            draw_text(
                framebuffer,
                geometry.dialog.x.saturating_add(20),
                footer_y.saturating_add(5),
                "Panel capacity reached; remove a selection before adding another.",
                text_style(WARNING, 10)?,
            );
        } else {
            draw_text(
                framebuffer,
                geometry.dialog.x.saturating_add(20),
                footer_y.saturating_add(5),
                status.as_str(),
                detail,
            );
        }
        draw_button(framebuffer, geometry.cancel, "Cancel", regular, false);
        draw_button(
            framebuffer,
            geometry.open,
            "Open",
            regular,
            !self.matches.is_empty() || !self.selected.is_empty(),
        );
        Ok(())
    }
}
fn draw_button(
    framebuffer: &mut Framebuffer,
    bounds: Rect,
    label: &str,
    style: TextStyle,
    primary: bool,
) {
    fill_rect(
        framebuffer,
        bounds,
        CornerRadius::SQUARE,
        if primary { SELECTED } else { ROW },
    );
    let width = style.extent(0, 0, label).map_or(0, |extent| extent.width);
    let x = bounds
        .x
        .saturating_add(bounds.width.saturating_sub(width) / 2);
    let y = bounds.y.saturating_add((bounds.height - 14).max(0) / 2);
    draw_text(framebuffer, x, y, label, style);
}

fn draw_border(framebuffer: &mut Framebuffer, bounds: Rect, color: Color) {
    fill_rect(
        framebuffer,
        Rect::new(bounds.x, bounds.y, bounds.width, 1),
        CornerRadius::SQUARE,
        color,
    );
    fill_rect(
        framebuffer,
        Rect::new(bounds.x, bounds.y + bounds.height - 1, bounds.width, 1),
        CornerRadius::SQUARE,
        color,
    );
    fill_rect(
        framebuffer,
        Rect::new(bounds.x, bounds.y, 1, bounds.height),
        CornerRadius::SQUARE,
        color,
    );
    fill_rect(
        framebuffer,
        Rect::new(bounds.x + bounds.width - 1, bounds.y, 1, bounds.height),
        CornerRadius::SQUARE,
        color,
    );
}
