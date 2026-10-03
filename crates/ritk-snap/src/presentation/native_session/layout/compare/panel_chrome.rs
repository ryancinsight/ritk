//! Draw panel maximize and close controls.

use super::{
    PANEL_ACTION_BACKGROUND, PANEL_ACTION_FOREGROUND, PANEL_ACTION_GAP, PANEL_ACTION_WIDTH,
    PANEL_HEADER_HEIGHT,
};
use anyhow::{anyhow, Result};
use metis_platform::rasterizer::{fill_rect, CornerRadius};
use metis_platform::{Framebuffer, Rect};

pub(super) fn draw_panel_actions(
    framebuffer: &mut Framebuffer,
    panel_x: u32,
    panel_y: u32,
    panel_width: u32,
    maximized: bool,
    show_maximize: bool,
) -> Result<()> {
    let right = panel_x
        .checked_add(panel_width)
        .ok_or_else(|| anyhow!("series-grid panel right edge overflows"))?;
    let close_x = right
        .checked_sub(
            u32::try_from(PANEL_ACTION_WIDTH)
                .map_err(|_| anyhow!("panel action width is negative"))?,
        )
        .ok_or_else(|| anyhow!("series-grid close action is outside its panel"))?;
    let action_y = panel_y
        .checked_add(4)
        .ok_or_else(|| anyhow!("series-grid action y overflows"))?;
    let action_width =
        u32::try_from(PANEL_ACTION_WIDTH).map_err(|_| anyhow!("panel action width is negative"))?;
    let action_height = PANEL_HEADER_HEIGHT.saturating_sub(8);
    fill_rect(
        framebuffer,
        super::make_rect(close_x, action_y, action_width, action_height)?,
        CornerRadius::SQUARE,
        PANEL_ACTION_BACKGROUND,
    );
    draw_close_icon(framebuffer, close_x, action_y, action_width, action_height)?;
    if show_maximize {
        let gap =
            u32::try_from(PANEL_ACTION_GAP).map_err(|_| anyhow!("panel action gap is negative"))?;
        let maximize_x = close_x
            .checked_sub(gap)
            .and_then(|x| x.checked_sub(action_width))
            .ok_or_else(|| anyhow!("series-grid maximize action is outside its panel"))?;
        fill_rect(
            framebuffer,
            super::make_rect(maximize_x, action_y, action_width, action_height)?,
            CornerRadius::SQUARE,
            PANEL_ACTION_BACKGROUND,
        );
        draw_maximize_icon(
            framebuffer,
            maximize_x,
            action_y,
            action_width,
            action_height,
            maximized,
        )?;
    }
    Ok(())
}

fn draw_close_icon(
    framebuffer: &mut Framebuffer,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> Result<()> {
    let origin_x = i32::try_from(
        x.checked_add(width.saturating_sub(7) / 2)
            .ok_or_else(|| anyhow!("panel close icon x overflows"))?,
    )?;
    let origin_y = i32::try_from(
        y.checked_add(height.saturating_sub(7) / 2)
            .ok_or_else(|| anyhow!("panel close icon y overflows"))?,
    )?;
    for offset in 0..7_i32 {
        let reverse = 6_i32.saturating_sub(offset);
        let icon_x = origin_x
            .checked_add(offset)
            .ok_or_else(|| anyhow!("panel close icon x exceeds i32"))?;
        let down_y = origin_y
            .checked_add(offset)
            .ok_or_else(|| anyhow!("panel close icon y exceeds i32"))?;
        let up_y = origin_y
            .checked_add(reverse)
            .ok_or_else(|| anyhow!("panel close icon y exceeds i32"))?;
        fill_rect(
            framebuffer,
            Rect::new(icon_x, down_y, 1, 1),
            CornerRadius::SQUARE,
            PANEL_ACTION_FOREGROUND,
        );
        fill_rect(
            framebuffer,
            Rect::new(icon_x, up_y, 1, 1),
            CornerRadius::SQUARE,
            PANEL_ACTION_FOREGROUND,
        );
    }
    Ok(())
}

fn draw_maximize_icon(
    framebuffer: &mut Framebuffer,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    maximized: bool,
) -> Result<()> {
    let offset_x = i32::try_from(
        x.checked_add(width.saturating_sub(8) / 2)
            .ok_or_else(|| anyhow!("panel maximize icon x overflows"))?,
    )?;
    let offset_y = i32::try_from(
        y.checked_add(height.saturating_sub(8) / 2)
            .ok_or_else(|| anyhow!("panel maximize icon y overflows"))?,
    )?;
    if maximized {
        for (x_offset, y_offset) in [(2, 0), (0, 2)] {
            let square_x = offset_x
                .checked_add(x_offset)
                .ok_or_else(|| anyhow!("panel restore icon x exceeds i32"))?;
            let square_y = offset_y
                .checked_add(y_offset)
                .ok_or_else(|| anyhow!("panel restore icon y exceeds i32"))?;
            draw_square(framebuffer, square_x, square_y, 6)?;
        }
    } else {
        draw_square(framebuffer, offset_x, offset_y, 7)?;
    }
    Ok(())
}

fn draw_square(framebuffer: &mut Framebuffer, x: i32, y: i32, size: i32) -> Result<()> {
    if size <= 0 {
        return Ok(());
    }
    let final_offset = size
        .checked_sub(1)
        .ok_or_else(|| anyhow!("panel icon square has invalid size"))?;
    let bottom_y = y
        .checked_add(final_offset)
        .ok_or_else(|| anyhow!("panel icon square y exceeds i32"))?;
    let right_x = x
        .checked_add(final_offset)
        .ok_or_else(|| anyhow!("panel icon square x exceeds i32"))?;
    for offset in 0..size {
        let horizontal_x = x
            .checked_add(offset)
            .ok_or_else(|| anyhow!("panel icon square x exceeds i32"))?;
        let vertical_y = y
            .checked_add(offset)
            .ok_or_else(|| anyhow!("panel icon square y exceeds i32"))?;
        for (px, py) in [
            (horizontal_x, y),
            (horizontal_x, bottom_y),
            (x, vertical_y),
            (right_x, vertical_y),
        ] {
            fill_rect(
                framebuffer,
                Rect::new(px, py, 1, 1),
                CornerRadius::SQUARE,
                PANEL_ACTION_FOREGROUND,
            );
        }
    }
    Ok(())
}
