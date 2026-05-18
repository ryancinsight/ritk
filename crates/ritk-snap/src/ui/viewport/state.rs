//! Viewport state types and struct definitions.
//!
//! [`ViewportRenderMode`] — rendering mode enum (Slice / Mip / Vr).
//! [`ViewportState`] — persistent per-viewport display state.
//! [`ViewportPanel`] — ephemeral per-frame viewport widget struct.
//!
//! # Image-to-screen transform
//!
//! The transform is a uniform scale followed by a translation:
//!
//! ```text
//! screen = offset + scale × img_pos
//! ```
//!
//! ## Transform theorem (invertibility)
//!
//! Let $T: \mathbb{R}^2 \to \mathbb{R}^2$ be
//! $$
//! T(p) = o + s p
//! $$
//! where $o \in \mathbb{R}^2$ is `offset` and $s \in \mathbb{R}$ is `scale`.
//! For `scale > 0`, $T$ is bijective with inverse
//! $$
//! T^{-1}(q) = \frac{q - o}{s}
//! $$
//! implemented by `screen_to_img_f32`.
//!
//! Proof sketch:
//! - Injective: if $T(p_1)=T(p_2)$ then $o+s p_1=o+s p_2$, so $s(p_1-p_2)=0$;
//!   since $s>0$, $p_1=p_2$.
//! - Surjective: for any $q$, choose $p=(q-o)/s$ and then $T(p)=q$.
//!
//! Therefore, pointer interactions expressed through this transform are
//! mathematically well-posed whenever `scale > 0`.
//!
//! where
//! - `img_pos` is in image-pixel coordinates `(col, row)`,
//! - `scale` is derived from `zoom` and the fit-to-viewport base scale,
//! - `offset` incorporates the fit centering and `pan_offset`.
//!
//! ## Slice index bounds
//!
//! | axis | dimension          | valid index range |
//! |------|--------------------|-------------------|
//! | 0    | `shape[0]` (depth) | `[0, depth−1]`   |
//! | 1    | `shape[1]` (rows)  | `[0, rows−1]`    |
//! | 2    | `shape[2]` (cols)  | `[0, cols−1]`    |
//!
//! Out-of-range indices are silently clamped by [`ViewportState::clamp_slice_index`].

use crate::{
    render::{colormap::Colormap, slice_render::WindowLevel},
    tools::{
        interaction::{Annotation, ToolState},
        kind::ToolKind,
    },
    LoadedVolume,
};
use egui::{pos2, Color32, Id, Pos2, Rect, Stroke, TextureHandle, Vec2};

/// Rendering mode for a viewport slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewportRenderMode {
    /// Conventional per-slice multi-planar reconstruction rendering.
    Slice,
    /// Axial maximum-intensity projection across full depth.
    Mip,
    /// Axial front-to-back alpha composited volume rendering.
    Vr,
}

// ── ViewportState ─────────────────────────────────────────────────────────────

/// Persistent per-viewport display state, independent of the egui frame.
///
/// One instance is stored per active viewport slot in [`crate::app::SnapApp`].
pub struct ViewportState {
    /// MPR axis: 0 = axial (fixed depth), 1 = coronal (fixed row),
    /// 2 = sagittal (fixed column).
    pub axis: usize,
    /// Currently displayed slice index along `axis`.
    pub slice_index: usize,
    /// Zoom factor relative to fit-to-viewport (1.0 = fit).
    pub zoom: f32,
    /// Pan offset in screen pixels, applied after fit centering.
    pub pan_offset: Vec2,
    /// Current window/level settings.
    pub wl: WindowLevel,
    /// Active colormap.
    pub colormap: Colormap,
    /// Completed measurement annotations for this viewport.
    pub annotations: Vec<Annotation>,
    /// In-progress tool state.
    pub tool_state: ToolState,
    /// Whether to draw the 4-corner DICOM text overlay.
    pub show_overlay: bool,
    /// Whether to draw the crosshair lines.
    pub show_crosshair: bool,
    /// Cached texture for the current slice (None when the slice has changed).
    pub texture: Option<TextureHandle>,
    /// Key identifying the currently cached texture `(axis, slice_index)`.
    /// `None` means the texture must be (re-)rendered.
    pub texture_slice_key: Option<(usize, usize)>,
    /// Rendering mode for this viewport slot.
    pub render_mode: ViewportRenderMode,
}

impl ViewportState {
    /// Construct a viewport state for the given `axis` and initial WL.
    pub fn new(axis: usize, wl: WindowLevel) -> Self {
        Self {
            axis,
            slice_index: 0,
            zoom: 1.0,
            pan_offset: Vec2::ZERO,
            wl,
            colormap: Colormap::Grayscale,
            annotations: Vec::new(),
            tool_state: ToolState::Idle,
            show_overlay: true,
            show_crosshair: true,
            texture: None,
            texture_slice_key: None,
            render_mode: ViewportRenderMode::Slice,
        }
    }

    /// Axial viewport (axis = 0).
    pub fn for_axial(wl: WindowLevel) -> Self {
        Self::new(0, wl)
    }

    /// Coronal viewport (axis = 1).
    pub fn for_coronal(wl: WindowLevel) -> Self {
        Self::new(1, wl)
    }

    /// Sagittal viewport (axis = 2).
    pub fn for_sagittal(wl: WindowLevel) -> Self {
        Self::new(2, wl)
    }

    /// 3-D / MIP viewport — uses axial data and enables MIP mode.
    pub fn for_mip(wl: WindowLevel) -> Self {
        let mut state = Self::new(0, wl);
        state.render_mode = ViewportRenderMode::Mip;
        state
    }

    /// Clamp `slice_index` to the valid range `[0, dim − 1]` for the
    /// current `axis` and `volume`.
    ///
    /// No-op when `volume.shape[axis] == 0` (degenerate volume).
    pub fn clamp_slice_index(&mut self, volume: &LoadedVolume) {
        let dim = volume.shape[self.axis.min(2)];
        if dim > 0 {
            self.slice_index = self.slice_index.min(dim - 1);
        }
    }

    /// Invalidate the cached texture so it is re-rendered on the next frame.
    pub fn invalidate_texture(&mut self) {
        self.texture = None;
        self.texture_slice_key = None;
    }

    /// Compute the image-to-screen transform for a viewport of `viewport_rect`
    /// displaying an image of `(img_w, img_h)` pixels.
    ///
    /// Returns `(offset, scale)` such that:
    /// ```text
    /// screen_pos = offset + scale × img_pos
    /// ```
    /// where `img_pos = Pos2 { x: col as f32, y: row as f32 }`.
    ///
    /// # Algorithm
    /// 1. Compute `base_scale` = min(vp_w / img_w, vp_h / img_h) (fit-to-viewport).
    /// 2. Apply zoom: `scale = base_scale × zoom`.
    /// 3. Centre the scaled image in the viewport.
    /// 4. Add `pan_offset`.
    pub fn image_transform(
        &self,
        viewport_rect: egui::Rect,
        img_w: usize,
        img_h: usize,
    ) -> (Vec2, f32) {
        let vp_w = viewport_rect.width();
        let vp_h = viewport_rect.height();
        let base_scale = if img_w == 0 || img_h == 0 || vp_w == 0.0 || vp_h == 0.0 {
            1.0_f32
        } else {
            (vp_w / img_w as f32).min(vp_h / img_h as f32)
        };
        let scale = base_scale * self.zoom;
        let img_screen_w = img_w as f32 * scale;
        let img_screen_h = img_h as f32 * scale;
        // Centre offset: shift so the image occupies the middle of the viewport.
        let center_x = viewport_rect.min.x + (vp_w - img_screen_w) * 0.5;
        let center_y = viewport_rect.min.y + (vp_h - img_screen_h) * 0.5;
        let offset = Vec2::new(center_x, center_y) + self.pan_offset;
        (offset, scale)
    }
}

// ── ViewportPanel ─────────────────────────────────────────────────────────────

/// Ephemeral per-frame viewport widget.
///
/// Borrows mutable state for one `update()` call then is discarded.
pub struct ViewportPanel<'a> {
    /// Unique widget id (used for egui interaction tracking).
    pub id: Id,
    /// The volume to display, or `None` when no volume is loaded.
    pub volume: Option<&'a LoadedVolume>,
    /// Mutable viewport state.
    pub state: &'a mut ViewportState,
    /// Currently active tool, used to route pointer events.
    pub active_tool: ToolKind,
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Return the `(width, height)` of the 2-D slice image for the given axis.
///
/// Matches the convention in [`LoadedVolume::extract_slice`]:
/// - axis 0 (axial):   `(cols, rows)`
/// - axis 1 (coronal): `(cols, depth)`
/// - axis 2 (sagittal):`(rows, depth)`
pub(super) fn slice_dims(volume: &LoadedVolume, axis: usize) -> (usize, usize) {
    let [depth, rows, cols] = volume.shape;
    match axis {
        0 => (cols, rows),
        1 => (cols, depth),
        _ => (rows, depth),
    }
}

/// Convert a screen position to integer image coordinates `(col, row)`.
///
/// Returns `None` when the position is outside the image bounds.
pub(super) fn screen_to_img(
    screen: Pos2,
    offset: Vec2,
    scale: f32,
    img_w: usize,
    img_h: usize,
) -> Option<(usize, usize)> {
    if scale <= 0.0 {
        return None;
    }
    let col_f = (screen.x - offset.x) / scale;
    let row_f = (screen.y - offset.y) / scale;
    if col_f < 0.0 || row_f < 0.0 {
        return None;
    }
    let col = col_f as usize;
    let row = row_f as usize;
    if col >= img_w || row >= img_h {
        return None;
    }
    Some((col, row))
}

/// Convert image coordinates `(col,row)` to screen coordinates.
///
/// This is the forward affine map `screen = offset + scale * image`.
#[inline]
pub(super) fn img_to_screen(img: Pos2, offset: Vec2, scale: f32) -> Pos2 {
    pos2(offset.x + img.x * scale, offset.y + img.y * scale)
}

/// Convert a screen position to floating-point image coordinates `(col, row)`.
///
/// No bounds check; returns `None` only when `scale <= 0`.
pub(super) fn screen_to_img_f32(screen: Pos2, offset: Vec2, scale: f32) -> Option<(f32, f32)> {
    if scale <= 0.0 {
        return None;
    }
    Some(((screen.x - offset.x) / scale, (screen.y - offset.y) / scale))
}

/// Map image-pixel coordinates `(col, row)` on a given `axis` slice to
/// volume-voxel coordinates `[depth, row, col]`.
///
/// | axis | slice    | img_col | img_row | volume coords                |
/// |------|----------|---------|---------|------------------------------|
/// | 0    | fixed d  | col     | row     | `[slice, img_row, img_col]`  |
/// | 1    | fixed r  | col     | depth   | `[img_row, slice, img_col]`  |
/// | 2    | fixed c  | row     | depth   | `[img_row, img_col, slice]`  |
pub(super) fn img_to_volume(
    img_col: usize,
    img_row: usize,
    slice: usize,
    axis: usize,
) -> [usize; 3] {
    match axis {
        0 => [slice, img_row, img_col],
        1 => [img_row, slice, img_col],
        _ => [img_row, img_col, slice],
    }
}

/// Draw a thin crosshair (full-length horizontal + vertical lines) through
/// the centre of `rect`.
pub(super) fn draw_crosshair(painter: &egui::Painter, rect: Rect, color: Color32) {
    let cx = rect.center().x;
    let cy = rect.center().y;
    let stroke = Stroke::new(0.75_f32, color);
    // Horizontal line
    painter.line_segment([pos2(rect.min.x, cy), pos2(rect.max.x, cy)], stroke);
    // Vertical line
    painter.line_segment([pos2(cx, rect.min.y), pos2(cx, rect.max.y)], stroke);
}
