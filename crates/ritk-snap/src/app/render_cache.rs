use super::state::ProjectionMode;

/// Per-voxel opacity scale for volume rendering. Canonical value per GPU VR spec.
const DEFAULT_VR_ALPHA: f32 = 0.06;
use super::state::SnapApp;
use super::viewport_render::{OVERLAY_LABEL_COLOR, OVERLAY_LABEL_FONT_SIZE, OVERLAY_LABEL_INSET};
use crate::render::mip_vr::{render_mip_axial_with_scratch, render_vr_axial_with_scratch};
use crate::render::slice_render::{SliceRenderer, WindowLevel};
use crate::ui::apply_to_image_into;
use crate::viewer::{DEFAULT_WINDOW_CENTER, DEFAULT_WINDOW_WIDTH};

impl SnapApp {
    pub(crate) fn rebuild_texture_for_axis(&mut self, ctx: &egui::Context, axis: usize) {
        let (color_image, tex_name) = {
            let Some(vol) = &self.loaded else {
                return;
            };
            let wc = self
                .viewer_state
                .window_center
                .unwrap_or(DEFAULT_WINDOW_CENTER) as f64;
            let ww = self
                .viewer_state
                .window_width
                .unwrap_or(DEFAULT_WINDOW_WIDTH)
                .max(1.0) as f64;
            let wl = WindowLevel::new(wc, ww);
            let slice_index = match axis {
                0 => self.viewer_state.slice_index,
                1 => self.coronal_slice,
                _ => self.sagittal_slice,
            };
            let name = match axis {
                0 => "slice_tex_axial",
                1 => "slice_tex_coronal",
                _ => "slice_tex_sagittal",
            };
            let img = SliceRenderer::render_with_scratch(
                &mut self.render_buffer_pool,
                vol,
                axis,
                slice_index,
                wl,
                self.colormap,
            );
            // Apply viewport orientation transform (flip/rotate) before GPU upload.
            let img = apply_to_image_into(&mut self.render_buffer_pool, &img, self.view_transform);
            (img, name)
        };
        // immutable borrow of self.loaded released here
        let tex = ctx.load_texture(tex_name, color_image, egui::TextureOptions::LINEAR);
        match axis {
            0 => self.texture = Some(tex),
            1 => self.coronal_tex = Some(tex),
            _ => self.sagittal_tex = Some(tex),
        }
    }

    /// Render the 3D-MIP projection through WL LUT and upload to the GPU.
    pub(crate) fn rebuild_texture_for_mip(&mut self, ctx: &egui::Context) {
        let Some(vol) = self.loaded.clone() else {
            return;
        };
        let wc = self
            .viewer_state
            .window_center
            .unwrap_or(DEFAULT_WINDOW_CENTER) as f64;
        let ww = self
            .viewer_state
            .window_width
            .unwrap_or(DEFAULT_WINDOW_WIDTH)
            .max(1.0) as f64;
        let wl = WindowLevel::new(wc, ww);

        // ── GPU-accelerated MIP and VR (native only) ─────────────────────────
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(ref mut gpu) = self.gpu_renderer {
            let gpu_img = match self.projection_mode {
                ProjectionMode::Mip => gpu.render_mip(&vol, wl, self.colormap),
                ProjectionMode::Vr => gpu.render_vr(&vol, wl, self.colormap, DEFAULT_VR_ALPHA),
            };
            if let Some(img) = gpu_img {
                self.mip_tex = Some(ctx.load_texture(
                    "slice_tex_mip_axial",
                    img,
                    egui::TextureOptions::LINEAR,
                ));
                return;
            }
            tracing::warn!(
                mode = ?self.projection_mode,
                "GPU render failed; falling back to CPU path"
            );
        }

        // ── CPU fallback (always available) ──────────────────────────────────
        let color_image = match self.projection_mode {
            ProjectionMode::Mip => render_mip_axial_with_scratch(
                &mut self.render_buffer_pool.rgba_u8,
                &vol,
                wl,
                self.colormap,
            ),
            ProjectionMode::Vr => render_vr_axial_with_scratch(
                &mut self.render_buffer_pool.rgba_u8,
                &vol,
                wl,
                self.colormap,
                0.06,
            ),
        };
        self.mip_tex = Some(ctx.load_texture(
            "slice_tex_mip_axial",
            color_image,
            egui::TextureOptions::LINEAR,
        ));
    }

    /// Render one 3D-MIP viewport into `ui`.
    pub(crate) fn render_mip_viewport(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let needs_rebuild = self.mip_dirty || self.mip_tex.is_none();

        if needs_rebuild && self.loaded.is_some() {
            self.rebuild_texture_for_mip(ctx);
            self.mip_dirty = false;
        }

        let Some((tex_id, [tex_w_usize, tex_h_usize])) =
            self.mip_tex.as_ref().map(|t| (t.id(), t.size()))
        else {
            ui.centered_and_justified(|ui| {
                ui.label("3D MIP — open a volume to begin");
            });
            return;
        };

        let tex_w = tex_w_usize as f32;
        let tex_h = tex_h_usize as f32;
        let available = ui.available_size();
        let fit_scale = if tex_w > 0.0 && tex_h > 0.0 {
            (available.x / tex_w).min(available.y / tex_h)
        } else {
            1.0
        };
        let s = fit_scale * self.zoom;
        let display_size = egui::vec2(tex_w * s, tex_h * s);

        let image_widget = egui::Image::new(egui::load::SizedTexture::new(tex_id, display_size));
        let response = ui.add(image_widget);

        if self.show_mesh_overlay && self.loaded_mesh.is_some() {
            if self.mesh_dirty || self.mesh_tex.is_none() {
                self.rebuild_mesh_texture(ctx, tex_w_usize, tex_h_usize);
            }
            if let Some(ref mesh_tex) = self.mesh_tex {
                let painter = ui.painter_at(response.rect);
                let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
                painter.image(mesh_tex.id(), response.rect, uv, egui::Color32::WHITE);
            }
        }

        let painter = ui.painter_at(response.rect);
        let label = match self.projection_mode {
            ProjectionMode::Mip => "3D MIP",
            ProjectionMode::Vr => "3D VR",
        };
        painter.text(
            response.rect.min + egui::vec2(OVERLAY_LABEL_INSET, OVERLAY_LABEL_INSET),
            egui::Align2::LEFT_TOP,
            label,
            egui::FontId::proportional(OVERLAY_LABEL_FONT_SIZE),
            OVERLAY_LABEL_COLOR,
        );

        response.context_menu(|ui| {
            ui.label("3D Projection");
            ui.separator();
            if ui
                .selectable_label(self.projection_mode == ProjectionMode::Mip, "MIP")
                .clicked()
            {
                self.projection_mode = ProjectionMode::Mip;
                self.mip_dirty = true;
                ui.close_menu();
            }
            if ui
                .selectable_label(self.projection_mode == ProjectionMode::Vr, "VR")
                .clicked()
            {
                self.projection_mode = ProjectionMode::Vr;
                self.mip_dirty = true;
                ui.close_menu();
            }
        });
    }

    pub(crate) fn rebuild_secondary_texture(
        &mut self,
        ctx: &egui::Context,
        axis: usize,
        slice_index: usize,
    ) {
        let (color_image, tex_name) = {
            let Some(vol) = &self.loaded_secondary else {
                return;
            };
            let wc = self
                .secondary_window_center
                .unwrap_or(DEFAULT_WINDOW_CENTER) as f64;
            let ww = self
                .secondary_window_width
                .unwrap_or(DEFAULT_WINDOW_WIDTH)
                .max(1.0) as f64;
            let wl = WindowLevel::new(wc, ww);
            let name = "slice_tex_secondary";
            let img = SliceRenderer::render_with_scratch(
                &mut self.render_buffer_pool,
                vol,
                axis,
                slice_index,
                wl,
                self.secondary_colormap,
            );
            let img = apply_to_image_into(&mut self.render_buffer_pool, &img, self.view_transform);
            (img, name)
        };
        self.secondary_texture =
            Some(ctx.load_texture(tex_name, color_image, egui::TextureOptions::LINEAR));
        self.secondary_texture_axis = axis;
        self.secondary_texture_slice = slice_index;
        self.secondary_texture_dirty = false;
    }
}
