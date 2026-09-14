//! Eframe shell state and its GUI-owned resource cache.
//!
//! [`SnapApp`] contains viewer state and RITK presentation contracts. This
//! module owns eframe texture handles and the shell implementation so those
//! resources cannot enter native or browser-neutral viewer state.

use super::slice_ops::CineTick;
use super::state::SnapApp;
use crate::render::RenderBufferPool;
use crate::ui::ViewTransform;
use std::ops::{Deref, DerefMut};

#[cfg(not(target_arch = "wasm32"))]
const CINE_REPAINT_INTERVAL: std::time::Duration = std::time::Duration::from_millis(8);

/// Cached RT-DOSE overlay texture for one eframe viewport axis.
pub(crate) struct RtDoseOverlayCacheEntry {
    pub(crate) slice_idx: usize,
    pub(crate) vol_shape: [usize; 3],
    pub(crate) dose_dims: [usize; 3],
    pub(crate) opacity_alpha: u8,
    pub(crate) view_transform: ViewTransform,
    pub(crate) texture: egui::TextureHandle,
}

/// Eframe render resources kept outside the host-neutral viewer state.
pub(crate) struct EguiRenderState {
    /// Scratch storage reused by eframe texture rebuilds.
    pub(crate) buffer_pool: RenderBufferPool,
    pub(crate) texture: Option<egui::TextureHandle>,
    pub(crate) secondary_texture: Option<egui::TextureHandle>,
    pub(crate) coronal_tex: Option<egui::TextureHandle>,
    pub(crate) sagittal_tex: Option<egui::TextureHandle>,
    pub(crate) mip_tex: Option<egui::TextureHandle>,
    pub(crate) mesh_tex: Option<egui::TextureHandle>,
    pub(crate) rt_dose_overlay_cache: [Option<RtDoseOverlayCacheEntry>; 3],
    /// Revision of [`SnapApp`] represented by the retained textures.
    pub(crate) visual_revision: u64,
    /// Secondary texture key `(axis, slice_index)` represented by the cache.
    pub(crate) secondary_texture_key: Option<(usize, usize)>,
}

impl Default for EguiRenderState {
    fn default() -> Self {
        Self {
            buffer_pool: RenderBufferPool::default(),
            texture: None,
            secondary_texture: None,
            coronal_tex: None,
            sagittal_tex: None,
            mip_tex: None,
            mesh_tex: None,
            rt_dose_overlay_cache: std::array::from_fn(|_| None),
            visual_revision: 0,
            secondary_texture_key: None,
        }
    }
}

impl EguiRenderState {
    pub(crate) fn invalidate(&mut self, revision: u64) {
        self.texture = None;
        self.secondary_texture = None;
        self.coronal_tex = None;
        self.sagittal_tex = None;
        self.mip_tex = None;
        self.mesh_tex = None;
        self.secondary_texture_key = None;
        self.clear_rt_dose_overlay_cache();
        self.visual_revision = revision;
    }

    pub(crate) fn clear_rt_dose_overlay_cache(&mut self) {
        self.rt_dose_overlay_cache = std::array::from_fn(|_| None);
    }
}

/// Eframe application wrapper around the RITK viewer state.
pub(crate) struct EguiApp {
    pub(crate) app: SnapApp,
    pub(crate) render: EguiRenderState,
}

impl EguiApp {
    /// Wrap a host-neutral viewer state in the eframe shell.
    pub(crate) fn new(app: SnapApp) -> Self {
        Self {
            app,
            render: EguiRenderState::default(),
        }
    }

    fn reconcile_render_resources(&mut self) {
        if self.render.visual_revision != self.visual_revision {
            self.render.invalidate(self.visual_revision);
        }
        if self.loaded.is_none() {
            self.render.texture = None;
            self.render.coronal_tex = None;
            self.render.sagittal_tex = None;
            self.render.mip_tex = None;
        }
        if self.loaded_secondary.is_none() {
            self.render.secondary_texture = None;
            self.render.secondary_texture_key = None;
        }
        if self.loaded_mesh.is_none() {
            self.render.mesh_tex = None;
        }
        if self.rt_dose.is_none() {
            self.render.clear_rt_dose_overlay_cache();
        }
    }

    /// Load RT-DOSE through the viewer state and invalidate shell textures.
    pub(crate) fn load_rt_dose_file(&mut self, path: std::path::PathBuf) {
        self.app.load_rt_dose_file(path);
        self.render.clear_rt_dose_overlay_cache();
    }

    fn tick_cine(&mut self, ctx: &egui::Context) {
        let now_seconds = ctx.input(|input| input.time);
        match self.app.tick_cine_at(now_seconds) {
            CineTick::Inactive => {}
            CineTick::Waiting => ctx.request_repaint_after(CINE_REPAINT_INTERVAL),
            CineTick::Advanced(_) => ctx.request_repaint(),
        }
    }
}

impl Deref for EguiApp {
    type Target = SnapApp;

    fn deref(&self) -> &Self::Target {
        &self.app
    }
}

impl DerefMut for EguiApp {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.app
    }
}

impl eframe::App for EguiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.handle_dropped_inputs(ctx);
        self.process_pending_loads();
        self.poll_load_tasks();
        self.poll_pacs_worker();
        self.reconcile_render_resources();

        self.tick_cine(ctx);
        self.consume_global_shortcuts(ctx);
        self.show_menu_bar(ctx);
        self.show_ribbon_toolbar(ctx);
        self.show_left_panel(ctx);
        self.show_bottom_bar(ctx);
        self.show_aux_windows(ctx);

        if self.compare_side_by_side {
            self.show_central_panel_compare(ctx);
        } else if self.multi_planar {
            self.show_central_panel_multi(ctx);
        } else if self.dual_plane {
            self.show_central_panel_dual(ctx);
        } else {
            self.show_central_panel_single(ctx);
        }
    }
}
