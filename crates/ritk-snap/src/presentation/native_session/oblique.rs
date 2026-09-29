//! Retained rendering and navigation state for the physical oblique panel.

use super::frame::window_level_for_app;
use super::session::{NativeViewerSession, ObliqueGesture};
use crate::app::SnapApp;
use crate::app::action_adapter::viewer_scroll_value;
use crate::geometry::affine::AffineTransform;
use crate::presentation::{PresentationEvent, PresentationFrame, PresentationSpacing};
use crate::render::{
    GrayscalePresentation, ResliceInterpolation, ResliceOrientation, ReslicePlane, map_scalar_value,
};
use crate::tools::interaction::{ToolState, ViewportOffset};
use crate::tools::kind::ToolKind;
use crate::ui::{zoom_from_drag_delta, zoom_from_scroll};
use anyhow::{Context, Result, anyhow};

const INITIAL_YAW_DEGREES: f64 = 25.0;
const INITIAL_PITCH_DEGREES: f64 = 15.0;
const ROTATION_STEP_DEGREES: f64 = 5.0;
pub(super) const NATIVE_WHEEL_DELTA: f64 = 120.0;

/// Retained RITK frame, reslice storage, and view-local navigation state.
#[derive(Debug)]
pub(super) struct ObliqueView {
    pub(super) frame: PresentationFrame,
    pub(super) plane: Option<ReslicePlane>,
    pub(super) orientation: ResliceOrientation,
    pub(super) zoom: f32,
    pub(super) pan_offset: ViewportOffset,
    scalar_pixels: Vec<f32>,
    rgba: Vec<u8>,
    rendered_revision: Option<u64>,
    dirty: bool,
}

impl ObliqueView {
    pub(super) fn new() -> Result<Self> {
        let frame = PresentationFrame::from_rgba_storage(1, 1, vec![0, 0, 0, 255])
            .context("construct empty native oblique frame")?;
        Ok(Self {
            frame,
            plane: None,
            orientation: ResliceOrientation::try_new(INITIAL_YAW_DEGREES, INITIAL_PITCH_DEGREES)
                .map_err(|error| anyhow!("construct initial oblique orientation: {error}"))?,
            zoom: 1.0,
            pan_offset: ViewportOffset::new(0.0, 0.0),
            scalar_pixels: Vec::new(),
            rgba: Vec::new(),
            rendered_revision: None,
            dirty: true,
        })
    }

    pub(super) fn render_if_stale(&mut self, app: &mut SnapApp) -> Result<()> {
        let Some(volume) = app.loaded.as_ref() else {
            if self.plane.is_some() || self.frame.width() != 1 || self.frame.height() != 1 {
                let frame = PresentationFrame::from_rgba_storage(1, 1, vec![0, 0, 0, 255])
                    .context("clear native oblique frame after closing the study")?;
                self.plane = None;
                self.frame = frame;
                self.rendered_revision = Some(app.visual_revision);
                self.dirty = false;
            }
            return Ok(());
        };

        let current_plane = self
            .plane
            .filter(|plane| plane.validate_source(volume).is_ok());
        if !self.dirty
            && self.rendered_revision == Some(app.visual_revision)
            && current_plane.is_some()
        {
            return Ok(());
        }

        let plane = match current_plane {
            Some(plane) => plane,
            None => {
                if let Some(stale) = self
                    .plane
                    .and_then(|plane| plane.validate_source(volume).err())
                {
                    app.status_message =
                        format!("Oblique plane recentered for the new source: {stale}");
                }
                let center = linked_patient_center(app, volume)?;
                ReslicePlane::oblique_at_patient(
                    volume,
                    center,
                    self.orientation,
                    ResliceInterpolation::Linear,
                )
                .map_err(|error| anyhow!("construct native oblique reslice plane: {error}"))?
            }
        };
        self.render_candidate(app, &plane, self.orientation)
    }

    fn render_candidate(
        &mut self,
        app: &mut SnapApp,
        plane: &ReslicePlane,
        orientation: ResliceOrientation,
    ) -> Result<()> {
        let volume = app
            .loaded
            .as_ref()
            .ok_or_else(|| anyhow!("cannot render an oblique plane without a loaded volume"))?;
        let dimensions = plane
            .compute_into(
                volume,
                crate::render::ProjectionStatistic::Maximum,
                &mut self.scalar_pixels,
            )
            .map_err(|error| anyhow!("compute native oblique scalar plane: {error}"))?;
        let display = GrayscalePresentation::for_volume(volume)
            .map_err(|error| anyhow!("validate oblique grayscale metadata: {error}"))?;
        let window_level = window_level_for_app(app);
        let byte_count = self
            .scalar_pixels
            .len()
            .checked_mul(4)
            .ok_or_else(|| anyhow!("native oblique RGBA byte count overflows usize"))?;
        self.rgba.resize(byte_count, 0);
        for (rgba, &value) in self.rgba.chunks_exact_mut(4).zip(&self.scalar_pixels) {
            rgba.copy_from_slice(&map_scalar_value(
                value,
                display,
                window_level,
                app.colormap,
            ));
        }
        let [row_spacing, column_spacing] = [
            vector_norm(plane.vertical_step()),
            vector_norm(plane.horizontal_step()),
        ];
        let spacing = PresentationSpacing::try_new(row_spacing, column_spacing)
            .context("validate native oblique pixel spacing")?;
        self.frame
            .replace_rgba_storage(
                u32::try_from(dimensions[0])
                    .map_err(|_| anyhow!("native oblique width exceeds u32"))?,
                u32::try_from(dimensions[1])
                    .map_err(|_| anyhow!("native oblique height exceeds u32"))?,
                spacing,
                &mut self.rgba,
            )
            .context("replace native oblique presentation frame")?;
        self.plane = Some(*plane);
        self.orientation = orientation;
        self.rendered_revision = Some(app.visual_revision);
        self.dirty = false;
        Ok(())
    }

    pub(super) fn rotate(&mut self, app: &mut SnapApp, yaw_delta: f64, pitch_delta: f64) -> bool {
        let Some(volume) = app.loaded.as_ref() else {
            return false;
        };
        let next_orientation = match self.orientation.rotated_by(yaw_delta, pitch_delta) {
            Ok(orientation) => orientation,
            Err(error) => {
                app.status_message = format!("Oblique rotation rejected: {error}");
                return false;
            }
        };
        let Some(plane) = self.plane else {
            app.status_message =
                "Oblique rotation is unavailable without a plane center".to_owned();
            return false;
        };
        let center = match plane_center(&plane) {
            Ok(center) => center,
            Err(error) => {
                app.status_message = format!("Oblique rotation is unavailable: {error}");
                return false;
            }
        };
        match ReslicePlane::oblique_at_patient(
            volume,
            center,
            next_orientation,
            ResliceInterpolation::Linear,
        ) {
            Ok(plane) => {
                if let Err(error) = self.render_candidate(app, &plane, next_orientation) {
                    app.status_message = format!("Oblique rotation rejected: {error:#}");
                    return false;
                }
                let cleared_length_start = clear_pending_length_start(app);
                let orientation_message = format!(
                    "Oblique orientation: yaw {:.0}°, pitch {:.0}°",
                    next_orientation.yaw_degrees(),
                    next_orientation.pitch_degrees()
                );
                app.status_message = if cleared_length_start {
                    format!("{orientation_message}; pending length start cleared")
                } else {
                    orientation_message
                };
                true
            }
            Err(error) => {
                app.status_message = format!("Oblique rotation rejected: {error}");
                false
            }
        }
    }

    pub(super) fn shift_depth(&mut self, app: &mut SnapApp, steps: f64) -> bool {
        let Some(volume) = app.loaded.as_ref() else {
            return false;
        };
        let Some(plane) = self.plane else {
            return false;
        };
        match plane.shifted_along_depth(volume, steps) {
            Ok(shifted) => {
                if let Err(error) = self.render_candidate(app, &shifted, self.orientation) {
                    app.status_message = format!("Oblique plane shift rejected: {error:#}");
                    return false;
                }
                let cleared_length_start = clear_pending_length_start(app);
                if cleared_length_start {
                    app.status_message =
                        "Oblique plane shifted; pending length start cleared".to_owned();
                }
                true
            }
            Err(error) => {
                app.status_message = format!("Oblique plane shift rejected: {error}");
                false
            }
        }
    }

    pub(super) fn set_pan(&mut self, x: f32, y: f32) {
        self.pan_offset = ViewportOffset::new(x, y);
    }
}

fn clear_pending_length_start(app: &mut SnapApp) -> bool {
    if matches!(&app.tool_state, ToolState::PatientLength1 { .. }) {
        app.tool_state = ToolState::Idle;
        true
    } else {
        false
    }
}

fn linked_patient_center(app: &SnapApp, volume: &crate::LoadedVolume) -> Result<[f64; 3]> {
    let voxel = if let Some(cursor) = app.linked_cursor {
        let voxel = cursor.voxel();
        [
            f64::from(u32::try_from(voxel[0]).context("linked cursor depth exceeds u32")?),
            f64::from(u32::try_from(voxel[1]).context("linked cursor row exceeds u32")?),
            f64::from(u32::try_from(voxel[2]).context("linked cursor column exceeds u32")?),
        ]
    } else {
        [
            midpoint_voxel(volume.shape[0], "depth")?,
            midpoint_voxel(volume.shape[1], "row")?,
            midpoint_voxel(volume.shape[2], "column")?,
        ]
    };
    let affine = AffineTransform::from_parts(volume.origin, volume.direction, volume.spacing)
        .context("validate native oblique source affine")?;
    Ok(affine.voxel_to_patient(voxel))
}

fn midpoint_voxel(size: usize, dimension: &str) -> Result<f64> {
    let last = size
        .checked_sub(1)
        .ok_or_else(|| anyhow!("native source {dimension} dimension is empty"))?;
    Ok(f64::from(
        u32::try_from(last).with_context(|| format!("native source {dimension} exceeds u32"))?,
    ) * 0.5)
}

fn plane_center(plane: &ReslicePlane) -> Result<[f64; 3]> {
    let dimensions = plane.dimensions();
    let column = f64::from(
        u32::try_from(dimensions[0].saturating_sub(1))
            .context("oblique width exceeds the native coordinate contract")?,
    ) * 0.5;
    let row = f64::from(
        u32::try_from(dimensions[1].saturating_sub(1))
            .context("oblique height exceeds the native coordinate contract")?,
    ) * 0.5;
    plane
        .patient_at_pixel([column, row])
        .map(|point| point.coordinates())
        .context("map oblique output center to patient coordinates")
}

fn vector_norm(vector: [f64; 3]) -> f64 {
    vector
        .into_iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
}

impl NativeViewerSession {
    pub(super) fn reduce_oblique_events(
        &mut self,
        events: &[PresentationEvent],
        selected_oblique: bool,
    ) -> Result<bool> {
        let mut repaint = false;
        for event in events {
            match event {
                PresentationEvent::KeyDown {
                    virtual_key,
                    repeated: false,
                    ..
                } if selected_oblique => {
                    let (yaw, pitch) = match *virtual_key {
                        0x25 => (-ROTATION_STEP_DEGREES, 0.0),
                        0x27 => (ROTATION_STEP_DEGREES, 0.0),
                        0x26 => (0.0, ROTATION_STEP_DEGREES),
                        0x28 => (0.0, -ROTATION_STEP_DEGREES),
                        _ => continue,
                    };
                    if let Some(oblique) = self.oblique.as_mut() {
                        repaint |= oblique.rotate(&mut self.app, yaw, pitch);
                    }
                }
                PresentationEvent::PointerWheel {
                    x,
                    y,
                    delta_y,
                    modifiers,
                    ..
                } if selected_oblique => {
                    let Some(viewport) = self.oblique_viewport else {
                        continue;
                    };
                    if viewport
                        .map(crate::presentation::ViewportPoint::new(*x, *y))
                        .is_none()
                        || *delta_y == 0.0
                    {
                        continue;
                    }
                    let Some(oblique) = self.oblique.as_mut() else {
                        continue;
                    };
                    if modifiers.ctrl() || modifiers.meta() {
                        let scroll = viewer_scroll_value(*delta_y)
                            .map_err(|error| anyhow!("map oblique zoom wheel: {error}"))?;
                        let zoom = zoom_from_scroll(oblique.zoom, scroll);
                        repaint |= zoom != oblique.zoom;
                        oblique.zoom = zoom;
                    } else {
                        repaint |=
                            oblique.shift_depth(&mut self.app, -*delta_y / NATIVE_WHEEL_DELTA);
                    }
                }
                PresentationEvent::PointerDown { x, y, button } if self.active_oblique => {
                    if *button != crate::presentation::PointerButton::Left {
                        continue;
                    }
                    self.oblique_gesture =
                        self.oblique
                            .as_ref()
                            .and_then(|oblique| match self.app.active_tool {
                                ToolKind::Pan => Some(ObliqueGesture::Pan { last: [*x, *y] }),
                                ToolKind::Zoom => Some(ObliqueGesture::Zoom {
                                    start_y: *y,
                                    initial_zoom: oblique.zoom,
                                }),
                                _ => None,
                            });
                }
                PresentationEvent::PointerMove { x, y } if self.active_oblique => {
                    match self.oblique_gesture {
                        Some(ObliqueGesture::Pan { last }) => {
                            let delta = [screen_delta(*x, last[0])?, screen_delta(*y, last[1])?];
                            if let Some(oblique) = self.oblique.as_mut() {
                                let offset = oblique.pan_offset;
                                oblique.set_pan(
                                    checked_screen_f32(f64::from(offset.x()) + delta[0])?,
                                    checked_screen_f32(f64::from(offset.y()) + delta[1])?,
                                );
                                repaint |= delta != [0.0, 0.0];
                            }
                            self.oblique_gesture = Some(ObliqueGesture::Pan { last: [*x, *y] });
                        }
                        Some(ObliqueGesture::Zoom {
                            start_y,
                            initial_zoom,
                        }) => {
                            let delta = checked_screen_f32(*y - start_y)?;
                            if let Some(oblique) = self.oblique.as_mut() {
                                let zoom = zoom_from_drag_delta(initial_zoom, delta);
                                repaint |= zoom != oblique.zoom;
                                oblique.zoom = zoom;
                            }
                        }
                        None => {}
                    }
                }
                PresentationEvent::PointerUp { .. } | PresentationEvent::PointerCancel { .. } => {
                    self.oblique_gesture = None;
                }
                PresentationEvent::FocusLost | PresentationEvent::CloseRequested => {
                    self.oblique_gesture = None;
                }
                _ => {}
            }
        }
        Ok(repaint)
    }
}

fn screen_delta(current: f64, previous: f64) -> Result<f64> {
    let delta = current - previous;
    if !current.is_finite() || !previous.is_finite() || !delta.is_finite() {
        return Err(anyhow!("oblique pointer displacement is not finite"));
    }
    Ok(delta)
}

fn checked_screen_f32(value: f64) -> Result<f32> {
    if !value.is_finite() || value < -f64::from(f32::MAX) || value > f64::from(f32::MAX) {
        return Err(anyhow!("oblique viewport displacement exceeds f32 range"));
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "native client coordinates and pan displacement are bounded by the signed surface dimensions"
    )]
    Ok(value as f32)
}
