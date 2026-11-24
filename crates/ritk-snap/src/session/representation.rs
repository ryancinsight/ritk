use super::{
    Annotation, NamedColorMap, SessionFormat, SidebarTab, StudySource, ToolKind,
    ViewerSessionSnapshot, ViewerState,
};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
#[serde(
    rename = "ViewerSessionSnapshot",
    expecting = "struct ViewerSessionSnapshot"
)]
pub(super) struct ViewerSessionRepresentation<S, A> {
    #[serde(default = "default_session_format_version")]
    format: u8,
    source: Option<S>,
    viewer_state: ViewerState,
    colormap: NamedColorMap,
    axis: usize,
    active_tool: ToolKind,
    multi_planar: bool,
    show_overlay: bool,
    show_crosshair: bool,
    show_rt_struct_overlay: bool,
    #[serde(default)]
    show_rt_dose_overlay: bool,
    #[serde(default = "default_rt_dose_opacity")]
    rt_dose_opacity: f32,
    show_series_browser: bool,
    sidebar_tab: SidebarTab,
    coronal_slice: usize,
    sagittal_slice: usize,
    pan_offset: [f32; 2],
    zoom: f32,
    cine_enabled: bool,
    cine_fps: f32,
    #[serde(default)]
    annotations: A,
}

impl<'a> From<&'a ViewerSessionSnapshot>
    for ViewerSessionRepresentation<&'a StudySource, &'a [Annotation]>
{
    fn from(snapshot: &'a ViewerSessionSnapshot) -> Self {
        let ViewerSessionSnapshot {
            format: _,
            source,
            viewer_state,
            colormap,
            axis,
            active_tool,
            multi_planar,
            show_overlay,
            show_crosshair,
            show_rt_struct_overlay,
            show_rt_dose_overlay,
            rt_dose_opacity,
            show_series_browser,
            sidebar_tab,
            coronal_slice,
            sagittal_slice,
            pan_offset,
            zoom,
            cine_enabled,
            cine_fps,
            annotations,
        } = snapshot;
        Self {
            format: session_format_version(annotations),
            source: source.as_ref(),
            viewer_state: *viewer_state,
            colormap: *colormap,
            axis: *axis,
            active_tool: *active_tool,
            multi_planar: *multi_planar,
            show_overlay: *show_overlay,
            show_crosshair: *show_crosshair,
            show_rt_struct_overlay: *show_rt_struct_overlay,
            show_rt_dose_overlay: *show_rt_dose_overlay,
            rt_dose_opacity: *rt_dose_opacity,
            show_series_browser: *show_series_browser,
            sidebar_tab: *sidebar_tab,
            coronal_slice: *coronal_slice,
            sagittal_slice: *sagittal_slice,
            pan_offset: *pan_offset,
            zoom: *zoom,
            cine_enabled: *cine_enabled,
            cine_fps: *cine_fps,
            annotations: annotations.as_slice(),
        }
    }
}

impl TryFrom<ViewerSessionRepresentation<StudySource, Vec<Annotation>>> for ViewerSessionSnapshot {
    type Error = &'static str;

    fn try_from(
        representation: ViewerSessionRepresentation<StudySource, Vec<Annotation>>,
    ) -> Result<Self, Self::Error> {
        let ViewerSessionRepresentation {
            format,
            source,
            viewer_state,
            colormap,
            axis,
            active_tool,
            multi_planar,
            show_overlay,
            show_crosshair,
            show_rt_struct_overlay,
            show_rt_dose_overlay,
            rt_dose_opacity,
            show_series_browser,
            sidebar_tab,
            coronal_slice,
            sagittal_slice,
            pan_offset,
            zoom,
            cine_enabled,
            cine_fps,
            annotations,
        } = representation;
        if format <= 3 && session_format_version(&annotations) == 4 {
            return Err("pixel-length annotations require viewer session format 4");
        }
        let format = match format {
            4 => SessionFormat,
            version => SessionFormat::try_from(version)?,
        };
        Ok(Self {
            format,
            source,
            viewer_state,
            colormap,
            axis,
            active_tool,
            multi_planar,
            show_overlay,
            show_crosshair,
            show_rt_struct_overlay,
            show_rt_dose_overlay,
            rt_dose_opacity,
            show_series_browser,
            sidebar_tab,
            coronal_slice,
            sagittal_slice,
            pan_offset,
            zoom,
            cine_enabled,
            cine_fps,
            annotations,
        })
    }
}

fn default_session_format_version() -> u8 {
    1
}

fn session_format_version(annotations: &[Annotation]) -> u8 {
    if annotations
        .iter()
        .any(|annotation| matches!(annotation, Annotation::PixelLength { .. }))
    {
        4
    } else {
        3
    }
}

fn default_rt_dose_opacity() -> f32 {
    0.5
}
