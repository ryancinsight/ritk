//! ritk-snap viewer application state.
//!
//! Owns the top-level `SnapApp` struct. All
//! domain logic (intensity mapping, slice extraction, annotation computation)
//! lives in the `render` and `tools` sub-modules; this module wires events and
//! drives state transitions. The format-neutral presentation module supplies
//! the Métis native host without moving DICOM or viewer state into that host.

#[cfg_attr(
    not(any(windows, target_arch = "wasm32", test)),
    expect(
        dead_code,
        reason = "only the Windows and browser hosts drive viewer input"
    )
)]
pub(crate) mod action_adapter;
#[cfg(any(target_arch = "wasm32", test))]
mod browser_cine;
#[cfg(any(target_arch = "wasm32", test))]
mod browser_crosshair;
#[cfg(any(target_arch = "wasm32", test))]
mod browser_geometry;
#[cfg(any(target_arch = "wasm32", test))]
mod browser_projection;
#[cfg(any(target_arch = "wasm32", test))]
mod browser_semantics;
#[cfg(any(target_arch = "wasm32", test))]
mod browser_slice_selection;
#[cfg(any(target_arch = "wasm32", test))]
mod browser_tool;
#[cfg(any(target_arch = "wasm32", test))]
mod browser_window_preset;
#[cfg(not(target_arch = "wasm32"))]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "ratchet RITK-SNAP-NATIVE-EXPORT-001")
)]
pub(crate) mod clinical_distribution;
#[cfg(not(target_arch = "wasm32"))]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "ratchet RITK-SNAP-NATIVE-FILTERS-001")
)]
mod filter;
#[cfg(not(target_arch = "wasm32"))]
#[expect(dead_code, reason = "ratchet RITK-SNAP-NATIVE-EXPORT-001")]
mod io_ops;
#[cfg(not(target_arch = "wasm32"))]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "ratchet RITK-SNAP-NATIVE-LOAD-001")
)]
mod load_tasks;
#[cfg(not(target_arch = "wasm32"))]
#[expect(dead_code, reason = "ratchet RITK-SNAP-NATIVE-VR-001")]
mod mesh_ops;
#[cfg(all(not(target_arch = "wasm32"), test))]
mod oblique_viewport;
#[cfg(not(target_arch = "wasm32"))]
#[expect(dead_code, reason = "ratchet RITK-SNAP-NATIVE-PACS-001")]
pub(crate) mod pacs_ops;
#[cfg_attr(
    not(any(windows, target_arch = "wasm32", test)),
    expect(
        dead_code,
        reason = "only the Windows and browser hosts drive viewer input"
    )
)]
pub(crate) mod pointer_bindings;
#[cfg_attr(
    not(any(windows, target_arch = "wasm32", test)),
    expect(
        dead_code,
        reason = "only the Windows and browser hosts drive viewer input"
    )
)]
mod pointer_ops;
#[cfg(any(target_arch = "wasm32", windows, test))]
mod presentation_snapshot;
#[cfg(not(target_arch = "wasm32"))]
#[expect(dead_code, reason = "ratchet RITK-SNAP-NATIVE-RT-001")]
mod rt_overlay;
#[cfg(not(target_arch = "wasm32"))]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "ratchet RITK-SNAP-NATIVE-RT-001")
)]
mod rt_struct_export;
#[cfg_attr(
    not(any(windows, target_arch = "wasm32", test)),
    expect(
        dead_code,
        reason = "only the Windows and browser hosts drive viewer input"
    )
)]
mod screen_image_geometry;
#[cfg(not(target_arch = "wasm32"))]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "ratchet RITK-SNAP-NATIVE-EXPORT-001")
)]
mod session_state;
#[cfg_attr(
    not(any(windows, target_arch = "wasm32", test)),
    expect(
        dead_code,
        reason = "only the Windows and browser hosts drive viewer input"
    )
)]
mod slice_ops;
pub(crate) mod state;
#[cfg(not(target_arch = "wasm32"))]
#[expect(dead_code, reason = "ratchet RITK-SNAP-NATIVE-EXPORT-001")]
mod surface_export;
#[cfg_attr(
    not(any(windows, target_arch = "wasm32", test)),
    expect(
        dead_code,
        reason = "only the Windows and browser hosts drive viewer input"
    )
)]
pub(crate) mod viewer_viewport;
#[cfg(not(target_arch = "wasm32"))]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "ratchet RITK-SNAP-NATIVE-LOAD-001")
)]
mod volume_input;
#[cfg(not(target_arch = "wasm32"))]
#[expect(dead_code, reason = "ratchet RITK-SNAP-NATIVE-OPEN-001")]
mod volume_ops;
mod volume_state;
#[cfg_attr(
    not(any(test, target_arch = "wasm32")),
    expect(dead_code, reason = "ratchet RITK-SNAP-NATIVE-DISPLAY-001")
)]
mod window_level;

#[cfg(target_arch = "wasm32")]
mod browser_canvas;
#[cfg(target_arch = "wasm32")]
mod browser_input;
#[cfg(target_arch = "wasm32")]
mod web_render;
#[cfg(target_arch = "wasm32")]
mod web_responsive;
#[cfg(target_arch = "wasm32")]
mod web_surface;
#[cfg(target_arch = "wasm32")]
mod web_viewer;

#[cfg(test)]
mod tests;

#[cfg(not(target_arch = "wasm32"))]
#[cfg(windows)]
pub(crate) use slice_ops::CineTick;
pub(crate) use state::SnapApp;

#[cfg(target_arch = "wasm32")]
pub(crate) use web_viewer::{
    select_web_slice, select_web_tool, set_web_cine_rate, set_web_window_preset, start_web_canvas,
    start_web_canvas_gpu, start_web_orthogonal_canvases, start_web_orthogonal_canvases_gpu,
    start_web_orthogonal_canvases_gpu_with_projection,
    start_web_orthogonal_canvases_with_projection, start_web_responsive_canvases, stop_web_canvas,
    toggle_web_cine, toggle_web_crosshair, web_canvas_listener_count, web_tool_count,
    web_tool_name, web_window_preset_count, web_window_preset_name,
};
