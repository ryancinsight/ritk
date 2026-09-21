//! Browser launch and viewer-control entrypoints.

/// Start the RITK browser canvas workflow through the Métis host.
///
/// This asynchronous entrypoint is exported only for `wasm32` and keeps the
/// original JavaScript bootstrap contract. It delegates to
/// [`start_web_canvas`], so the browser path receives only the format-neutral
/// Métis canvas and bounded file handoff; RITK retains DICOM decoding and
/// viewer state.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn start_web(canvas_id: String) -> Result<(), wasm_bindgen::JsValue> {
    use wasm_bindgen_futures::JsFuture;

    crate::app::start_web_canvas(canvas_id)?;

    // Preserve the async bootstrap contract so existing JavaScript callers can
    // await startup while the bounded browser task begins its first tick.
    JsFuture::from(js_sys::Promise::resolve(&wasm_bindgen::JsValue::UNDEFINED))
        .await
        .map_err(|e| {
            wasm_bindgen::JsValue::from_str(&format!("web startup promise failed: {e:?}"))
        })?;

    Ok(())
}

/// Start the RITK browser canvas workflow with Métis and Moirai.
///
/// The workflow receives bounded browser file bytes from Métis, lets RITK
/// classify and decode them, and presents the selected RITK frame through the
/// named HTML5 canvas. [`start_web_orthogonal_canvases`] presents the three
/// RITK orthogonal frames through three named canvases. [`start_web`] is the
/// asynchronous JavaScript-compatible wrapper for this single-canvas path.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn start_web_canvas(canvas_id: String) -> Result<(), wasm_bindgen::JsValue> {
    crate::app::start_web_canvas(canvas_id)
}

/// Start the RITK browser canvas workflow through an explicit WebGPU surface.
///
/// The future resolves after WebGPU adapter/device setup and listener
/// registration complete. Setup errors are returned to JavaScript; the
/// existing raster entrypoint is never selected implicitly.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn start_web_canvas_gpu(canvas_id: String) -> Result<(), wasm_bindgen::JsValue> {
    crate::app::start_web_canvas_gpu(canvas_id).await
}

/// Start the RITK browser canvas workflow with three orthogonal views.
///
/// The identifiers are ordered axial, coronal, sagittal. RITK owns the
/// decoded volume, slice selection and display semantics; Métis owns only the
/// browser canvases and bounded file handoff.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn start_web_orthogonal_canvases(
    axial_id: String,
    coronal_id: String,
    sagittal_id: String,
) -> Result<(), wasm_bindgen::JsValue> {
    crate::app::start_web_orthogonal_canvases([axial_id, coronal_id, sagittal_id])
}

/// Start the RITK orthogonal browser workflow through explicit WebGPU surfaces.
///
/// Identifiers are ordered axial, coronal, sagittal. The future rejects when
/// any canvas cannot acquire WebGPU or register its bounded input listeners.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn start_web_orthogonal_canvases_gpu(
    axial_id: String,
    coronal_id: String,
    sagittal_id: String,
) -> Result<(), wasm_bindgen::JsValue> {
    crate::app::start_web_orthogonal_canvases_gpu([axial_id, coronal_id, sagittal_id]).await
}

/// Start the browser workflow with three interactive planes and one
/// display-only scalar projection.
///
/// Canvas identifiers are ordered axial, coronal, sagittal, projection.
/// `projection` is `0` for maximum, `1` for minimum and `2` for average.
/// The value is validated before the viewer mounts.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn start_web_orthogonal_canvases_with_projection(
    axial_id: String,
    coronal_id: String,
    sagittal_id: String,
    projection_id: String,
    projection: f64,
) -> Result<(), wasm_bindgen::JsValue> {
    crate::app::start_web_orthogonal_canvases_with_projection(
        [axial_id, coronal_id, sagittal_id, projection_id],
        projection,
    )
}

/// Start the four-canvas browser workflow with explicit WebGPU surfaces.
///
/// Setup errors are returned to JavaScript; the raster provider is never
/// selected implicitly. Canvas identifiers and statistic indices follow
/// [`start_web_orthogonal_canvases_with_projection`].
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn start_web_orthogonal_canvases_gpu_with_projection(
    axial_id: String,
    coronal_id: String,
    sagittal_id: String,
    projection_id: String,
    projection: f64,
) -> Result<(), wasm_bindgen::JsValue> {
    crate::app::start_web_orthogonal_canvases_gpu_with_projection(
        [axial_id, coronal_id, sagittal_id, projection_id],
        projection,
    )
    .await
}

/// Select an exact zero-based slice on one browser viewer axis.
///
/// Axes are `0` axial, `1` coronal and `2` sagittal. A successful change
/// invalidates the cached frames; the next animation frame renders the new
/// slice and republishes its `data-ritk-*` semantics.
///
/// # Errors
///
/// Returns a JavaScript error value when no viewer or study is available, the
/// viewer is handling another callback, or the axis/index is out of range.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn select_web_slice(axis: f64, index: f64) -> Result<(), wasm_bindgen::JsValue> {
    crate::app::select_web_slice(axis, index)
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}

/// Toggle host-neutral cine playback for the loaded browser study.
///
/// The next browser animation frame establishes the timing anchor and
/// republishes `data-ritk-cine-enabled` on every RITK canvas.
///
/// # Errors
///
/// Returns a JavaScript error when the viewer is not mounted, another browser
/// callback owns it, or no study has been loaded.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn toggle_web_cine() -> Result<bool, wasm_bindgen::JsValue> {
    crate::app::toggle_web_cine()
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}

/// Toggle the host-neutral linked MPR crosshair for the loaded browser study.
///
/// The three RITK canvases publish the new visibility and linked voxel state
/// on the next animation frame. DICOM data and cursor reduction remain inside
/// RITK; the browser consumer only chooses whether to draw the overlay.
///
/// # Errors
///
/// Returns a JavaScript error when the viewer is not mounted, another callback
/// owns it, or no study has been loaded.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn toggle_web_crosshair() -> Result<bool, wasm_bindgen::JsValue> {
    crate::app::toggle_web_crosshair()
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}

/// Set the exact bounded cine playback rate for the loaded browser study.
///
/// `rate` must be a finite integral value from 1 through 60 frames per
/// second. Invalid values are rejected before viewer state changes.
///
/// # Errors
///
/// Returns a JavaScript error when the value is invalid, the viewer is not
/// mounted, another browser callback owns it, or no study has been loaded.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn set_web_cine_rate(rate: f64) -> Result<bool, wasm_bindgen::JsValue> {
    crate::app::set_web_cine_rate(rate)
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}

/// Select one loaded-study interaction tool from the RITK browser table.
///
/// The index is a finite integer in the range reported by
/// [`web_tool_count`]. Selecting a tool clears any in-progress gesture while
/// leaving the decoded study and rendered pixels unchanged.
///
/// # Errors
///
/// Returns a JavaScript error when the value is invalid, the viewer is not
/// mounted, another callback owns it, no study is loaded, or the index is
/// outside the table.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn select_web_tool(index: f64) -> Result<bool, wasm_bindgen::JsValue> {
    crate::app::select_web_tool(index)
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}

/// Return the number of interaction tools exposed to the browser palette.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn web_tool_count() -> Result<usize, wasm_bindgen::JsValue> {
    crate::app::web_tool_count()
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}

/// Return one interaction-tool label for browser palette construction.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn web_tool_name(index: f64) -> Result<String, wasm_bindgen::JsValue> {
    crate::app::web_tool_name(index)
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}

/// Apply one exact loaded-modality window/level preset to every browser view.
///
/// The index is validated as a finite non-negative integer against the table
/// selected from the loaded DICOM modality. A successful change invalidates
/// all retained frames; the next animation frame renders the new intensity
/// mapping and publishes the updated window attributes.
///
/// # Errors
///
/// Returns a JavaScript error when no viewer or study is available, the viewer
/// is handling another callback, or the index is outside the active preset
/// table.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn set_web_window_preset(index: f64) -> Result<(), wasm_bindgen::JsValue> {
    crate::app::set_web_window_preset(index)
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}

/// Return the number of window/level presets for the loaded modality.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn web_window_preset_count() -> Result<usize, wasm_bindgen::JsValue> {
    crate::app::web_window_preset_count()
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}

/// Return one window/level preset name for the loaded modality.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn web_window_preset_name(index: f64) -> Result<String, wasm_bindgen::JsValue> {
    crate::app::web_window_preset_name(index)
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}

/// Stop the RITK browser canvas workflow and release its browser task.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn stop_web_canvas() {
    crate::app::stop_web_canvas();
}

/// Return the number of browser canvas listener guards retained by RITK.
///
/// The count is zero after [`stop_web_canvas`] returns. A mounted single-canvas
/// viewer reports one provider input set; the orthogonal viewer reports three.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
#[must_use]
pub fn web_canvas_listener_count() -> usize {
    crate::app::web_canvas_listener_count()
}
