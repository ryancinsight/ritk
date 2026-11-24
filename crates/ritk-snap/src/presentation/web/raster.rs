//! Configure the backing store on the first context acquisition. A later
//! `getContext` call returns the existing context and ignores its options.
use std::io;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::HtmlCanvasElement;

pub(super) fn prepare(id: &str) -> io::Result<()> {
    let document = web_sys::window()
        .and_then(|window| window.document())
        .ok_or_else(|| io::Error::other("browser document is unavailable"))?;
    let canvas = document
        .get_element_by_id(id)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "canvas element is absent"))?
        .dyn_into::<HtmlCanvasElement>()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "element is not a canvas"))?;
    let options = js_sys::Object::new();
    js_sys::Reflect::set(&options, &"willReadFrequently".into(), &JsValue::TRUE)
        .map_err(|error| io::Error::other(format!("setting canvas readback option: {error:?}")))?;
    canvas
        .get_context_with_context_options("2d", &options)
        .map_err(|error| io::Error::other(format!("opening raster canvas: {error:?}")))?
        .ok_or_else(|| io::Error::other("Canvas2D context is unavailable"))?;
    Ok(())
}
