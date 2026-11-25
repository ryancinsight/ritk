//! Clip the patient-space plane trace to the visible target image before rasterization.
use super::super::frame::RenderedView;
use super::{screen_coordinate, NativeViewport};
use crate::geometry::patient_stack::PlaneTrace;
use anyhow::Result;

pub(in crate::presentation::native_session) fn reference_segment(
    trace: &PlaneTrace,
    view: &RenderedView,
    viewport: NativeViewport,
) -> Result<Option<[(i32, i32); 2]>> {
    let im = viewport.image;
    let pane = viewport.panel;
    let x0 = im.x.max(pane.x);
    let y0 = im.y.max(pane.y);
    let x1 = (im.x + im.width).min(pane.x + pane.width);
    let y1 = (im.y + im.height).min(pane.y + pane.height);
    let x_pixels = [x0.ceil(), x1.ceil() - 1.];
    let y_pixels = [y0.ceil(), y1.ceil() - 1.];
    if x_pixels[1] < x_pixels[0] || y_pixels[1] < y_pixels[0] {
        return Ok(None);
    }
    let [w, h] = view.source_size;
    let [ow, oh] = view.transform.output_size(view.source_size);
    let size = [f64::from(u32::try_from(w)?), f64::from(u32::try_from(h)?)];
    let output = [f64::from(u32::try_from(ow)?), f64::from(u32::try_from(oh)?)];
    let corners = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
    let distances = corners.map(|[x, y]| {
        let source = view.transform.output_to_source_coordinates(
            [
                (x - im.x) / im.width * output[0],
                (y - im.y) / im.height * output[1],
            ],
            view.source_size,
        );
        trace.distance([source[0] - size[0] * 0.5, source[1] - size[1] * 0.5])
    });
    let Some([a, b]) = PlaneTrace::clip(corners, distances) else {
        return Ok(None);
    };
    let pixel = |[x, y]: [f64; 2]| -> Result<_> {
        Ok((
            screen_coordinate(x.clamp(x_pixels[0], x_pixels[1]), "reference column")?,
            screen_coordinate(y.clamp(y_pixels[0], y_pixels[1]), "reference row")?,
        ))
    };
    Ok(Some([pixel(a)?, pixel(b)?]))
}
