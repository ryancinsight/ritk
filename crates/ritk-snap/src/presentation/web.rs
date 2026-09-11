//! Browser canvas adapter for RITK's format-neutral presentation frame.

use super::PresentationFrame;
use metis_web::{CanvasFrame, CanvasSurface};
use std::io;

/// Presents RITK-owned pixels through a Metis browser canvas.
pub struct WebCanvasPresenter {
    surface: CanvasSurface,
}

impl WebCanvasPresenter {
    /// Resolves a canvas in the current browser document.
    ///
    /// # Errors
    /// Returns a typed I/O error when the browser document, canvas element, or
    /// two-dimensional rendering context is unavailable.
    pub fn from_canvas_id(id: &str) -> io::Result<Self> {
        Ok(Self {
            surface: CanvasSurface::from_current_document(id)?,
        })
    }

    /// Returns the identifier of the resolved browser canvas.
    #[must_use]
    pub fn canvas_id(&self) -> String {
        self.surface.id()
    }

    /// Presents one RITK display frame without copying it in the Rust host.
    ///
    /// RITK retains ownership of the frame and its display semantics; Metis
    /// receives only the borrowed RGBA view at this boundary.
    ///
    /// # Errors
    /// Returns a typed I/O error when the frame violates the provider bounds
    /// or the browser rejects the upload.
    pub fn present(&self, frame: &PresentationFrame) -> io::Result<()> {
        self.surface.present(frame)
    }
}

impl CanvasFrame for PresentationFrame {
    fn width(&self) -> u32 {
        PresentationFrame::width(self)
    }

    fn height(&self) -> u32 {
        PresentationFrame::height(self)
    }

    fn rgba(&self) -> &[u8] {
        PresentationFrame::rgba(self)
    }
}
