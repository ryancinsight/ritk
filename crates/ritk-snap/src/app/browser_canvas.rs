//! Browser canvas state and presentation metadata for the RITK viewer.

use super::browser_geometry::PhysicalCanvasAspect;
use super::browser_semantics::BrowserCanvasSemantics;
use crate::presentation::{PresentationFrame, WebCanvasPresenter};
use moirai_pal::wasm::{WebDocument, WebElement};

pub(super) struct BrowserCanvas {
    presenter: WebCanvasPresenter,
    element: WebElement,
    last_semantics: Option<BrowserCanvasSemantics>,
    last_physical_aspect: Option<PhysicalCanvasAspect>,
    frame_generation: u64,
}

impl BrowserCanvas {
    pub(super) fn from_id(id: &str) -> std::io::Result<Self> {
        let presenter = WebCanvasPresenter::from_canvas_id_with_input(id)?;
        Self::from_presenter(id, presenter)
    }

    pub(super) async fn from_id_gpu(id: &str) -> std::io::Result<Self> {
        let presenter = WebCanvasPresenter::from_canvas_id_gpu_with_input(id).await?;
        Self::from_presenter(id, presenter)
    }

    fn from_presenter(id: &str, presenter: WebCanvasPresenter) -> std::io::Result<Self> {
        let document = WebDocument::current()?;
        let element = document.get_element_by_id(id).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "browser canvas element disappeared during setup",
            )
        })?;
        // Keyboard events target the focused canvas; make the retained
        // presentation surface keyboard-focusable at the RITK boundary.
        element.set_attribute("tabindex", "0")?;
        Ok(Self {
            presenter,
            element,
            last_semantics: None,
            last_physical_aspect: None,
            frame_generation: 0,
        })
    }

    pub(super) fn listener_count(&self) -> usize {
        self.presenter.listener_count()
    }

    pub(super) fn present(&self, frame: &PresentationFrame) -> std::io::Result<()> {
        self.presenter.present(frame)
    }

    pub(super) fn take_events(
        &self,
    ) -> Result<
        Box<[crate::presentation::PresentationEvent]>,
        crate::presentation::WebCanvasInputError,
    > {
        self.presenter.take_events()
    }

    /// Counts newly rendered frames only after their canvas upload succeeds.
    /// Cached animation-frame uploads do not establish repaint evidence.
    pub(super) fn present_rendered_frame(
        &mut self,
        frame: &PresentationFrame,
    ) -> std::io::Result<()> {
        let generation = self
            .frame_generation
            .checked_add(1)
            .ok_or_else(|| std::io::Error::other("browser rendered-frame generation exhausted"))?;
        self.presenter.present(frame)?;
        self.element
            .set_attribute("data-ritk-frame-generation", &generation.to_string())?;
        self.frame_generation = generation;
        Ok(())
    }

    pub(super) fn publish_semantics(
        &mut self,
        semantics: BrowserCanvasSemantics,
        physical_aspect: Option<PhysicalCanvasAspect>,
    ) -> std::io::Result<()> {
        self.publish_physical_aspect(physical_aspect)?;
        if self.last_semantics == Some(semantics) {
            return Ok(());
        }
        let axis = semantics.axis.to_string();
        let slice_index = semantics.slice_index.to_string();
        let slice_count = semantics.slice_count.to_string();
        let (width, height) = semantics.frame_dimensions_or_zero();
        let width = width.to_string();
        let height = height.to_string();
        let cine_fps = semantics.cine_fps_value();
        let window_center = semantics.window_center_value();
        let window_width = semantics.window_width_value();
        let window_preset_index = semantics.window_preset_index_value();
        self.element
            .set_attribute("data-ritk-load-state", semantics.load_state_value())?;
        self.element
            .set_attribute("data-ritk-frame-state", semantics.frame_state_value())?;
        self.element.set_attribute("data-ritk-axis", &axis)?;
        self.element
            .set_attribute("data-ritk-slice-index", &slice_index)?;
        self.element
            .set_attribute("data-ritk-slice-count", &slice_count)?;
        self.element
            .set_attribute("data-ritk-frame-width", &width)?;
        self.element
            .set_attribute("data-ritk-frame-height", &height)?;
        self.element
            .set_attribute("data-ritk-cine-fps", &cine_fps)?;
        self.element
            .set_attribute("data-ritk-window-center", &window_center)?;
        self.element
            .set_attribute("data-ritk-window-width", &window_width)?;
        self.element
            .set_attribute("data-ritk-window-preset-index", &window_preset_index)?;
        self.last_semantics = Some(semantics);
        Ok(())
    }

    fn publish_physical_aspect(
        &mut self,
        physical_aspect: Option<PhysicalCanvasAspect>,
    ) -> std::io::Result<()> {
        let Some(physical_aspect) = physical_aspect else {
            return Ok(());
        };
        if self.last_physical_aspect == Some(physical_aspect) {
            return Ok(());
        }
        let value = physical_aspect.attribute_value();
        self.element.set_style_property("width", "100%")?;
        self.element.set_style_property("height", "auto")?;
        self.element.set_style_property("aspect-ratio", &value)?;
        self.element
            .set_attribute("data-ritk-display-aspect", &value)?;
        self.last_physical_aspect = Some(physical_aspect);
        Ok(())
    }
}
