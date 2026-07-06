use anyhow::Result;
use ritk_core::image::Image;
use ritk_io::DicomReadMetadata;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Shared CPU backend used by the default viewer core tests and headless tools.
///
/// The viewer core itself is backend-agnostic at the API boundary; this alias
/// is provided for convenience in headless validation and examples.
pub type DefaultBackend = burn_ndarray::NdArray<f32>;

/// A loaded study ready for viewing.
///
/// This wraps the image together with optional DICOM metadata so a backend can
/// inspect geometry, slice spacing, and modality-specific context.
#[derive(Debug, Clone)]
pub struct Study<B: ritk_image::tensor::Backend, const D: usize> {
    /// The voxel image.
    pub image: Image<B, D>,
    /// Optional DICOM metadata associated with the image.
    pub dicom: Option<DicomReadMetadata>,
    /// Source path if the study originated from disk.
    pub source: Option<PathBuf>,
}

impl<B: ritk_image::tensor::Backend, const D: usize> Study<B, D> {
    /// Create a new study from an image.
    pub fn new(image: Image<B, D>) -> Self {
        Self {
            image,
            dicom: None,
            source: None,
        }
    }

    /// Attach DICOM metadata.
    pub fn with_dicom(mut self, dicom: DicomReadMetadata) -> Self {
        self.dicom = Some(dicom);
        self
    }

    /// Attach the source path.
    pub fn with_source(mut self, source: PathBuf) -> Self {
        self.source = Some(source);
        self
    }
}

/// Navigation state for a volume viewer.
///
/// `slice_index` is interpreted along the first axis of the `Image<B, 3>`
/// tensor shape `[depth, rows, cols]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ViewerState {
    /// Current slice index along the depth axis.
    pub slice_index: usize,
    /// Window center for intensity display.
    pub window_center: Option<f32>,
    /// Window width for intensity display.
    pub window_width: Option<f32>,
}

impl ViewerState {
    /// Create a default state at the first slice.
    pub fn new() -> Self {
        Self {
            slice_index: 0,
            window_center: None,
            window_width: None,
        }
    }
}

impl Default for ViewerState {
    fn default() -> Self {
        Self::new()
    }
}

/// Default window centre used when the viewer state has no explicit W/L set.
///
/// This is the conservative 8-bit equivalent that works for US and unknown modalities.
pub(crate) const DEFAULT_WINDOW_CENTER: f32 = 128.0;

/// Default window width used when the viewer state has no explicit W/L set.
pub(crate) const DEFAULT_WINDOW_WIDTH: f32 = 256.0;

/// Events emitted by the viewer core for presentation backends.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ViewerEvent {
    /// The current slice index changed.
    SliceChanged { index: usize },
    /// Window/level changed.
    WindowChanged {
        center: Option<f32>,
        width: Option<f32>,
    },
    /// A study was loaded.
    StudyLoaded {
        /// Number of slices.
        depth: usize,
        /// Number of rows.
        rows: usize,
        /// Number of columns.
        cols: usize,
    },
    /// A textual status message.
    Status { message: String },
}

/// Presentation backend contract for `ritk-snap`.
///
/// A backend may be a native desktop renderer, a web UI shell, or a headless
/// inspection implementation used for tests and automation.
pub trait ViewerBackend {
    /// Backend-specific error type.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Initialize the backend.
    fn initialize(&mut self) -> Result<(), Self::Error>;
    /// Load a study into the backend.
    fn load_study<B: ritk_image::tensor::Backend, const D: usize>(
        &mut self,
        study: &Study<B, D>,
        state: &ViewerState,
    ) -> Result<(), Self::Error>;
    /// Update the current state.
    fn update_state(&mut self, state: &ViewerState) -> Result<(), Self::Error>;
    /// Render a frame or emit the latest presentation state.
    fn render(&mut self) -> Result<(), Self::Error>;
    /// Handle an event from the viewer core.
    fn handle_event(&mut self, event: ViewerEvent) -> Result<(), Self::Error>;
}

/// Headless viewer core that owns the current study and presentation state.
#[derive(Debug)]
pub struct ViewerCore<B: ritk_image::tensor::Backend, const D: usize> {
    pub(crate) study: Option<Study<B, D>>,
    pub(crate) state: ViewerState,
}

impl<B: ritk_image::tensor::Backend, const D: usize> ViewerCore<B, D> {
    /// Create an empty viewer core.
    pub fn new() -> Self {
        Self {
            study: None,
            state: ViewerState::default(),
        }
    }

    /// Load a study into the core.
    pub fn load_study(&mut self, study: Study<B, D>) -> ViewerEvent {
        let shape = study.image.shape();
        self.state.slice_index = 0;
        self.study = Some(study);
        ViewerEvent::StudyLoaded {
            depth: shape[0],
            rows: shape[1],
            cols: shape[2],
        }
    }

    /// Set the current slice index.
    pub fn set_slice_index(&mut self, index: usize) -> ViewerEvent {
        self.state.slice_index = index;
        ViewerEvent::SliceChanged { index }
    }

    /// Set the window center and width.
    pub fn set_window_level(&mut self, center: Option<f32>, width: Option<f32>) -> ViewerEvent {
        self.state.window_center = center;
        self.state.window_width = width;
        ViewerEvent::WindowChanged { center, width }
    }

    /// Access the current state.
    pub fn state(&self) -> &ViewerState {
        &self.state
    }

    /// Access the current study.
    pub fn study(&self) -> Option<&Study<B, D>> {
        self.study.as_ref()
    }
}

impl<B: ritk_image::tensor::Backend, const D: usize> Default for ViewerCore<B, D> {
    fn default() -> Self {
        Self::new()
    }
}
