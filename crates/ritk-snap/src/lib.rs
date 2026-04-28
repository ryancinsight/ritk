//! `ritk-snap` viewer core.
//!
//! This crate defines the viewer domain model and backend abstraction for
//! DICOM and other medical image studies. It does not perform I/O itself;
//! loading is delegated to `ritk-io` or another data source.
//!
//! The design goal is to keep the viewer frontend/backend split explicit:
//! - core state and navigation live here,
//! - rendering and presentation live behind a backend trait,
//! - DICOM/volume loading remains in `ritk-io`.
//!
//! This crate is intended to support multiple presentation targets, including
//! native desktop and web-backed shells, without duplicating viewer logic.
//!
//! Geometry handling is modality-aware at the summary layer:
//! - CT summaries may be derived from DICOM spatial metadata or loaded image geometry.
//! - MRI summaries preserve the same affine contract but do not assume CT-specific table/bed semantics.
//! - Ultrasound summaries must respect acquisition-specific orientation metadata and may not use CT-only display heuristics.

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
pub struct Study<B: burn::tensor::backend::Backend, const D: usize> {
    /// The voxel image.
    pub image: Image<B, D>,
    /// Optional DICOM metadata associated with the image.
    pub dicom: Option<DicomReadMetadata>,
    /// Source path if the study originated from disk.
    pub source: Option<PathBuf>,
}

impl<B: burn::tensor::backend::Backend, const D: usize> Study<B, D> {
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
    fn load_study<B: burn::tensor::backend::Backend, const D: usize>(
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
pub struct ViewerCore<B: burn::tensor::backend::Backend, const D: usize> {
    study: Option<Study<B, D>>,
    state: ViewerState,
}

impl<B: burn::tensor::backend::Backend, const D: usize> ViewerCore<B, D> {
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

impl<B: burn::tensor::backend::Backend, const D: usize> Default for ViewerCore<B, D> {
    fn default() -> Self {
        Self::new()
    }
}

/// Geometry summary for display and validation.
///
/// The summary is intentionally modality-aware:
/// - CT may use DICOM metadata or image geometry for display.
/// - MRI preserves the same affine contract without CT-specific assumptions.
/// - Ultrasound requires acquisition-specific orientation handling and should
///   not be normalized through CT bed/table heuristics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeometrySummary {
    /// Image dimensions.
    pub dimensions: [usize; 3],
    /// Voxel spacing derived from the loaded image geometry.
    pub spacing: [f64; 3],
    /// Image origin derived from the loaded image geometry.
    pub origin: [f64; 3],
    /// Direction matrix flattened in row-major display order derived from the loaded image geometry.
    pub direction: [f64; 9],
}

impl GeometrySummary {
    /// Build a geometry summary from DICOM metadata.
    pub fn from_dicom(metadata: &DicomReadMetadata) -> Self {
        Self {
            dimensions: metadata.dimensions,
            spacing: metadata.spacing,
            origin: metadata.origin,
            direction: metadata.direction,
        }
    }
}

/// Simple human-readable viewer status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ViewerStatus {
    /// Status message.
    pub message: String,
}

impl ViewerStatus {
    /// Create a new status message.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// Viewer operation result type.
pub type ViewerResult<T> = Result<T>;

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor, TensorData};
    use ritk_core::image::Image;
    use ritk_core::spatial::{Direction, Point, Spacing};

    type Backend = burn_ndarray::NdArray<f32>;

    #[test]
    fn test_viewer_state_default() {
        let state = ViewerState::default();
        assert_eq!(state.slice_index, 0);
        assert_eq!(state.window_center, None);
        assert_eq!(state.window_width, None);
    }

    #[test]
    fn test_viewer_core_load_study_emits_event() {
        let device: <Backend as burn::tensor::backend::Backend>::Device = Default::default();
        let tensor = Tensor::<Backend, 3>::from_data(
            TensorData::new(vec![1.0f32; 8], Shape::new([2, 2, 2])),
            &device,
        );
        let image = Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        );
        let study = Study::new(image);
        let mut core = ViewerCore::<Backend, 3>::new();
        let event = core.load_study(study);

        match event {
            ViewerEvent::StudyLoaded { depth, rows, cols } => {
                assert_eq!(depth, 2);
                assert_eq!(rows, 2);
                assert_eq!(cols, 2);
            }
            other => panic!("unexpected event: {:?}", other),
        }
    }

    #[test]
    fn test_geometry_summary_from_dicom() {
        let meta = DicomReadMetadata {
            series_instance_uid: None,
            study_instance_uid: None,
            frame_of_reference_uid: None,
            series_description: None,
            modality: Some("CT".to_string()),
            patient_id: None,
            patient_name: None,
            study_date: None,
            series_date: None,
            series_time: None,
            dimensions: [3, 4, 5],
            spacing: [0.8, 0.8, 1.5],
            origin: [10.0, 20.0, -30.0],
            direction: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            bits_allocated: Some(16),
            bits_stored: Some(16),
            high_bit: Some(15),
            photometric_interpretation: Some("MONOCHROME2".to_string()),
            slices: Vec::new(),
            private_tags: std::collections::HashMap::new(),
            preservation: Default::default(),
        };

        let summary = GeometrySummary::from_dicom(&meta);
        assert_eq!(summary.dimensions, [3, 4, 5]);
        assert_eq!(summary.spacing, [0.8, 0.8, 1.5]);
        assert_eq!(summary.origin, [10.0, 20.0, -30.0]);
    }

    #[test]
    fn test_viewer_status_new() {
        let status = ViewerStatus::new("loaded");
        assert_eq!(status.message, "loaded");
    }
}
