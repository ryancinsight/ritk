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
use ritk_core::filter::{BedSeparationConfig, BedSeparationFilter, GaussianFilter, MedianFilter};
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

impl<B: burn::tensor::backend::Backend> ViewerCore<B, 3> {
    /// Apply a filter to the currently loaded study's image.
    ///
    /// # Precondition
    /// If no study is loaded, returns `Ok(ViewerEvent::Status { message })` with a
    /// message indicating no study is present — no error is raised.
    ///
    /// # Postcondition
    /// On success, `self.study` contains the filtered image with its spatial metadata
    /// (origin, spacing, direction) preserved. The filter name and new shape are
    /// encoded in the returned `ViewerEvent::Status` message.
    ///
    /// On filter failure, the original study is restored and the error is propagated.
    pub fn apply_filter(&mut self, kind: &FilterKind) -> anyhow::Result<ViewerEvent> {
        let study = match self.study.take() {
            None => {
                return Ok(ViewerEvent::Status {
                    message: "no study loaded".to_string(),
                });
            }
            Some(s) => s,
        };

        // Apply the selected filter. The borrow of study.image is released before
        // `filter_result` is consumed below, enabling safe move of study fields.
        let filter_result: anyhow::Result<Image<B, 3>> = match kind {
            FilterKind::BedSeparation(config) => {
                BedSeparationFilter::new(*config).apply(&study.image)
            }
            FilterKind::Gaussian { sigma } => {
                // GaussianFilter takes physical-unit sigmas per dimension.
                // Broadcasting a single sigma across all three axes.
                Ok(GaussianFilter::<B>::new(vec![f64::from(*sigma); 3]).apply(&study.image))
            }
            FilterKind::Median { radius } => MedianFilter::new(*radius).apply(&study.image),
        };

        let filter_name = match kind {
            FilterKind::BedSeparation(_) => "BedSeparation",
            FilterKind::Gaussian { .. } => "Gaussian",
            FilterKind::Median { .. } => "Median",
        };

        match filter_result {
            Err(e) => {
                // Restore the original study so the core remains usable after a
                // filter error.
                self.study = Some(study);
                Err(e)
            }
            Ok(new_image) => {
                let shape = new_image.shape();
                self.study = Some(Study {
                    image: new_image,
                    dicom: study.dicom,
                    source: study.source,
                });
                Ok(ViewerEvent::Status {
                    message: format!("{filter_name} applied; shape {:?}", shape),
                })
            }
        }
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

// ── Remote serde helper ──────────────────────────────────────────────────────
//
// `BedSeparationConfig` is defined in `ritk-core` and does not derive
// `Serialize`/`Deserialize`. The `#[serde(remote = "...")]` pattern generates
// `BedSeparationConfigSerde::serialize` and `BedSeparationConfigSerde::deserialize`
// static methods that serde dispatches to via `#[serde(with = "BedSeparationConfigSerde")]`
// on the `FilterKind::BedSeparation` field.
#[derive(Serialize, Deserialize)]
#[serde(remote = "BedSeparationConfig")]
struct BedSeparationConfigSerde {
    pub body_threshold: f32,
    pub background_threshold: f32,
    pub keep_largest_component: bool,
    pub closing_radius: usize,
    pub opening_radius: usize,
    pub outside_value: f32,
}

/// Selectable image filters exposed through the viewer core.
///
/// Each variant maps 1-to-1 onto a concrete filter implementation in
/// `ritk_core::filter`. Dispatch in `ViewerCore::apply_filter` is exhaustive
/// and concrete — no trait objects are used.
///
/// # Variant invariants
/// - `BedSeparation`: body_threshold and background_threshold must be valid
///   Hounsfield-range values; outside_value must be representable as f32.
/// - `Gaussian`: sigma > 0.0 (zero sigma is a no-op but not an error; the
///   underlying filter skips dimensions where sigma ≤ 1e-6).
/// - `Median`: radius = 0 is identity (each voxel is its own sole neighbour).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterKind {
    /// CT bed/table separation mask filter.
    ///
    /// Applies intensity thresholding, optional largest-component selection,
    /// and morphological closing/opening to isolate the patient body.
    /// Voxels outside the retained mask are replaced by `config.outside_value`.
    BedSeparation(#[serde(with = "BedSeparationConfigSerde")] BedSeparationConfig),

    /// Isotropic Gaussian smoothing filter.
    ///
    /// `sigma` is the standard deviation in physical units (mm), broadcast
    /// identically across all three spatial dimensions.
    Gaussian {
        /// Standard deviation in physical units (mm).
        sigma: f32,
    },

    /// Sliding-window median filter.
    ///
    /// Replaces each voxel with the median of its `(2·radius+1)³` axis-aligned
    /// neighbourhood using replicate (clamp) boundary conditions.
    Median {
        /// Neighbourhood half-width in voxels.
        radius: usize,
    },
}

/// Intensity display defaults derived from DICOM modality.
///
/// Window centre and width follow standard clinical display conventions:
///
/// | Modality | Centre  | Width | Rationale                                        |
/// |----------|---------|-------|--------------------------------------------------|
/// | CT       | -400 HU | 1500  | Standard lung window; HU range [-1150, 350]      |
/// | MR/MRI   | 600     | 1200  | Relative intensity; typical brain soft-tissue    |
/// | US       | 128     | 256   | 8-bit acoustic impedance range [0, 255]          |
/// | Default  | 128     | 256   | Conservative unsigned 8-bit equivalent           |
///
/// # Mathematical basis
/// For a window (c, w), the display range is [c − w/2, c + w/2].
/// CT lung: [-400 − 750, -400 + 750] = [-1150, 350] HU (standard lung protocol).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModalityDisplay {
    /// Window centre for intensity display.
    pub window_center: f64,
    /// Window width for intensity display.
    pub window_width: f64,
    /// Modality string used to select the defaults.
    pub modality: String,
}

impl ModalityDisplay {
    /// Return display defaults for the given DICOM modality string.
    ///
    /// Matching is exact and case-sensitive to preserve DICOM tag semantics.
    /// Unknown or absent modalities fall back to the 8-bit unsigned default.
    pub fn for_modality(modality: Option<&str>) -> Self {
        match modality {
            Some("CT") => Self {
                window_center: -400.0,
                window_width: 1500.0,
                modality: "CT".to_string(),
            },
            Some("MR") => Self {
                window_center: 600.0,
                window_width: 1200.0,
                modality: "MR".to_string(),
            },
            Some("MRI") => Self {
                window_center: 600.0,
                window_width: 1200.0,
                modality: "MRI".to_string(),
            },
            Some("US") => Self {
                window_center: 128.0,
                window_width: 256.0,
                modality: "US".to_string(),
            },
            other => Self {
                window_center: 128.0,
                window_width: 256.0,
                modality: other.unwrap_or("").to_string(),
            },
        }
    }
}

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

    // ── New tests ─────────────────────────────────────────────────────────────

    /// Analytically constructed 2×4×4 volume:
    /// - First 16 voxels (slice 0, all rows/cols): -1000.0 HU  (air)
    /// - Next  16 voxels (slice 1, all rows/cols):     0.0 HU  (soft tissue)
    ///
    /// BedSeparationConfig:
    ///   body_threshold      = -500.0  → voxels < -500 are background
    ///   background_threshold = -700.0 (unused by current impl; kept for API completeness)
    ///   keep_largest_component = false (avoids connected-component overhead)
    ///   closing_radius = 0, opening_radius = 0 (identity morphology)
    ///   outside_value = -2048.0
    ///
    /// Expected outcome:
    ///   All 16 air voxels (-1000 < -500) → replaced by -2048.0
    ///   All 16 tissue voxels (0 ≥ -500)  → unchanged at 0.0
    ///   Shape is preserved: [2, 4, 4]
    #[test]
    fn test_filter_kind_bed_separation_dispatch_replaces_study_image() {
        let device: <Backend as burn::tensor::backend::Backend>::Device = Default::default();

        // Analytically: first 16 = -1000.0 (air, below threshold), next 16 = 0.0 (tissue).
        let mut vals = vec![-1000.0_f32; 16];
        vals.extend_from_slice(&[0.0_f32; 16]);
        assert_eq!(vals.len(), 32, "2×4×4 = 32 voxels");

        let tensor =
            Tensor::<Backend, 3>::from_data(TensorData::new(vals, Shape::new([2, 4, 4])), &device);
        let image = Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        );
        let mut core = ViewerCore::<Backend, 3>::new();
        core.load_study(Study::new(image));

        let config = BedSeparationConfig {
            body_threshold: -500.0,
            background_threshold: -700.0,
            keep_largest_component: false,
            closing_radius: 0,
            opening_radius: 0,
            outside_value: -2048.0,
        };
        let result = core.apply_filter(&FilterKind::BedSeparation(config));

        // Event must be Status.
        let event = result.expect("apply_filter must succeed");
        assert!(
            matches!(event, ViewerEvent::Status { .. }),
            "expected ViewerEvent::Status, got {:?}",
            event
        );

        // Study must still be present and shape must be preserved.
        let study = core.study().expect("study must be present after filter");
        assert_eq!(
            study.image.shape(),
            [2, 4, 4],
            "shape must be preserved by BedSeparationFilter"
        );

        // Extract voxel values to validate filter computation.
        let out_vals: Vec<f32> = study
            .image
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .expect("must extract f32 slice")
            .to_vec();

        // At least one voxel must equal outside_value (-2048.0): the air voxels.
        let has_outside = out_vals.iter().any(|&v| (v - (-2048.0_f32)).abs() < 1e-6);
        assert!(
            has_outside,
            "at least one voxel must equal outside_value=-2048.0 after BedSeparation; got {:?}",
            &out_vals[..4]
        );

        // All tissue voxels (originally 0.0) must remain at 0.0.
        for (i, &v) in out_vals[16..].iter().enumerate() {
            assert!(
                (v - 0.0_f32).abs() < 1e-6,
                "tissue voxel at index {} expected 0.0, got {}",
                16 + i,
                v
            );
        }
    }

    /// An empty `ViewerCore` (no study loaded) must return a `Status` event
    /// whose message contains "no study", not an error.
    ///
    /// Precondition: `self.study` is `None`.
    /// Postcondition: `Ok(ViewerEvent::Status { message })` where `message`
    ///   contains the substring "no study".
    #[test]
    fn test_filter_kind_no_study_returns_status_message() {
        let mut core = ViewerCore::<Backend, 3>::new();
        let result = core.apply_filter(&FilterKind::Gaussian { sigma: 1.0 });

        let event = result.expect("apply_filter on empty core must return Ok");
        match event {
            ViewerEvent::Status { message } => {
                assert!(
                    message.contains("no study"),
                    "status message must contain 'no study'; got: {:?}",
                    message
                );
            }
            other => panic!(
                "expected ViewerEvent::Status for empty core, got {:?}",
                other
            ),
        }
    }

    /// Verify `ModalityDisplay::for_modality` against the analytically derived
    /// standard clinical window parameters documented in the struct-level rustdoc.
    ///
    /// CT lung window:  centre = -400, width = 1500  → range [-1150, 350] HU
    /// MR brain window: centre =  600, width = 1200  → typical soft-tissue
    /// US 8-bit range:  centre =  128, width =  256  → [0, 255]
    /// None / unknown:  centre =  128, width =  256  → conservative default
    #[test]
    fn test_modality_display_ct_window_parameters() {
        let ct = ModalityDisplay::for_modality(Some("CT"));
        assert_eq!(
            ct.window_center, -400.0,
            "CT window_center must be -400.0 (standard lung window)"
        );
        assert_eq!(
            ct.window_width, 1500.0,
            "CT window_width must be 1500.0 (standard lung window)"
        );
        assert_eq!(ct.modality, "CT");

        let mr = ModalityDisplay::for_modality(Some("MR"));
        assert_eq!(
            mr.window_center, 600.0,
            "MR window_center must be 600.0 (typical brain window)"
        );
        assert_eq!(
            mr.window_width, 1200.0,
            "MR window_width must be 1200.0 (typical brain window)"
        );
        assert_eq!(mr.modality, "MR");

        let us = ModalityDisplay::for_modality(Some("US"));
        assert_eq!(
            us.window_center, 128.0,
            "US window_center must be 128.0 (8-bit acoustic range midpoint)"
        );
        assert_eq!(
            us.window_width, 256.0,
            "US window_width must be 256.0 (full 8-bit range)"
        );
        assert_eq!(us.modality, "US");

        let unknown = ModalityDisplay::for_modality(None);
        assert_eq!(
            unknown.window_center, 128.0,
            "None modality must fall back to default centre 128.0"
        );
        assert_eq!(
            unknown.window_width, 256.0,
            "None modality must fall back to default width 256.0"
        );
        assert_eq!(unknown.modality, "");
    }
}
