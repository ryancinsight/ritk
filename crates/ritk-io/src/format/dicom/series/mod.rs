//! DICOM series scanning, loading, and the `DicomReader` facade.

mod loader;
mod scan;
mod types;

pub use loader::{load_dicom_series, read_dicom_series, DicomReader};
pub use scan::scan_dicom_directory;
pub use types::DicomSeriesInfo;
