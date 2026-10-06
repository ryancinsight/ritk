//! DICOM series cataloging, selection, loading, and the `DicomReader` facade.

mod catalog;
mod loader;
mod scan;
mod types;

pub use catalog::DicomStudyCatalog;
pub use loader::{
    load_dicom_series, load_native_dicom_series, read_dicom_series, read_native_dicom_series,
    read_native_dicom_series_with_uid, DicomReader,
};
pub use scan::scan_dicom_directory;
pub use types::DicomSeriesInfo;
