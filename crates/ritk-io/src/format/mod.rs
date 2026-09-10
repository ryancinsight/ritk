pub mod analyze;
pub mod dicom;
/// Native DICOMweb transport backed by the blocking HTTP client.
///
/// Browser hosts use their platform fetch implementation and pass the
/// completed, bounded DICOM bytes through the byte-reader API instead.
#[cfg(not(target_arch = "wasm32"))]
pub mod dicomweb;
pub mod jpeg;
pub mod metaimage;
pub mod mgh;
pub mod minc;
pub mod nifti;
pub mod nrrd;
pub mod png;
pub mod tiff;
pub mod vtk;

#[cfg(test)]
#[path = "tests_native_readers.rs"]
mod tests_native_readers;
