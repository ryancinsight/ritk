pub mod analyze;
pub mod dicom;
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

#[cfg(all(test, feature = "coeus"))]
#[path = "tests_native_readers.rs"]
mod tests_native_readers;
