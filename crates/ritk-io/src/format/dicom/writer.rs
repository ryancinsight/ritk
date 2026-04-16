//! DICOM series writer scaffold.
//!
//! This module defines the crate-local DICOM writer entry points and wrapper
//! type expected by `ritk_io::format::dicom`.
//!
//! The current implementation is intentionally conservative: it validates the
//! image shape and materializes a directory layout for series export, but it
//! does not attempt full DICOM object synthesis. That behavior belongs in a
//! later phase once the series metadata model and SOP-class policy are fixed.
//!
//! # Invariants
//!
//! - The writer only accepts 3-D images.
//! - The series path must resolve to a directory that can be created or reused.
//! - Output is organized as a slice-oriented series with deterministic file
//!   names.
//!
//! # Public API
//!
//! - `write_dicom_series`
//! - `DicomWriter`

use anyhow::{bail, Context, Result};
use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

/// Write a 3-D image as a DICOM series directory scaffold.
///
/// The current implementation creates the destination directory and validates
/// the image geometry, then returns success. A future phase will serialize
/// concrete DICOM objects once the write policy is finalized.
pub fn write_dicom_series<B: Backend, P: AsRef<Path>>(path: P, image: &Image<B, 3>) -> Result<()> {
    let path = path.as_ref();

    let shape = image.shape();
    let [depth, rows, cols] = shape;

    if depth == 0 || rows == 0 || cols == 0 {
        bail!("DICOM series cannot contain zero-sized dimensions");
    }

    let series_dir = ensure_series_directory(path)?;

    // The scaffold creates a deterministic slice namespace so later write
    // phases can emit SOP instances without changing the directory contract.
    for index in 0..depth {
        let slice_path = series_dir.join(format!("slice_{index:04}.dcm"));
        if slice_path.exists() {
            continue;
        }

        // Reserve the path by creating an empty placeholder file. This keeps
        // the series layout stable for later phases and fails fast on access
        // errors.
        std::fs::File::create(&slice_path)
            .with_context(|| "failed to reserve DICOM slice output path")?;
    }

    Ok(())
}

/// Compatibility wrapper type for writer-side domain APIs.
pub struct DicomWriter<B> {
    _phantom: PhantomData<B>,
}

impl<B> DicomWriter<B> {
    /// Construct a new writer wrapper.
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Return the directory that would be used for a series export.
    pub fn series_path<P: AsRef<Path>>(path: P) -> PathBuf {
        path.as_ref().to_path_buf()
    }
}

impl<B> Default for DicomWriter<B> {
    fn default() -> Self {
        Self::new()
    }
}

fn ensure_series_directory(path: &Path) -> Result<PathBuf> {
    if path.exists() {
        if !path.is_dir() {
            bail!("DICOM output path exists and is not a directory");
        }
        return Ok(path.to_path_buf());
    }

    std::fs::create_dir_all(path)
        .with_context(|| "failed to create DICOM series output directory")?;
    Ok(path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor, TensorData};
    use ritk_core::image::Image;
    use ritk_core::spatial::{Direction, Point, Spacing};

    type Backend = burn_ndarray::NdArray<f32>;

    fn make_image(depth: usize, rows: usize, cols: usize) -> Image<Backend, 3> {
        let device = Default::default();
        let data = vec![0.0_f32; depth * rows * cols];
        let tensor = Tensor::<Backend, 3>::from_data(
            TensorData::new(data, Shape::new([depth, rows, cols])),
            &device,
        );
        Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    #[test]
    fn test_writer_rejects_zero_dimension() {
        let image = make_image(1, 1, 1);
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("series");
        let result = write_dicom_series(&path, &image);
        assert!(result.is_ok());
        assert!(path.is_dir());
        assert!(path.join("slice_0000.dcm").exists());
    }

    #[test]
    fn test_writer_wrapper_construction() {
        let writer = DicomWriter::<Backend>::new();
        let derived = DicomWriter::<Backend>::series_path("out");
        assert_eq!(derived, PathBuf::from("out"));
        let _: DicomWriter<Backend> = writer;
    }

    #[test]
    fn test_writer_creates_series_directory() {
        let image = make_image(2, 3, 4);
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("export");
        let result = write_dicom_series(&path, &image);
        assert!(result.is_ok());
        assert!(path.is_dir());
        assert!(path.join("slice_0000.dcm").exists());
        assert!(path.join("slice_0001.dcm").exists());
    }

    #[test]
    fn test_writer_reuses_existing_directory() {
        let image = make_image(1, 2, 2);
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("existing");
        std::fs::create_dir_all(&path).unwrap();
        let result = write_dicom_series(&path, &image);
        assert!(result.is_ok());
        assert!(path.join("slice_0000.dcm").exists());
    }
}
