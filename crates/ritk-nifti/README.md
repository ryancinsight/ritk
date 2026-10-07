# ritk-nifti

NIfTI-1 and NIfTI-2 single-file image I/O for RITK. The crate reads `.nii` and
`.nii.gz` documents and writes explicit NIfTI-1 or NIfTI-2 image volumes,
series, and label maps.

## Stored documents

`NiftiDocument` validates and preserves the original bytes for all ten fixed-
width scalar datatypes: `u8`, `i8`, `u16`, `i16`, `u32`, `i32`, `u64`, `i64`,
`f32`, and `f64`. It keeps integer values and IEEE 754 bit patterns unchanged
when transcoding between `.nii` and `.nii.gz`. The image convenience readers
convert only their supported input types to `f32`; label readers return `u32`.

```rust,no_run
use ritk_nifti::{NiftiDocument, NiftiDocumentError};

fn inspect(path: &str) -> Result<(), NiftiDocumentError> {
    let document = NiftiDocument::read(path)?;
    let sample_bytes = document.sample_bytes();
    let sample_width = usize::from(document.header().bits_per_sample / 8);
    assert_eq!(sample_bytes.len() % sample_width, 0);
    Ok(())
}
```

See the [NIfTI format manual](https://github.com/ryancinsight/ritk/blob/main/docs/book/nifti_format.md)
and [API reference](https://docs.rs/ritk-nifti/latest/ritk_nifti/).

## Converting stored samples

`NiftiDocument::from_stored_series` builds a NIfTI-1 or NIfTI-2 document from
RITK's shared stored-value model. It preserves each supported scalar sample's
bits while encoding the payload in NIfTI's little-endian order. The caller
provides the source-format identifier and any scoped source metadata losses
that the shared model cannot carry; preparation rejects reported losses and
target-incompatible series before a destination path is opened.

```rust
use ritk_codecs::SampleBuffer;
use ritk_image::ImageMetadata;
use ritk_image_io::{IntensityCalibration, SeriesAxis, StoredSeries, StoredVolume};
use ritk_nifti::{NiftiDocument, NiftiVersion};
use ritk_spatial::CoordinateMap;

fn make_document() -> Result<(), Box<dyn std::error::Error>> {
    let volume = StoredVolume::new(
        [1, 1, 2],
        SampleBuffer::from_samples(vec![12_i16, 34]),
        ImageMetadata::default(),
        CoordinateMap::Cartesian,
        IntensityCalibration::Identity,
    )?;
    let series = StoredSeries::new(vec![volume], SeriesAxis::SingleVolume)?;
    let document =
        NiftiDocument::from_stored_series("nrrd", &series, NiftiVersion::Two, [])?;
    assert_eq!(document.header().dimensions, [3, 2, 1, 1, 1, 1, 1, 1]);
    Ok(())
}
```

NIfTI-1 stores spatial and scaling fields as 32-bit floats; construction
rejects an affine that becomes singular after this narrowing. NIfTI-2 stores
them as 64-bit floats. Both versions store voxel samples without converting
their scalar type. A singleton ordered acquisition axis remains rank 4.

## API roles

- [`NiftiDocument`] and [`transcode_nifti_document`] validate and transport a
  complete single-file document without rewriting its uncompressed bytes.
- [`read_nifti`] and [`read_nifti_from_bytes`] read one supported image volume.
- [`read_nifti_series`] and [`read_nifti_series_from_bytes`] read ordered
  acquisition volumes.
- [`read_nifti_labels`] reads a label map as `u32` values.
- [`write_nifti`] and [`write_nifti2`] write one image volume.
- [`write_nifti_series`] and [`write_nifti2_series`] write acquisition volumes.
- [`NiftiDocument::from_stored_series`] constructs a document from shared
  typed samples, geometry, calibration, and acquisition-axis metadata.
- [`write_nifti_labels`] and [`write_nifti2_labels`] write label maps.

Analyze 7.5 `.hdr`/`.img` pairs belong to `ritk-analyze`; this crate handles
single-file NIfTI magic only.

## Spatial conventions

RITK tensors use `[Z, Y, X]` with LPS physical coordinates; NIfTI stores
`[X, Y, Z]` with RAS affines. Image APIs perform the axis and coordinate
conversion. A rank-3 file is one volume; rank-4 input is an ordered acquisition
series and single-volume readers reject multiple volumes.

An internal affine maps `[depth, row, column]` to LPS. NIfTI affine columns map
`[x, y, z]` to RAS, so the writer reorders the internal columns and changes
the first two physical axes from LPS to RAS at the format boundary.
