# ritk-nifti

`ritk-nifti` reads and writes NIfTI-1 and NIfTI-2 single-file images for RITK.
It owns NIfTI byte parsing and encoding; `ritk-io` exposes the same codec
through its format dispatch surface. Analyze 7.5 `.hdr`/`.img` pairs belong to
`ritk-analyze`.

## Choose a data surface

`NiftiDocument` validates and retains the complete uncompressed `.nii` stream,
including header fields, extensions, and exact sample bytes. It reads gzip
wrapped `.nii.gz` input and writes either framing without changing the
uncompressed bytes. Use [`NiftiDocument::from_stored_series`] to construct a
document from RITK stored samples. That path preserves all ten supported
scalar sample types and their bit patterns. The NIfTI header carries one
Cartesian spatial transform and one global linear calibration; the converter
returns a typed rejection when source semantics do not fit that model.

The image convenience functions (`read_nifti`, `read_nifti_series`, and
`read_nifti_labels`) project voxel values into `f32` images or `u32` labels.
Use `NiftiDocument` when the stored datatype and exact integer or floating
point bits must remain unchanged.

## Read and transcode a document

```rust,no_run
use ritk_nifti::NiftiDocument;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let document = NiftiDocument::read("scan.nii.gz")?;
document.write("scan-copy.nii")?;
# Ok(())
# }
```

The example for constructing a typed document from a `StoredSeries` is on
[`NiftiDocument::from_stored_series`].

## Spatial and acquisition conventions

RITK images use `[depth, row, column]` sample order and LPS physical
coordinates. NIfTI stores `[x, y, z]` with RAS affines. The codec performs the
axis permutation and coordinate conversion at the format boundary.

Multiple volumes become one rank-4 NIfTI series in input order. All volumes
must share shape, sample type, geometry, and representable calibration. NIfTI-1
stores dimensions as signed 16-bit integers and spatial and scaling fields as
32-bit floats. Each positive axis and volume count therefore fits at most
32,767, and floating-point fields reject values that would need rounding.
NIfTI-2 stores dimensions as signed 64-bit integers and spatial and scaling
fields as 64-bit floats. The sample payload remains in its declared scalar
datatype in either version.

## Documentation

- [NIfTI format guide](../../docs/book/nifti_format.md)
- [API reference](https://docs.rs/ritk-nifti)
