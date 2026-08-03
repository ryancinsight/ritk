# Diffusion Gradient Schemes

`ritk-diffusion-scheme` is RITK's single-source-of-truth for validated
diffusion-MRI acquisition metadata. A diffusion series is meaningful only
when each volume carries a physically typed weighting, a unit gradient
direction, and a declared coordinate frame. This crate owns that
format-neutral contract; format crates convert DICOM, NRRD, MRtrix, or
companion-file metadata at their trust boundaries.

## Why a separate crate?

Diffusion model fitting consumes three inputs — a 4-D image, a gradient
scheme, and (for constrained methods) a response function. If the scheme is
just two `f64` arrays, each consumer re-implements the same validation:
non-finite checks, zero/unit-vector contracts, b-value thresholding, shell
grouping, frame tracking, and reorientation. A single validated type
eliminates this duplication and prevents the single most common defect class
in this domain: a rigid-motion correction that does not rotate the gradient
table.

## Diffusion weighting

`DiffusionWeighting` stores the diffusion sensitization factor with dimension
time per area. Internal storage uses canonical SI seconds per square meter
through the Aequitas quantity system. The scanner-facing constructor and
accessor use the MRI convention seconds per square millimeter:

```rust,ignore
use ritk_diffusion_scheme::DiffusionWeighting;

let b1000 = DiffusionWeighting::from_seconds_per_square_millimeter(1_000.0)?;
assert_eq!(b1000.seconds_per_square_millimeter(), 1_000.0);
```

Construction rejects negative, NaN, or infinite values, and catches overflow
when converting to canonical SI. The `is_unweighted` method checks whether
the value is exactly zero in the scanned representation.

## Gradient directions and frames

A `GradientDirection` pairs a weighting with a validated direction vector.
The invariant depends on the weighting:

- An unweighted (b = 0) entry requires an exact zero vector.
- A weighted entry requires a finite unit vector within 1e-6 Euclidean norm.

Both are enforced at construction:

```rust,ignore
use ritk_diffusion_scheme::{GradientDirection, DiffusionWeighting};
use ritk_spatial::Vector;

let b0_weighting = DiffusionWeighting::from_seconds_per_square_millimeter(0.0)?;
let b1000_weighting = DiffusionWeighting::from_seconds_per_square_millimeter(1_000.0)?;

// b0 entry: zero vector.
let b0 = GradientDirection::new(b0_weighting, Vector::new([0.0, 0.0, 0.0]))?;

// Weighted entry: unit vector.
let dir = GradientDirection::new(b1000_weighting, Vector::new([1.0, 0.0, 0.0]))?;

// This fails — weighted volume with non-unit vector.
assert!(GradientDirection::new(
    b1000_weighting,
    Vector::new([0.5, 0.5, 0.5]),
).is_err());
```

`GradientFrame` is the coordinate frame of every direction in a scheme:

| Variant | Meaning | Typical source |
|---|---|---|
| `ImageAxis` | Image index-axis coordinates | FSL `.bvec`, MRtrix `DW_scheme` |
| `Lps` | Physical Left-Posterior-Superior patient coordinates | DICOM `(0018,9089)`, NRRD after measurement-frame conversion |

A direction is numerically valid only within its declared frame. Applying
an ImageAxis direction in LPS or vice versa produces a silently wrong
tensor field. The frame is pinned at construction and persistent: every
format codec states which frame it produces, and the reader converts once.

## The GradientScheme type

`GradientScheme` is an ordered collection of validated gradient directions
with one declared frame. Construction is infallible once all entries are
validated:

```rust,ignore
use ritk_diffusion_scheme::{GradientScheme, GradientFrame};
use ritk_spatial::Vector;

let scheme = GradientScheme::from_seconds_per_square_millimeter(
    vec![
        (0.0, Vector::new([0.0, 0.0, 0.0])),
        (1_000.0, Vector::new([1.0, 0.0, 0.0])),
        (1_000.0, Vector::new([0.0, 1.0, 0.0])),
        (2_000.0, Vector::new([0.0, 0.0, 1.0])),
    ],
    GradientFrame::ImageAxis,
)?;

assert_eq!(scheme.len(), 4);
```

### B0 thresholding

`from_seconds_per_square_millimeter` applies a default **50 s/mm² threshold**.
Scanner baseline acquisitions at or below this value are canonicalized to
exact zero weighting and direction, because their gradient orientation is
either absent or not physically meaningful. Values above the threshold must
carry a unit direction.

### Query

- `directions()` — all entries in acquisition order.
- `frame()` — the declared `GradientFrame`.
- `b0_indices(threshold)` and `dwi_indices(threshold)` — split the scheme at a
  given weighting threshold.
- `shells()` — unique nonzero shell weightings sorted in ascending order.
  Multi-shell acquisitions (e.g. b = 1 000, 2 000, 3 000) produce the
  corresponding list.

### Gradient reorientation

When a rigid or affine correction is applied to a diffusion series, the
gradient directions must rotate with the image. A motion correction that
does not rotate the gradient table produces a silently wrong tensor field —
this is the single most common defect class in diffusion MRI.

`reorient` applies one proper orthonormal rotation to every weighted
direction. b = 0 volumes pass through unchanged because their placeholder
vector carries no physical meaning:

```rust,ignore
// 90° rotation about the z-axis.
let rotation = [
    [0.0, -1.0, 0.0],
    [1.0,  0.0, 0.0],
    [0.0,  0.0, 1.0],
];
let reoriented = scheme.reorient(rotation)?;
```

`reorient_per_volume` is the motion-correction path: each volume gets its
own rotation matrix from the per-volume registration. It requires exactly
one rotation per volume and rejects any matrix that is non-finite, not
orthonormal within 1e-9, or has a determinant other than +1.

## Format codecs

Every codec states which frame it produces and converts once. Any transform
applied to the series reorients the directions with it.

### FSL `.bval` / `.bvec`

FSL stores gradient metadata as companion text files. The `.bval` file is a
single line of space-separated s/mm² values. The `.bvec` file is three lines
— one per spatial component (x, y, z) — of space-separated unit direction
components.

Directions are in **image-axis coordinates** (`GradientFrame::ImageAxis`).

```rust,ignore
use ritk_diffusion_scheme::fsl::{read_fsl_scheme, write_fsl_scheme};

let bval = "0 1000 1000";
let bvec = "0 1 0.5\n0 0 0.866\n0 0 0";
let scheme = read_fsl_scheme(bval, bvec)?;

let (bval_out, bvec_out) = write_fsl_scheme(&scheme);
```

`parse_fsl_bval` and `parse_fsl_bvec` provide standalone parsing when the
caller already has validated b-values or directions separately.

### MRtrix `.mif` DW_scheme

MRtrix embeds the gradient table in the `.mif` container header as a
`DW_scheme` key. The value is a matrix with four columns per volume:
gradient x, y, z (unit direction, or [0,0,0] for b = 0) and b-value in
s/mm².

Directions are in **image-axis coordinates** (`GradientFrame::ImageAxis`).

```rust,ignore
use ritk_diffusion_scheme::mrtrix::{read_mrtrix_scheme, write_mrtrix_scheme};

let header_text = "\
DW_scheme: 3,4
0,0,0,0
1,0,0,1000
0.5,0.866,0,1000
";
let scheme = read_mrtrix_scheme(header_text)?;
let dw_block = write_mrtrix_scheme(&scheme);
```

The reader validates the four-column structure, rejects missing or malformed
dimension declarations, and checks that every component is parseable and
finite. The writer emits the `DW_scheme: N,4` header block ready for
inclusion in a `.mif` file.

### NA-MIC DWI NRRD

The NA-MIC DWI convention stores gradient metadata as NRRD key-value pairs.
One nominal `DWMRI_b-value` is combined with per-volume
`DWMRI_gradient_NNNN` direction keys whose four-digit index starts at 0000.

Directions are read through the NRRD measurement frame and returned in
**RITK physical LPS coordinates** (`GradientFrame::Lps`).

```rust,ignore
use ritk_nrrd::read_nrrd_gradient_scheme;

let scheme = read_nrrd_gradient_scheme("dwi.nrrd")?;
```

The reader validates consecutive index ordering, rejects non-contiguous or
out-of-range indices, and handles `DWMRI_NEX` and B-matrix encodings by
failing explicitly rather than guessing.

### DICOM

DICOM diffusion metadata lives in two standard top-level tags:

- Diffusion b-value `(0018,9087)` — s/mm²
- Diffusion Gradient Orientation `(0018,9089)` — unit direction in the
  **patient coordinate system** (`GradientFrame::Lps`)

```rust,ignore
use ritk_dicom::diffusion::read_dicom_gradient_scheme_from_files;

let scheme = read_dicom_gradient_scheme_from_files(&dicom_paths)?;
```

When standard tags are absent, the reader attempts extraction from vendor
private blocks. **Siemens CSA headers** (`(0029,1020)` series or
`(0029,1010)` image level) carry `B_value` and `DiffusionGradientDirection`
in a binary SV10 table. GE and Philips private groups are recognised as
present but return `Ok(None)` until a vendor-specific decoder is added.

The reader does not infer volume grouping from a directory or guess
enhanced functional groups; those require a separate sequence-aware
parser. For an unweighted frame, a finite zero b-value with no orientation
is mapped to the required zero vector.

## Cross-codec verification

ADR 0036 verification condition 8 requires that every codec round-trip
recovers the same typed scheme, and that all four codecs agree on one
dataset expressed in all four conventions. The cross-codec test in
`crates/ritk-diffusion-scheme/tests/cross_codec.rs` writes and reads a
single-shell scheme through all four codecs and asserts identity:

```text
FSL round-trip    → scheme_fsl
MRtrix round-trip → scheme_mrtrix
NRRD round-trip   → scheme_nrrd
DICOM round-trip  → scheme_dicom

assert scheme_fsl == scheme_mrtrix == scheme_nrrd == scheme_dicom
```

A multi-dataset variant repeats the check against the ds002087 (single-shell
b = 1 000) and ds004666 (multi-shell) OpenNeuro datasets when their test
data has been downloaded.

## Error types

`GradientSchemeError` is a `#[non_exhaustive]` enum covering every
validation failure with an acquisition index where applicable:

| Variant | Condition |
|---|---|
| `Empty` | Scheme has no volumes |
| `InvalidWeighting { index, value }` | Negative, NaN, or infinite b-value |
| `InvalidDirection { index, reason }` | Non-finite, non-zero b0, or non-unit DWI vector |
| `InvalidToken { field, token }` | Unparseable numeric token in a metadata file |
| `InvalidBVectorTable` | FSL b-vector rows do not match |
| `InvalidMrtrixHeader` | MRtrix DW_scheme is malformed |
| `LengthMismatch { weightings, directions }` | Count mismatch in paired FSL files |
| `InvalidRotation` | Non-orthonormal, non-proper, or non-finite rotation matrix |
| `RotationCountMismatch { expected, actual }` | Per-volume rotation count ≠ volume count |

Every error carries enough context to identify the exact volume or metadata
field that failed, supporting actionable diagnostics at the format boundary.

## Invariant

`ritk-diffusion-scheme` is the only crate that constructs
`GradientScheme`. Format crates convert raw bytes to validated entries at
their boundaries; consumers receive an already-validated scheme. This
prevents diffusion models from silently operating on unvalidated or
frame-mismatched metadata.
