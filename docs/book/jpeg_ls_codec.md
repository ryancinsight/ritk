# JPEG-LS Native Codec

RITK implements grayscale JPEG-LS in Rust. The encoder and decoder do not call
CharLS, GDCM, or another C/C++ codec through FFI. This keeps malformed marker
handling, dimension validation, entropy state, and reconstructed samples
inside Rust's checked ownership boundary.

JPEG-LS is a predictive codec designed for low complexity. It is distinct
from the block-DCT JPEG family and from wavelet-based JPEG 2000. The normative
baseline is
[ITU-T T.87](https://www.itu.int/rec/T-REC-T.87-199806-I/en), equivalent to
ISO/IEC 14495-1. DICOM PS3.5
[section 8.2.3](https://dicom.nema.org/medical/dicom/current/output/chtml/part05/sect_8.2.3.html)
encapsulates JPEG-LS streams and assigns transfer syntax UIDs:

- `1.2.840.10008.1.2.4.80` for lossless coding;
- `1.2.840.10008.1.2.4.81` for near-lossless coding.

For `MONOCHROME1` and `MONOCHROME2`, DICOM permits one sample per pixel,
8- or 16-bit allocation, and 2–16 stored bits. RITK's decoder accepts that
2–16-bit precision range. The current public encoder accepts unsigned
single-component 8–16-bit samples.

## Prediction and local gradients

For the sample at row `y`, column `x`, JPEG-LS uses four causal neighbors:

```text
c  b  d       previous reconstructed row
a  x          current reconstructed row
```

The edge-detecting predictor is

```text
          min(a, b),        if c >= max(a, b)
Px =      max(a, b),        if c <= min(a, b)
          a + b - c,        otherwise
```

The first row and column use the boundary rules defined by T.87 Annex A.
Three local gradients,

```text
g1 = d - b
g2 = b - c
g3 = c - a
```

are quantized into a context. Flat neighborhoods enter run mode; all other
neighborhoods enter regular mode.

## Regular mode

Regular mode predicts `x`, computes the prediction error, maps its signed value
to a nonnegative integer, and writes a Golomb-Rice code. Context statistics
adapt the Rice parameter and a small bias correction as samples are processed.
The decoder performs the inverse mapping and reconstructs the same causal
sample before moving to the next column.

For lossless coding, the quantized error equals the exact prediction error:

```text
NEAR = 0  =>  reconstructed[x] = source[x]
```

For near-lossless coding, the error is quantized in steps of
`2 × NEAR + 1`. The reconstruction contract is the exact per-sample bound

```text
abs(reconstructed[x] - source[x]) <= NEAR
```

This is not an average quality target. Every decoded sample must satisfy it.

## Run mode

When all three quantized gradients are zero, the neighborhood is locally flat.
Run mode emits adaptive run-length segments using the standard `J` table.
When a run ends before the row boundary, one run-interruption sample is coded
against the appropriate neighbor context.

Run mode explains why a smooth medical image can compress well without a
transform: long predictable regions require few coded bits, while boundaries
return to regular mode.

## Encoding through the public API

```rust,ignore
use ritk_codecs::jpeg_ls::encoder::encode_grayscale_jpeg_ls;

let pixels = [0, 128, 512, 4095];
let lossless = encode_grayscale_jpeg_ls(&pixels, 2, 2, 12, 0)?;
let near_lossless = encode_grayscale_jpeg_ls(&pixels, 2, 2, 12, 3)?;
```

The encoder validates all external metadata before allocating entropy state:

- rows and columns are nonzero and fit the 16-bit SOF55 fields;
- the checked product equals `pixels.len()`;
- precision is in the implemented 8–16-bit encoder range;
- every sample is no greater than `2^precision - 1`; and
- `NEAR` fits the SOS field and is no greater than `MAXVAL / 2`.

Invalid input returns `JpegLsEncodeError`. RITK does not clamp geometry,
truncate `NEAR`, or emit a partial stream.

## Bounded reconstruction memory

Prediction reads only the previous row and the reconstructed prefix of the
current row. RITK therefore shares one `ReconstructionRows` implementation
between encoder and decoder:

```text
previous row: cols × i32
current row:  cols × i32
total:        2 × cols × 4 bytes
```

The previous implementation retained `(rows + 1) × cols` values. At
512 × 512, reconstruction scratch decreases from 1,050,624 bytes to 4,096
bytes. The new bound is `O(cols)` and does not grow with image height. The
decoded output still requires one value per pixel because the public API
returns the complete image.

## DICOM boundary

DICOM image metadata and the JPEG-LS interchange header must agree on rows,
columns, sample count, and precision. RITK validates that geometry before scan
allocation. One marker pass checks every declared segment bound and returns
the exact entropy slice, avoiding a second scan through compressed bytes. It
supports the single-component, non-interleaved profile used by the grayscale
native path. Unsupported color/interleaved streams, mapping tables, restart
intervals, non-preset LSE records, and nonzero point transforms return errors;
they do not fall back to a foreign codec.

## Verification

RITK checks the codec at complementary levels:

1. exact lossless round trips over uniform, gradient, edge, run-interruption,
   8-, 12-, and 16-bit fixtures;
2. randomized lossless equality and near-lossless error-bound properties;
3. complete DICOM encapsulation and decode round trips;
4. malformed geometry, precision, `NEAR`, marker lengths, interleave, mapping
   table, and restart-interval rejection, plus bounded arbitrary marker bytes;
5. a structural assertion that the reconstruction workspace is exactly two
   rows; and
6. unchanged 512 × 512 Criterion encode/decode workloads.

Continue with the [worked JPEG-LS example](examples/jpeg_ls_codec.md) to see
why the source and near-lossless anatomy panels can look almost identical and
how the magnified error panel exposes the actual changes.
