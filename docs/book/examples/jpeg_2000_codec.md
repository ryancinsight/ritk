# Example: Reversible and Irreversible JPEG 2000

This example creates one deterministic 96 × 96, 12-bit medical-style phantom.
It encodes and decodes that same image twice through RITK's public native API:
three-level reversible 5/3 and three-level irreversible 9/7.

![Source phantom, exact reversible reconstruction, irreversible reconstruction, and magnified absolute error](../figures/jpeg_2000_codec.svg)

All three anatomy panels use the same `[0, 4095]` display range. The reversible
panel must therefore match the source both visually and numerically. The
fourth panel is not another independently contrast-stretched anatomy image: it
is `abs(irreversible - source)`, displayed on `[0, max error]`. Its independent
scale and the printed maximum error make the small 9/7 changes visible.

The metrics below the panels are computed from the same arrays that are
rendered:

- the 5/3 mismatch count is an exact equality oracle and must be zero;
- each codestream size is the actual returned byte length;
- 9/7 maximum error is measured per sample; and
- PSNR uses the declared 12-bit peak value, 4095.

## Source and command

Source: `crates/ritk-codecs/examples/book_jpeg_2000.rs`

```text
cargo run -p ritk-codecs --example book_jpeg_2000 -- \
  docs/book/figures/jpeg_2000_codec.svg
```

The example fails rather than writing a misleading figure if reversible
reconstruction differs at any sample, the lossy error panel is degenerate, or
the rendered arrays do not match the declared geometry.

## Adapt the workflow

For an unsigned grayscale image:

```rust,ignore
let codestream = encode_grayscale_j2k(
    &samples,
    rows,
    columns,
    bits_stored,
    PixelSignedness::Unsigned,
    decomposition_levels,
    WaveletTransform::Reversible,
)?;

let reconstructed = decode_jpeg2000_fragment(
    &codestream,
    PixelLayout {
        rows: usize::try_from(rows)?,
        cols: usize::try_from(columns)?,
        samples_per_pixel: 1,
        bits_allocated: u16::try_from(bits_stored)?,
        pixel_representation: PixelSignedness::Unsigned,
        rescale_slope: 1.0,
        rescale_intercept: 0.0,
    },
)?;
```

Use `WaveletTransform::Reversible` for a lossless DICOM transfer syntax. The
current irreversible path demonstrates the 9/7 transform and quantization
contract; it is not a rate-control API.
