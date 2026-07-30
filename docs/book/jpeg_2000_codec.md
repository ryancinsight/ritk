# JPEG 2000 Native Codec

RITK implements the JPEG 2000 codestream path in Rust. It does not call
OpenJPEG, `openjp2`, or another C codec through FFI. This matters at a medical
image boundary: malformed dimensions, precision, or packet lengths remain Rust
errors instead of crossing an unsafe foreign interface.

JPEG 2000 is a transform codec, not the older block-DCT JPEG format. The Part 1
pipeline is:

```text
integer samples
    │
    ├─ unsigned DC level shift
    ▼
discrete wavelet transform (5/3 or 9/7)
    ▼
quantization (none for reversible 5/3)
    ▼
EBCOT code-block coding and tier-2 packets
    ▼
SOC | SIZ | COD | QCD | tile-part | EOC
```

The normative codestream and transform definitions are in
[ITU-T T.800 (11/2015), identical to ISO/IEC 15444-1](https://handle.itu.int/11.1002/1000/12682),
especially Annexes A, B, D, E, F, and G. DICOM embeds a raw J2K codestream in
encapsulated Pixel Data rather than a JP2 file container. DICOM PS3.5
[Section 8.2.4](https://dicom.nema.org/medical/dicom/current/output/chtml/part05/sect_8.2.4.html)
defines that encoding boundary, while
[Section A.4.4](https://dicom.nema.org/medical/dicom/current/output/chtml/part05/sect_a.4.4.html)
maps the reversible path to transfer syntax UID `1.2.840.10008.1.2.4.90` and
the reversible or irreversible path to `.91`.

## Reversible 5/3 transform

The reversible transform uses integer lifting. For an interleaved one-
dimensional signal, the forward predict and update steps are

```text
d[n] = x[2n+1] - floor((x[2n] + x[2n+2]) / 2)
s[n] = x[2n]   + floor((d[n-1] + d[n] + 2) / 4)
```

Whole-sample symmetric extension supplies samples at image boundaries. RITK
applies the one-dimensional transform by columns and rows at each resolution,
placing LL, HL, LH, and HH bands in Mallat layout. The inverse lifting steps
are exact integer inverses. With no quantization, decoded samples therefore
equal the source samples bit-for-bit.

## Irreversible 9/7 transform

The irreversible path uses the floating-point 9/7 lifting transform and
dead-zone scalar quantization. RITK currently uses a unit quantization step:
the path is lossy because floating-point lifting and coefficient quantization
do not preserve every integer exactly, but it does not yet expose a target
bit-rate or quality parameter. Treat codestream size as an observed result,
not a requested rate.

The worked example deliberately renders the 9/7 absolute error with its own
magnified scale. Source and reconstructed anatomy otherwise appear nearly
identical when all three image panels share the same 12-bit display range.

## Encoding through the public API

```rust,ignore
use ritk_codecs::jpeg_2000::encoder::{
    encode_grayscale_j2k, WaveletTransform,
};
use ritk_codecs::PixelSignedness;

let pixels = [0, 128, 512, 4095];
let codestream = encode_grayscale_j2k(
    &pixels,
    2,
    2,
    12,
    PixelSignedness::Unsigned,
    1,
    WaveletTransform::Reversible,
)?;
```

The encoder validates before constructing the transform buffer:

- rows and columns are nonzero and their product fits `usize`;
- `pixels.len()` equals `rows × columns`;
- precision is in the current native encoder range, 1–16 bits;
- decomposition depth does not exceed the geometry's meaningful resolution
  count; and
- every sample fits the declared signed or unsigned range.

These are errors, not panic conditions. The current `i32` entropy path
intentionally rejects DICOM's wider legal precision range rather than
truncating values or risking lifting overflow. Color components, chroma
subsampling, multiple tiles, custom precincts, rate control, and JP2
containers are also outside the current encoder contract.

## Memory and packet structure

Unsigned JPEG 2000 components are centered by subtracting
`2^(precision-1)`. RITK applies that offset while constructing the Mallat
transform buffer. It does not first allocate a second, complete shifted image.
For a reversible image of `N` samples, this removes `4N` bytes of peak
full-image storage and one pass that writes those bytes. Code-block coefficient
vectors remain bounded by the 64 × 64 code-block geometry.

The encoder emits one image-wide tile, one quality layer, LRCP progression,
one precinct per resolution, and 64 × 64 nominal code-blocks. The decoder
validates SIZ and SOT geometry before allocation and bounds packet-header
lengths before consuming packet data.

## Interoperability evidence

RITK tests the native paths at three levels:

1. analytical round trips require zero error for reversible 5/3;
2. captured OpenJPEG 2.5.4 codestreams exercise independent producer/consumer
   interoperability without loading OpenJPEG at test time; and
3. malformed geometry and packet tests require typed rejection instead of a
   panic or partial image.

Continue with the [worked codec example](examples/jpeg_2000_codec.md) to read
the reconstruction and error panels.
