# JPEG 2000 Native Codec

RITK implements the JPEG 2000 codestream path in Rust. It does not call
OpenJPEG, `openjp2`, or another C codec through FFI. This matters at a medical
image boundary: malformed dimensions, precision, or packet lengths remain Rust
errors instead of crossing an unsafe foreign interface.

## Decoder contract

The native decoder currently accepts a complete grayscale codestream with one
component, LRCP progression, no progression changes, no multiple-component
transform, inline packet headers, one tile-part per tile, and unit component
sampling. Packet coding uses default precincts, 64 × 64 nominal code-blocks,
default code-block style, no SOP/EPH markers, and at most 32-bit component
precision, matching the decoder's signed coefficient representation. These
constraints match the encoder and the grayscale DICOM workflow
shown in this chapter. The decoder preflights main and tile headers plus complete
tile coverage before allocating output. Color or other multi-component streams,
chroma subsampling, progression overrides, packed packet headers, multi-part
tiles, non-default packet coding, and MCT are typed errors. Returning an error is necessary here: replaying
the same packet cursor independently for each component can produce
plausible-looking but duplicated channels.

The codestream boundary is exact. Every marker segment must include a valid
length and remain inside its tile-part or codestream boundary, SOD is found by
structural marker parsing rather than a byte-pattern search, every tile declared
by SIZ must appear, and EOC must terminate the stream. Every expected LRCP packet
header must also be present. `Psot = 0` extends only the final tile-part to the
validated terminal EOC; marker-looking bytes inside a length-delimited COM
payload are never treated as boundaries. Arbitrary bytes after EOC are rejected.
The one permitted exception is a single zero byte when an odd-length codestream
must be padded to an even DICOM Fragment Item length, as required by DICOM PS3.5
[Section 8.2](https://dicom.nema.org/medical/dicom/current/output/chtml/part05/sect_8.2.html).

An included code-block must declare a nonzero packet-body length. Tier-1 also
checks the pass count against the code-block's bit-plane budget and tracks MQ
terminal fill consumption. JPEG 2000 arithmetic termination permits bounded
look-ahead into an artificial `0xFF 0xFF` marker; consuming more than the two
terminal reads used by OpenJPEG's predictable-termination check is treated as
truncation. A decoded first tile followed by a truncated marker tail or entropy
body therefore returns an error instead of a partially populated image whose
missing coefficients or voxels appear as valid zeros.

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
dead-zone scalar quantization. `QuantizationStep` controls the coefficient
interval width `Δ`: larger steps map more nearby coefficients to the same
integer index, usually reducing packet bytes while increasing reconstruction
error. JPEG 2000 stores `Δ` as a five-bit exponent and eleven-bit mantissa in
QCD. RITK rounds the requested positive finite step to the nearest
representable QCD value for each subband, then uses that exact represented
value for both coefficient quantization and metadata.

This is scalar quality control, not target-rate control. The same `Δ` can
produce different byte lengths for different anatomy, noise, precision, and
wavelet depth. Treat codestream size and PSNR as measured outputs. Target-byte
or target-bitrate encoding requires multiple coding-pass truncation points and
rate-distortion optimization, which this one-layer encoder does not claim.

The worked example shows unit and coarse (`Δ = 32`) reconstructions side by
side, prints each actual byte count and PSNR, and renders the coarse absolute
error with its own magnified black→red→yellow scale. That separate palette and
scale make changes visible even where the shared 12-bit anatomy display range
hides them.

## Encoding through the public API

```rust,ignore
use ritk_codecs::jpeg_2000::encoder::{
    encode_grayscale_j2k, Jpeg2000Encoding,
};
use ritk_codecs::PixelSignedness;

let pixels = [0, 128, 512, 4095];
let codestream = encode_grayscale_j2k(
    &pixels,
    2,
    2,
    12,
    PixelSignedness::Unsigned,
    Jpeg2000Encoding::Lossless {
        decomposition_levels: 1,
    },
)?;
```

The encoder validates before constructing the transform buffer:

- rows and columns are nonzero and their product fits `usize`;
- `pixels.len()` equals `rows × columns`;
- precision is in the current native encoder range, 1–16 bits;
- decomposition depth does not exceed the geometry's meaningful resolution
  count; and
- every sample fits the declared signed or unsigned range.

Lossy construction validates the scalar step separately:

```rust,ignore
use ritk_codecs::jpeg_2000::encoder::{Jpeg2000Encoding, QuantizationStep};

let encoding = Jpeg2000Encoding::Lossy {
    decomposition_levels: 3,
    quantization_step: QuantizationStep::new(8.0)?,
};
```

Zero, negative, NaN, and infinite steps fail at `QuantizationStep::new`.
A finite step whose QCD exponent is unavailable for any requested subband
fails before wavelet allocation.

These are errors, not panic conditions. The current `i32` entropy path
intentionally rejects DICOM's wider legal precision range rather than
truncating values or risking lifting overflow. Color components, chroma
subsampling, multiple tiles, custom precincts, target-rate control, and JP2
containers are also outside the current encoder contract.

## Memory and packet structure

Unsigned JPEG 2000 components are centered by subtracting
`2^(precision-1)`. RITK applies that offset while constructing the Mallat
transform buffer. It does not first allocate a second, complete shifted image.
For a reversible image of `N` samples, this removes `4N` bytes of peak
full-image storage and one pass that writes those bytes. Code-block coefficient
vectors remain bounded by the 64 × 64 code-block geometry. Irreversible encode
similarly retains one `f32` Mallat volume and quantizes one code block at a
time; it no longer materializes a second complete `i32` coefficient volume.

The encoder emits one image-wide tile, one quality layer, LRCP progression,
one precinct per resolution, and 64 × 64 nominal code-blocks. Before output
allocation, the decoder validates SIZ/SOT geometry, tile-header marker extents,
single-part tile coverage, and EOC. Packet decode then requires each LRCP packet
header and bounds its declared body length before consuming packet data.

## Interoperability evidence

RITK tests the native paths at three levels:

1. analytical round trips require zero error for reversible 5/3;
2. quantization tests require each requested step to round to one QCD pair and
   require packet coefficients to use the reconstructed QCD value;
3. captured OpenJPEG 2.5.4 codestreams exercise independent producer/consumer
   interoperability without loading OpenJPEG at test time; and
4. malformed geometry, unsupported component/progression overrides, truncated
   markers, zero-length or exhausted entropy bodies, incomplete packets,
   multi-part declarations, missing tiles, invalid `Psot = 0` boundaries,
   trailing payload, and missing EOC require typed rejection instead of a panic
   or partial image.

Continue with the [worked codec example](examples/jpeg_2000_codec.md) to read
the reconstruction and error panels.
