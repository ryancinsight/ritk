# ritk-codecs

RITK's shared sample-buffer and DICOM pixel-codec crate. Format-specific
readers and writers live in their RITK format crates; this crate provides
typed sample decoding, explicit numeric conversion, and DICOM transfer-syntax
codec primitives.

`SampleBuffer` keeps the primitive type selected by a file header until a
format adapter chooses a conversion. Taking a vector out with `try_into_vec`
succeeds only when its type matches the stored type. `try_convert` rejects any
changed numeric value or floating-point bit representation. Call
`convert_lossy` only when the caller accepts a cast, and inspect its
`ConversionReport` before using the result.

```rust
use ritk_codecs::sample::SampleBuffer;

let stored = SampleBuffer::I32(vec![(1_i32 << 24) + 1]);
let (rounded, report) = stored
    .convert_lossy::<f32>()
    .expect("small output allocation")
    .into_parts();

assert_eq!(rounded, [16_777_216.0]);
assert_eq!(report.changed_samples, 1);
assert_eq!(report.first_changed_sample, Some(0));
```

The crate also supplies native JPEG, JPEG-LS, JPEG 2000, PackBits, and RLE
implementations for DICOM pixel data. These codecs do not link C or C++ code.
