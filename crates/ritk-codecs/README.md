# ritk-codecs

RITK-native pixel codec implementations for [RITK](https://github.com/ryancinsight/ritk).

Single source of truth for DICOM pixel codec primitives: pixel layout
arithmetic, native sample decoding, and encapsulated transfer-syntax decoders.
It also owns fixed-width sample buffers shared by RITK volume-format adapters.
The buffers retain the type selected by the format header; adapters choose
whether to require an exact conversion or apply an explicit cast.
`Conversion::report` describes possible value changes from the source and
target types without claiming to count data-dependent changes.

See the [typed sample I/O migration guide](../../docs/migration_typed_sample_io.md)
for exact and explicit-cast examples.

| Codec | Implementation |
|---|---|
| JPEG 2000 | ISO 15444-1, multi-level reversible 5/3 and irreversible 9/7 |
| JPEG | 8/12-bit DCT and 2–16-bit lossless |
| JPEG-LS | RITK-native |
| PackBits | RITK-native |
| RLE Lossless | RITK-native |

Every codec is pure Rust; none links a C or C++ library.

## Usage

```toml
[dependencies]
ritk-codecs = "0.6.0"
```
