# ritk-codecs

RITK-native pixel codec implementations for [RITK](https://github.com/ryancinsight/ritk).

Single source of truth for DICOM pixel codec primitives: pixel layout
arithmetic, native sample decoding, and encapsulated transfer-syntax decoders.

| Codec | Implementation |
|---|---|
| JPEG 2000 | ISO 15444-1, multi-level reversible 5/3 and irreversible 9/7 |
| JPEG | Baseline, Extended, and Lossless grayscale |
| JPEG-LS | RITK-native |
| PackBits | RITK-native |
| RLE Lossless | RITK-native |

Every codec is pure Rust; none links a C or C++ library.

## Usage

```toml
[dependencies]
ritk-codecs = "0.6.0"
```
