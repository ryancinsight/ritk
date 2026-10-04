# ritk-codecs

RITK-native pixel codec implementations for [RITK](https://github.com/ryancinsight/ritk).

Single source of truth for RITK image-format sample and codec primitives:
fixed-width stored-sample decoding, pixel layout arithmetic, and encapsulated
transfer-syntax decoders.

| Codec | Implementation |
|---|---|
| JPEG 2000 | ISO 15444-1, multi-level reversible 5/3 and irreversible 9/7 |
| JPEG | 8/12-bit DCT and 2–16-bit lossless |
| JPEG-LS | RITK-native |
| PackBits | RITK-native |
| RLE Lossless | RITK-native |

Every codec is pure Rust; none links a C or C++ library.

## Stored samples

`SampleBuffer` decodes and encodes signed and unsigned 8-, 16-, 32-, and 64-bit
integers plus 32- and 64-bit floats in either byte order. It preserves the
declared sample type and floating-point bit patterns. A partial trailing sample
is an error. Extracting another sample type returns the original buffer in the
error, without rounding or discarding data. `SampleBuffer::write_to` streams the
encoded bytes to a `std::io::Write` implementation without allocating a second
buffer the size of the image; file adapters should pass a buffered writer.

The codec layer does not depend on Coeus' algorithm scalar contract. Format
adapters retain geometry and format-specific intensity calibration beside the
stored samples, then make any numeric conversion explicit at the image or
algorithm boundary. The existing `byte_decode` API remains available while
format readers migrate. See [ADR 0053](../../docs/adr/0053-typed-sample-io.md) for
the stored-sample ownership and conversion boundary.

## Usage

```toml
[dependencies]
ritk-codecs = "0.6.0"
```
