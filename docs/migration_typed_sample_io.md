# Typed sample I/O migration

This guide covers the shared sample API introduced by ADR 0053. RITK format
crates own file decoding, encoding, and conversion. Application code uses the
RITK format API; it does not decode medical-image samples itself. The current
unified format matrix lives in the [RITK README](../README.md#io-ritk-io).

## Read stored samples without changing their type

The legacy `decode_bytes_to_f32` remains available while format adapters
migrate. New adapters select the stored type from the format header, then
decode into a `SampleBuffer` using the byte order owned by `consus-core`:

```rust
use consus_core::ByteOrder;
use ritk_codecs::sample::{SampleBuffer, SampleType};

let stored = SampleBuffer::decode(
    &[0x2a, 0x00],
    SampleType::I16,
    ByteOrder::LittleEndian,
)
.expect("one complete i16 sample");
assert_eq!(stored, SampleBuffer::I16(vec![42]));
```

## Choose a conversion policy

`SampleBuffer::into_vec::<T>()` moves the allocation when the stored type is
`T`, performs a lossless widening when the entire stored type fits in `T`,
and returns the original buffer for every other type pair:

```rust
use ritk_codecs::sample::{SampleBuffer, SampleType};

let stored = SampleBuffer::decode(
    &[0x2a, 0x00],
    SampleType::I16,
    consus_core::ByteOrder::LittleEndian,
)
.expect("one complete sample");
let values = stored.into_vec::<f32>().expect("i16 widens exactly to f32");
assert_eq!(values, [42.0]);
```

If a caller accepts a potentially lossy conversion, it names `Cast`. Inspect
the type-level report before conversion; `Possible` means some values for the
stored/requested pair may change, not that this buffer was checked sample by
sample. `Applied` means the policy permits the conversion:

```rust
use ritk_codecs::sample::{
    Cast, Conversion, ConversionDisposition, SampleBuffer, ValueChange,
};

let stored = SampleBuffer::U16(vec![300]);
let report = Cast.report::<u8>(&stored);
assert_eq!(report.value_change(), ValueChange::Possible);
assert_eq!(report.disposition(), ConversionDisposition::Applied);
assert_eq!(report.sample_count(), 1);

let values = Cast
    .convert::<u8>(stored)
    .expect("Cast accepts an explicit narrowing conversion");
assert_eq!(values, [44]);
```

`Exact.report` uses the same type-level analysis and reports `Refused` for
non-widening pairs. Conversion errors retain the original `SampleBuffer`, so
an exact refusal does not consume the stored samples.

Do not route integer samples through `f32` or `f64`; that can round integer
samples before the requested conversion occurs. RITK may widen a narrower
integer exactly to its signed or unsigned carrier, or widen `f32` exactly to
`f64`, before converting once to the requested destination type.
Format-specific rescale coefficients remain separate from stored samples and
are applied only under the format reader's documented policy.

## Use the shared byte-order type

New sample codecs and format adapters use `consus_core::ByteOrder`. The legacy
`ritk_codecs::ByteOrder` remains available until its last format caller
migrates, then it is removed with the byte-to-`f32` helper.

## Migrate format call sites

Typed reader and writer signatures land with their format-specific RITK
increments. Bind the requested sample type at the call site, then select
`Exact` or `Cast` explicitly. The shared `ritk-io` API owns cross-format
conversion planning and loss reporting; GUI, CLI, and Python callers should
not duplicate that policy.
