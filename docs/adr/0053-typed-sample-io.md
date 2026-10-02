# ADR 0053: Typed sample I/O

- Status: Accepted
- Date: 2026-09-30
- Board: `RITK-TYPED-SAMPLES-001` through `RITK-TYPED-SAMPLES-011`
- Delivery: [RITK PR #696](https://github.com/ryancinsight/ritk/pull/696) (foundation)

## Context

Every volume reader converts its samples to `f32` and every writer emits
`float32`, whatever the file stores (audit of `main` at `8a401b73`):

| Format | Stored types read | Written | Defects beyond the cast |
| --- | --- | --- | --- |
| NIfTI | u8, i16, i32, u32, f32 | f32 | int8, uint16, int64, uint64 and float64 files are rejected; `scl_slope`/`scl_inter` are never applied |
| Analyze 7.5 | u8, i16, i32, f32, f64 | f32 | |
| MGH/MGZ | u8, i16, i32, f32 | f32 | |
| NRRD | u8, i8, u16, i16, u32, i32, f32, f64 | f32 | 64-bit integer types unrecognized |
| MetaImage | u8, i16, u16, i32, u32, f32, f64 | f32 | `MET_CHAR` and 64-bit types unrecognized |
| MIF | u8 … u32, f32, f64 | f32 | |
| MINC 2 | u8 … i32, f32, f64 | f32 | |
| DICOM | 8/16/24/32-bit integers | u16 | writer re-quantizes every image to u16 with a computed rescale; rescale coefficients truncated to `f32` |
| VTK legacy | u8, i16, u16, i32, u32, f32, f64 | f32 | |

An `f32` holds integers exactly only up to 2^24, so any `i32`/`u32` sample
above that, every 64-bit integer above 2^24, and every `f64` needing more than
24 significand bits is rounded on read; a label map or a CT stored as `i16`
cannot be written back in its own type. The byte-order and element-type
conversion was repeated per format: a shared `decode_bytes_to_f32`
(`ritk-codecs`, used by NRRD and MetaImage) plus local copies in NIfTI,
Analyze, MGH, MIF, MINC and VTK, and a second `ByteOrder` enum beside
`consus_core::ByteOrder`. The `ritk-io` dispatch (`ImageFormat`,
`read_image_native`) is fixed to `Image<f32, NativeBackend, 3>` and has no MINC
or MIF entry.

## Decision

1. **One sample vocabulary in `ritk_codecs::sample`.** `SampleType` is the
   closed runtime descriptor of the ten fixed-width types (u8 … u64, i8 … i64,
   f32, f64). `Sample` is its compile-time counterpart, implemented for exactly
   those ten primitives, with `const TYPE: SampleType` routing a Rust type to
   its descriptor so a `match` on `T::TYPE` folds per monomorphization.
   `Sample: coeus_core::Scalar`, so every sample type is an `Image` element.
   `ritk-codecs` is the home because it reaches `coeus-core` and `consus-core`
   without depending on `ritk-image`, which every format crate can take.
2. **Readers return the stored type.** `SampleBuffer` owns a vector in the
   stored type, selected by the header, dispatched by exhaustive `match`.
   `SampleBuffer::into_vec::<T>()` moves the vector out unchanged when `T` is
   the stored type and otherwise converts each sample directly from its stored
   type to `T` with the primitive cast (`num_traits::AsPrimitive`) — never
   through a third type, so an `i64` read as `i64` is exact.
3. **Bulk decode, byte order resolved once.** `decode_samples::<T>` copies the
   bytes into the typed vector in one bulk copy and reverses each sample's
   bytes in place only when the file order differs from the target's.
   `consus_core::ByteOrder` is the only byte-order type.
4. **Rescale is separate from storage.** A linear rescale (NIfTI
   `scl_slope`/`scl_inter`, DICOM Rescale Slope/Intercept, Analyze `funused1`,
   MINC `valid_range` to `image-min`/`image-max`) is carried as `f64`
   coefficients beside the stored samples. It applies in the arithmetic of a
   floating-point target; a non-identity rescale requested into an integer
   target is a typed error, and the stored-sample read serves callers that
   need exact stored values.
5. **Writers emit the type of `T`.** Each format maps `T::TYPE` to its on-disk
   code; a type the format cannot store (MGH has four, Analyze five) is a typed
   error naming the format's set.
6. **Dispatch generic over `T`.** `ritk-io` reads and writes `Image<T, B, 3>`
   for every `T: Sample`, adds MINC and MIF, and exposes a stored-type read
   returning the `SampleBuffer`. Python and viewer surfaces remain `f32`
   consumers of that dispatch; typed Python arrays are outside this decision.

## Delivery order

Foundation (this module, NRRD and MetaImage converted, `decode_bytes_to_f32`
and the second `ByteOrder` deleted) → NIfTI (all ten types and rescale — the
correctness defect) → MGH, Analyze, MIF, NRRD, MetaImage, VTK, MINC, each
deleting its local decode → `ritk-io` dispatch → DICOM. Each increment is one
board item and one PR.

## Rejected alternatives

- **A generic reader over `T` with no runtime buffer.** The element type comes
  from the header, so the reader must branch on it once; without
  `SampleBuffer` each reader re-implements that ten-way branch.
- **Converting through `f64` (`Scalar::to_f64`/`from_f64`).** Exact for every
  type except `i64`/`u64` above 2^53, which is the precision loss this
  decision removes, moved rather than fixed.
- **Keeping `f32` as the working type and adding a parallel typed path.** Two
  decoders per format is the duplication this decision consolidates.
- **`consus_core::decode_extend` for the bulk decode.** Correct, but a
  per-sample decode; a single bulk copy plus an in-place swap only on foreign
  byte order is the lower-cost form for packed buffers, and the locked
  `consus-core` revision predates it.

## Consequences

- Call sites of readers made generic over `T` that do not bind the result type
  need an explicit `f32` (or other) annotation.
- Formats whose writers now emit the input type produce files other tools read
  in that type; a caller wanting `float32` output converts first.

## Overturning evidence

A consumer measurably slowed by `SampleBuffer`'s one-time branch, or a format
storing a sample type outside the ten (complex, RGB, bit-packed), would reopen
the closed set.
