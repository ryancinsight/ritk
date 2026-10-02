# ADR 0053: RITK owns typed image-format I/O

- Status: Accepted
- Date: 2026-09-30
- Delivery: [RITK PR #696](https://github.com/ryancinsight/ritk/pull/696)
- Revised: 2026-10-02 — retain the legacy `f32` byte decoder until every format
  adapter has migrated; remove it with the final remaining caller.

## Context

At `origin/main` `20461c2e916542735fb1f5e14c7365e2ed94254e`,
`ritk-io::NativeImage` is `Image<f32, NativeBackend, 3>`. Its dispatch routes
the supported scalar formats through that carrier, so a stored `i32`, `u64`,
or `f64` value may change before a caller can inspect it. Format crates also
repeat sample decoding, and MINC and MIF are not both reachable through the
shared path. DICOM studies have series identity and metadata that a single
volume cannot represent.

## Decision

1. RITK owns every supported medical-image format implementation. Each format
   crate parses and writes its format. `ritk-io` owns format discovery,
   cross-format conversion planning and execution, and reports numeric,
   geometry, and metadata loss before output is written. CLI, Python, and
   Métis call these RITK APIs. Métis owns file selection and presentation;
   format parsing and voxel conversion do not live in the GUI.
2. `ritk_codecs::sample` owns the fixed-width sample vocabulary shared by
   scalar volume formats: the ten primitive integer and IEEE floating-point
   types, a runtime `SampleType`, and an exhaustive `SampleBuffer`. Format
   headers select the stored variant. Packed samples use `consus_core` byte
   order and codecs. The RITK `Sample` contract owns explicit numeric
   conversion semantics; production format paths do not use a generic cast
   trait whose implementation hides the conversion behavior.
3. Conversion is explicit. Exact extraction moves a same-type vector or
   performs a mathematically lossless widening; otherwise it returns the
   untouched buffer. `Cast` applies Rust's primitive numeric conversion rules
   and emits a tracing warning when the stored type cannot widen to the
   requested type. Integer narrowing retains the low target-width bits;
   float-to-integer conversion truncates toward zero and saturates, with NaN
   mapping to zero; integer-to-float and `f64`-to-`f32` conversions round to
   nearest, ties to even. These rules follow the [Rust Reference's numeric
   cast section](https://doc.rust-lang.org/reference/expressions/operator-expr.html#numeric-cast).
   The
   `Conversion::report` value records stored and requested types, sample count,
   whether that type pair admits value changes, and whether the policy applies
   or refuses it. This is a type-level risk report; it does not claim to count
   changed samples. Integer samples widen only to their exact signed or
   unsigned source carrier, and `f32` widens exactly to `f64`; integer to
   float rounding occurs once at the requested precision. Cross-format
   preflight reports type-level and format-level loss before writing.
4. Stored samples and physical rescale are separate. Readers preserve the
   stored values and expose format coefficients separately; applying a
   non-identity rescale to an integer target is an error. Floating-point
   rescale arithmetic executes in the requested floating-point type.
5. Structured data stays in its domain model. DICOM selection and conversion
   retain study and series identity; ambiguous series selection is an error,
   never an implicit choice of the first series. A writer reports metadata or
   geometry it cannot preserve.

## Rejected alternatives

- A shared `f32` carrier loses integer values above its 24-bit significand
  and rounds `f64`; callers cannot recover the source samples afterward.
- Per-format sample enums or decoders duplicate the header-to-type dispatch
  and risk inconsistent byte order and conversion behavior.
- Converting through `f64` still rounds `i64` and `u64` values above 2^53.
- Putting format parsers or cross-format conversion in Métis duplicates RITK
  domain behavior and prevents CLI and Python from using the same contract.
- Choosing the first DICOM series hides an ambiguous input from the user.

## Consequences

New and migrated consumers use `SampleBuffer`, `SampleType`, and
`consus_core::ByteOrder`; lossy conversion is an explicit `Cast`. The legacy
byte-to-`f32` helper and duplicate byte-order type remain available until the
last format reader and caller migrate, then the final dispatch increment
removes them together. Reader and writer signatures become typed as each RITK
format adapter migrates. The `ritk-io` carrier and CLI call sites then migrate
to typed dispatch. The final removals require a major-version migration path;
no release is authorized by this decision.

## Overturning evidence

Reopen this decision if a supported format has a scalar representation the
ten-type vocabulary cannot preserve, or if a measured consumer requires a
different representation with no loss of format fidelity.
