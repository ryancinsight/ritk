# ADR 0058: Read budgets are owned by `ritk-codecs`

- Status: Accepted

## Context

Header fields (`nframes`, `dim`, `DimSize`, SIZ) are claims about a file, and
readers size allocations and inflation from them. `ImageReadBudget` bounds
encoded bytes, decoded bytes, and series volumes, and lived in `ritk-image-io`.
`ritk-image-io` depends on `ritk-codecs` for `SampleBuffer`, so the JPEG 2000
decoder in `ritk-codecs` could not take the budget without a dependency cycle,
and the MGH, MIF, and MetaImage readers had no budget at all. The decode-size
caps of the JPEG and JPEG-LS decoders already live in `ritk-codecs`
(`dimensions.rs`).

## Decision

`ImageReadBudget`, `ImageReadBudgetError`, and `ImageReadResource` move to
`ritk-codecs`, the deepest crate every reader already depends on or can depend
on without a cycle. `ritk-image-io` no longer defines them and does not
re-export them.

Every public reader that sizes work from a header takes an `ImageReadBudget`
and rejects an oversized declaration with a typed error before allocating,
reading, or inflating:

| Reader | Error | Checked before work |
| --- | --- | --- |
| `decode_jpeg2000_fragment` | `Jpeg2000DecodeError::Budget` | fragment length; `f32` output of the layout |
| `read_metaimage` | `MetaImageError::Budget` | inflated payload and `f32` output from `DimSize`; encoded payload through a bounded reader |
| `read_mgh`, `read_mgh_series` | `MghError::Budget` | file size; `nframes`; `f32` output of all frames |
| `read_mif`, `read_mif_series` | `MifError::Budget` | frame count; payload; `f32` output of all frames |

Decoded storage is the `f32` output, which is never smaller than the stored
voxels of these formats. `ritk-io` passes `ImageReadBudget::DEFAULT` from its
budget-free facade functions.

`MghError`, `MifError`, `MetaImageError`, and `Jpeg2000DecodeError` replace
`anyhow` on the read paths with `thiserror` enums. Failures from parsers that
still return `anyhow` (the `.mif` text header, the shared byte decoder, image
construction, the JPEG 2000 codestream stages) are carried as `anyhow::Error`
sources until those parsers have their own variants.

## Alternatives

- Keep the budget in `ritk-image-io` and add `ritk-image-io` edges to the
  format crates. Rejected: the JPEG 2000 decoder would need a cycle, or a
  second, codec-local limit type beside the first.
- A new `ritk-read-budget` crate. Rejected: the type is three small items and
  every consumer already depends on `ritk-codecs`; a crate adds a publish unit
  without removing an edge.
- Add `ritk-core` or `ritk-image` as a codec dependency. Rejected: both pull the
  tensor stack into a codec crate.

## Consequences

`ritk-image-io` 0.1.0 loses three public items and `ritk-mgh`, `ritk-mif`,
`ritk-metaimage`, and `ritk-codecs` change public signatures; all in-repository
callers migrate in the same change. Migration: import the budget from
`ritk_codecs`, pass `ImageReadBudget::DEFAULT` for the previous behavior, and
match the new error enums instead of downcasting `anyhow::Error`.

`MAX_MGZ_UNCOMPRESSED_BYTES` is deleted: an MGZ stream is consumed only up to
the voxel count its header declares, and that count passes the budget first.

The JPEG and JPEG-LS decoders keep their fixed pixel cap in `dimensions.rs`
until they take a budget.

## Overturning evidence

A reader that must size work from a field the budget cannot express, or a
measured need for one limit type spanning crates below `ritk-codecs`.
