# Example: Validation Suite

> **Status**: Planned — implementation forthcoming.
> **Source**: `crates/ritk-registration/examples/validation_suite.rs` *(not yet created)*

## Description

This planned example will gather the main post-registration checks in one place: geometry consistency, transformation sanity, label-overlap metrics, and summary quality values such as correlation or convergence state. Rather than introducing a new algorithm, the point is to show how `ritk-registration::validation` can be used to decide whether a registration result is believable. A useful implementation would combine image-space checks with label-space checks, so the report covers both continuous alignment quality and discrete structure agreement.

The Atlas integration angle is especially important here because validation often crosses every major boundary in the stack. Images may originate from `ritk-io`, metrics may be computed on Coeus-backed images or Leto-converted classical volumes, and label maps may be warped through the same transform used for the intensity registration itself. The suite should therefore act as the final guardrail that ties algorithm output back to reproducible evidence.

## Planned workflow

- Load fixed and moving images plus optional fixed and moving label maps.
- Compute geometry checks before and after applying the transform.
- Report overlap measures such as Dice or Jaccard on labels.
- Summarize convergence, similarity, and physical error metrics together.

## Verification goals

- The report flags geometry mismatches before metric interpretation.
- Overlap scores improve after a successful registration.
- Numerical summaries are stable and reproducible on the same inputs.
