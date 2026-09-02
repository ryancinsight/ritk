# ADR 0022 — Reusable DTI-volume tractography boundary

- **Status:** Accepted
- **Board item:** `FEAT-686-01`
- **Class:** [minor] [arch]
- **Date:** 2026-09-02

## Context

`ritk-tractography` already owns Euler integration and the DTI direction-field
adapter, while `ritk-diffusion::maps::DtiVolume` owns spatial lookup, fitted
voxel masking, and its anisotropy floor. The `ritk-cli` `tract dti` command
still selected FA seeds and composed those two reusable pieces privately. A
downstream application therefore had to duplicate the seed threshold, evenly
strided cap, and DTI-volume orchestration policy.

## Decision

Add `DtiTractographyConfig`, `dti_volume_seed_points`, and
`dti_volume_tractography` to `ritk-tractography`. The configuration validates
the inclusive seed FA threshold and carries the existing validated integration
configuration. Seed selection treats the DTI mask as authoritative, supports
an unlimited zero cap, and otherwise selects an evenly strided subset of
qualifying voxels in `[depth, row, column]` storage order. Orchestration returns
`NoSeeds` with the threshold and fitted FA peak rather than hiding an empty
pipeline result.

`ritk-cli` calls this surface. It retains only input loading, image-frame point
mapping, output-format selection, and file writing.

## Alternatives rejected

- Keep the policy in the CLI: rejected because downstream consumers would
  continue to fork the same threshold and cap behavior.
- Expose only `dti_volume_direction_field`: rejected because it leaves seed
  selection and orchestration duplicated at every application boundary.
- Add a compatibility wrapper around the private function: rejected by the
  replacement policy; the private implementation is deleted and all in-repo
  callers migrate in the same change.

## Verification and limits

The library tests cover inclusive and invalid thresholds, evenly strided caps,
unfitted voxels at zero threshold, empty selection, and end-to-end DTI-volume
tracking. The runnable example fits a deterministic analytical tensor phantom
and asserts that SVG primitive counts and FA labels match computed values.
These checks establish software and numerical-contract behavior only; they do
not establish anatomical or clinical validity.
