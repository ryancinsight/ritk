# Example: Gaussian Smoothing

> **Status**: Planned — implementation forthcoming.
> **Source**: `crates/ritk-filter/examples/gaussian_smoothing.rs` *(not yet created)*

## Description

This planned example will demonstrate Gaussian smoothing as the canonical low-pass filter in ritk. It should cover both isotropic smoothing with one sigma value and anisotropic smoothing where each axis is smoothed differently to respect voxel spacing or acquisition characteristics. The relevant API surface comes from `ritk-filter`'s Gaussian and recursive or discrete Gaussian filters, but the example should keep the user-facing story simple: start from a noisy scalar volume, blur it in a controlled way, and preserve spatial metadata so later stages can still reason about the output physically.

Atlas integration shows up in two places. First, the same `ritk-image::Image` boundary is used before and after smoothing, so no extra format logic is needed. Second, Gaussian smoothing is one of the filters whose host-core implementation is reused across wrapper paths, making it a good conceptual bridge to later benchmarking and backend-dispatch chapters.

## Planned workflow

- Load or synthesize a noisy 3-D image.
- Apply isotropic Gaussian smoothing with one sigma.
- Repeat with anisotropic sigma values aligned to the image axes.
- Compare output ranges and local edge softness.

## Verification goals

- Smoothing reduces local variance while preserving shape.
- Anisotropic settings blur more strongly along selected axes.
- Output geometry matches the input exactly.
