# Example: Windowing and Rescaling

> **Status**: Planned — implementation forthcoming.
> **Source**: `crates/ritk-filter/examples/windowing_rescale.rs` *(not yet created)*

## Description

This planned example will demonstrate two common pointwise intensity workflows on a `ritk-image::Image<f32, _, 3>`: CT windowing and linear output rescaling. The intended pipeline is to load a scalar volume, apply `IntensityWindowingFilter` with clinically familiar HU presets such as soft-tissue and lung windows, then apply `RescaleIntensityFilter` to map the result into a display- or model-friendly range. Because both filters only remap voxel values, the example should make it obvious that origin, spacing, direction, and shape are preserved exactly.

The page will also show how ritk fits into Atlas internals. The public image boundary stays Coeus-backed, so the same logic can run on the default sequential backend or on Moirai without rewriting the example. The important behavior to verify is endpoint mapping: intensities below the window clamp to the output minimum, intensities above the window clamp to the maximum, and values inside the window are transformed affinely.

## Planned workflow

- Load a CT volume and report the native min/max range.
- Apply soft-tissue and lung windows with different center/width pairs.
- Rescale one result to `0..255` and another to `0.0..1.0`.
- Inspect representative HU anchor values before and after mapping.

## Verification goals

- Known HU values land at expected output intensities.
- Geometry metadata is unchanged after each transform.
- Output min/max matches the requested target range.
