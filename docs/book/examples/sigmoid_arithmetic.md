# Example: Sigmoid and Arithmetic

> **Status**: Planned — implementation forthcoming.
> **Source**: `crates/ritk-filter/examples/sigmoid_arithmetic.rs` *(not yet created)*

## Description

This planned example will combine `SigmoidImageFilter` with simple arithmetic intensity operators such as add, subtract, multiply, and divide. The goal is to show a practical normalization pattern: use a sigmoid remap to compress dynamic range around a meaningful center value, then apply arithmetic filters to shift, scale, or combine images for downstream processing. Because the operations are pointwise, they are good demonstrations of ritk's substrate-agnostic host-core design and of the fact that the `ritk-image::Image` boundary stays intact across chained transforms.

The Atlas angle is that these are inexpensive building blocks that can execute through the same Coeus-backed image abstraction used elsewhere in the toolkit. An eventual implementation could easily compare sequential and Moirai execution without changing the actual filter composition. The example should therefore read as a miniature pipeline rather than as isolated API calls, showing how normalization and arithmetic cooperate before filtering or registration.

## Planned workflow

- Start from a scalar volume or synthetic ramp image.
- Apply a sigmoid centered on a chosen intensity with tunable slope.
- Shift and scale the result into a normalized range.
- Optionally combine two images with add or subtract for contrast enhancement.

## Verification goals

- The sigmoid output stays bounded in the requested range.
- Arithmetic steps match analytically expected voxel values.
- Chaining operations preserves shape and physical metadata.
