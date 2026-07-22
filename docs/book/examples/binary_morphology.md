# Example: Binary Erosion/Dilation

> **Status**: Planned — implementation forthcoming.
> **Source**: `crates/ritk-morphology/examples/binary_morphology.rs` *(not yet created)*

## Description

This planned example will pair ritk's binary morphology filters with the structuring-element types exported by `ritk-morphology`. The intended walkthrough is erosion, dilation, opening, and closing on a binary mask, using simple `Cube`, `Cross`, or `Ball` neighborhoods to show how foreground topology changes under each operation. By keeping the input binary, the example can emphasize the algebraic meaning of each transform: erosion removes thin foreground structures, dilation expands them, opening removes small protrusions, and closing fills narrow gaps.

Atlas integration matters because the structuring element and the image have different ownership roles. `ritk-morphology` provides zero-sized shape markers and borrowed offset lists, while the actual image still lives in the standard Coeus-backed `ritk-image::Image` boundary consumed by the filters. That split is intentional and worth documenting because it keeps morphology shape definitions zero-cost while letting the rest of the pipeline reuse ordinary image abstractions.

## Planned workflow

- Create or load a binary mask with small holes and thin bridges.
- Apply erosion and dilation with a chosen neighborhood.
- Compose them into opening and closing.
- Compare foreground voxel counts and qualitative topology changes.

## Verification goals

- Radius-zero structuring elements behave as identity.
- Opening is anti-extensive and closing is extensive.
- Output masks preserve the input geometry contract.
