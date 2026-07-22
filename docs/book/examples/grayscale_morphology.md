# Example: Grayscale Opening/Closing

> **Status**: Planned — implementation forthcoming.
> **Source**: `crates/ritk-morphology/examples/grayscale_morphology.rs` *(not yet created)*

## Description

This planned example will extend the morphology story to grayscale data, demonstrating erosion, dilation, opening, closing, and possibly top-hat residues on a scalar image. The purpose is to show that grayscale morphology is not merely binary masking with different output types: it acts as local min/max filtering and can suppress bright speckle, fill dark pits, or estimate background structure without leaving the spatial frame of the original image. A small phantom or microscopy-style slice would make these behaviors easy to see.

The Atlas boundary is again straightforward. The image remains a Coeus-backed `ritk-image::Image`, while morphology-specific neighborhood definitions come from the `ritk-morphology` crate. That separation lets grayscale morphology compose naturally with thresholding, edge detection, and diffusion filters without introducing a special image container or backend rule just for morphology.

## Planned workflow

- Load a grayscale image with bright spots and dark gaps.
- Apply grayscale erosion and dilation with one structuring element.
- Form opening and closing from those primitives.
- Inspect white- or black-top-hat style residual behavior.

## Verification goals

- Erosion never raises local intensities and dilation never lowers them.
- Opening suppresses small bright artifacts; closing suppresses small dark artifacts.
- Output geometry is identical to the input geometry.
