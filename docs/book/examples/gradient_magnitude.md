# Example: Gradient Magnitude

> **Status**: Planned — implementation forthcoming.
> **Source**: `crates/ritk-filter/examples/gradient_magnitude.rs` *(not yet created)*

## Description

This planned example will compare two common ways of measuring local intensity change in ritk: direct gradient magnitude filters and gradient recursive Gaussian variants that smooth and differentiate in one pipeline. The example should start from a scalar image with visible edges, then show how Sobel-style discrete gradients and recursive-Gaussian derivatives emphasize different structures or noise levels. That makes it a natural prelude to edge detection, vesselness, and registration metrics such as NGF.

The Atlas integration story is the same one used throughout ritk: the volume is a Coeus-backed `ritk-image::Image`, and the filters operate through the standard image boundary rather than through a format-specific or backend-specific API. Once implemented, the example would also be a good place to connect conceptual behavior to the existing benchmark page, since recursive Gaussian gradients already have a dedicated performance example in the repository.

## Planned workflow

- Load a scalar volume or synthetic phantom with clear edges.
- Compute a basic gradient magnitude image.
- Compute recursive-Gaussian gradient components or magnitude with a chosen sigma.
- Compare noise sensitivity and edge localization qualitatively.

## Verification goals

- Flat regions produce near-zero response.
- Strong boundaries produce higher magnitude than smooth interiors.
- Output retains the input image geometry.
