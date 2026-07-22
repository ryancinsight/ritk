# Example: Canny Edge Detection

> **Status**: Planned — implementation forthcoming.
> **Source**: `crates/ritk-filter/examples/canny_edges.rs` *(not yet created)*

## Description

This planned example will walk through the full Canny edge detection pipeline in ritk: optional Gaussian smoothing, gradient computation, non-maximum suppression, and hysteresis thresholding. Rather than treating Canny as a black-box call, the page should explain why each stage exists and how sigma and threshold choices affect the final binary edge map. That makes the example useful both for image-processing users and for readers who want to understand the edge-aware metrics used later in registration workflows.

Within Atlas, Canny is another good showcase for ritk's shared host-core design. The public object remains a `ritk-image::Image`, while the implementation reuses lower-level gradient and suppression primitives that can be validated independently. Once the source exists, the example should make it clear that the algorithm operates on image content only and does not disturb physical metadata.

## Planned workflow

- Load a noisy slice or small 3-D volume with prominent boundaries.
- Smooth with a Gaussian kernel before edge extraction.
- Run Canny with low and high hysteresis thresholds.
- Compare sparse, well-connected edges against unsmoothed gradients.

## Verification goals

- Edge voxels concentrate on strong boundaries rather than flat interiors.
- Hysteresis threshold changes edge connectivity in the expected direction.
- Image geometry is preserved even though intensities become a binary edge map.
