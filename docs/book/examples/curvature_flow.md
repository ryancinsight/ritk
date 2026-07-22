# Example: Curvature Flow

> **Status**: Planned — implementation forthcoming.
> **Source**: `crates/ritk-filter/examples/curvature_flow.rs` *(not yet created)*

## Description

This planned example will cover curvature-flow smoothing, a PDE-based filter that evolves intensities according to local curvature rather than simple isotropic diffusion. The practical goal is to remove small-scale roughness while preserving the larger-scale shape of boundaries, making the example a useful contrast to Gaussian smoothing and to Perona-Malik diffusion. A clean presentation would show the same input processed by multiple iteration budgets so the reader can see when the flow improves structure and when it starts to oversmooth.

Atlas integration follows the usual ritk pattern: one `ritk-image::Image` crosses the file boundary, curvature flow operates inside the Coeus-backed processing layer, and the result can feed later validation or registration steps without losing spatial metadata. Documenting that continuity is important because PDE filters are iterative and conceptually complex, but they should still look like ordinary image-to-image transforms at the API level.

## Planned workflow

- Load a scalar image with staircase noise or jagged edges.
- Run curvature flow for several iteration counts or time steps.
- Compare results against a simpler Gaussian blur.
- Inspect how boundary smoothness changes over time.

## Verification goals

- Small rough features are suppressed progressively.
- Large boundaries remain better localized than under naive blurring.
- Output image geometry remains unchanged.
