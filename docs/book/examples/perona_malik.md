# Example: Perona-Malik Diffusion

> **Status**: Planned — implementation forthcoming.
> **Source**: `crates/ritk-filter/examples/perona_malik.rs` *(not yet created)*

## Description

This planned example will demonstrate Perona-Malik anisotropic diffusion as an edge-preserving alternative to ordinary smoothing. The core idea is to smooth strongly inside relatively uniform regions while reducing diffusion across steep gradients, so homogeneous tissue becomes less noisy without blurring away meaningful boundaries. The page should compare at least two conductance choices or parameter settings so readers can see the trade-off between denoising strength and edge preservation.

In Atlas terms, this example belongs squarely in the Coeus-backed image pipeline: the image is loaded once, diffusion iterations update intensity content only, and the same geometry metadata flows unchanged into whatever follows next. Because anisotropic diffusion often serves as a preprocessing step for segmentation or registration, the planned example should highlight both visual improvement and the practical downstream motivation for the filter.

## Planned workflow

- Load a noisy scalar volume with visible boundaries.
- Run Perona-Malik diffusion for several iteration counts.
- Compare exponential and inverse-quadratic conductance behavior.
- Inspect denoised regions and preserved edges side by side.

## Verification goals

- Interior noise decreases as iterations increase.
- Major edges remain sharper than with comparable isotropic smoothing.
- Shape and physical metadata are preserved.
