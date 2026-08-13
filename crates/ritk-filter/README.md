# ritk-filter

Medical image filtering algorithms for [RITK](https://github.com/ryancinsight/ritk).

| Category | Algorithms |
|---|---|
| Smoothing | Gaussian, Recursive Gaussian (Deriche), Median, Bilateral |
| Diffusion | Anisotropic (Perona-Malik), curvature, min/max curvature flow |
| Edge detection | Gradient magnitude, Laplacian, Sobel, Prewitt, Canny, Laplacian of Gaussian |
| Vesselness | Frangi and Sato Hessian-based line filters |
| Morphology | Grayscale erosion/dilation, morphological Laplacian, rank and percentile filters |
| Bias correction | N4 bias field correction (B-spline fitting) |
| Resampling | Downsample, resample, multi-resolution pyramid |
| Distance transform | Euclidean (Meijster 2000) and chamfer (chessboard / taxicab) |
| Deconvolution | Regularized deconvolution |
| Denoising | Patch-based denoising |
| Colormap | Color component mapping and lookup-table application |

Filters are generic over `Backend` and `const D: usize`; one implementation
serves every supported scalar type and dimensionality. Numerical behavior is
cross-validated against published references and `scipy.ndimage` where an
equivalent exists.

## Usage

```toml
[dependencies]
ritk-filter = "0.3.0"
```
