# ritk-segmentation

Medical image segmentation algorithms for [RITK](https://github.com/ryancinsight/ritk).

| Module | Algorithms |
|---|---|
| `threshold` | Otsu, Multi-Otsu, Li, Yen, Kapur, Triangle |
| `morphology` | Binary erosion, dilation, opening, closing, skeletonization, fill holes, morphological gradient |
| `labeling` | Connected-component labeling (Hoshen-Kopelman) with statistics |
| `region_growing` | Connected-threshold, confidence-connected, neighborhood-connected |
| `clustering` | K-Means, SLIC superpixels |
| `watershed` | Marker-controlled watershed (Meyer flooding) |
| `level_set` | Chan-Vese, Geodesic Active Contour, Shape Detection, Threshold Level Set, Laplacian Level Set |

All algorithms are generic over the Coeus backend and image dimension.

## Usage

```toml
[dependencies]
ritk-segmentation = "0.3.0"
```
