# ritk-interpolation

Medical image interpolation operators for [RITK](https://github.com/ryancinsight/ritk).

| Method | Notes |
|---|---|
| Linear | 1-D through 4-D |
| Nearest neighbor | 1-D through 4-D |
| B-spline | Cubic kernel |
| Tensor trilinear | Optimized separable 3-D path |

Interpolators are generic over the Coeus backend and image dimension. The
`native` module carries the host-buffer fast paths used by the resampling
filters.

## Usage

```toml
[dependencies]
ritk-interpolation = "0.5.0"
```
