# ritk-statistics

Image statistics, information theory, comparison metrics, normalization, and
noise estimation for [RITK](https://github.com/ryancinsight/ritk).

| Module | Contents |
|---|---|
| `image_statistics` | Min, max, mean, variance, percentile (with mask support) |
| `label_statistics` / `label_shape_extended` | Per-label intensity and shape measures |
| `label_overlap` | Dice, Jaccard, and overlap measures |
| `image_comparison` | Hausdorff distance, mean surface distance, PSNR, SSIM |
| `information` | Entropy and mutual-information quantities |
| `histogram` | Ranged, binned histograms |
| `normalization` | Min-max, z-score, histogram matching, Nyul-Udupa, White Stripe |
| `noise_estimation` | MAD-based noise estimation |
| `position_extrema` | `maximum_position` / `minimum_position` |
| `value_indices` | Per-value index map (`scipy.ndimage.value_indices` equivalent) |
| `jacobian` | Deformation-field Jacobian determinant statistics |

Generic over `Backend` and `const D: usize`. Reference implementations are
differentially tested against `scipy` where an equivalent exists.

## Usage

```toml
[dependencies]
ritk-statistics = "0.4.0"
```
