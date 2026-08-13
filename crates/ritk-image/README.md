# ritk-image

Native medical image storage and metadata for [RITK](https://github.com/ryancinsight/ritk).

Defines `Image<T, B, D>` — typed Coeus tensor storage carrying origin, spacing,
and direction metadata, with index-to-physical and physical-to-index transforms.

## Types

| Type | Description |
|---|---|
| `Image<T, B, D>` | Scalar volume over a Coeus backend with physical metadata |
| `RgbVolume` / `ColorVolume` | Multi-channel color volumes |

Depends on `ritk-spatial` for spatial types and the Coeus tensor contracts for
storage. Carries no I/O, filtering, or registration logic.

## Usage

```toml
[dependencies]
ritk-image = "0.3.0"
```
