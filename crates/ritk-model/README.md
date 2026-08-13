# ritk-model

Medical image model and inference integration for [RITK](https://github.com/ryancinsight/ritk).

| Module | Contents |
|---|---|
| `transmorph` | Transformer-based deformable registration model |
| `ssmmorph` | Statistical shape model registration |
| `onnx` | ONNX graph import via `onnx-ir` (initializers, graph validation) |
| `monai` | MONAI-compatible model definitions |
| `affine` | Learned affine parameter heads |

Models execute through the Coeus tensor and autodiff contracts; the crate holds
no bespoke tensor arithmetic.

## Usage

```toml
[dependencies]
ritk-model = "0.2.0"
```
