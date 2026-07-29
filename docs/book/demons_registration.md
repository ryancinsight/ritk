# Deformable Registration

RITK's classic Thirion Demons implementation estimates a dense displacement
field from the fixed-image gradient and the moving-to-fixed intensity
difference. Each iteration applies a bounded force update, optional fluid
regularization, optional diffusion regularization, and a forward warp. The
result carries the warped moving image, the three displacement components, the
final MSE, and the performed iteration count.

The public API is intentionally slice-oriented:

```rust,ignore
use ritk_registration::demons::{DemonsConfig, ThirionDemonsRegistration};

let registration = ThirionDemonsRegistration::new(DemonsConfig::default());
let result = registration.register(
    &fixed,
    &moving,
    [depth, height, width],
    [spacing_z, spacing_y, spacing_x],
)?;
```

The `dims` order is `[z, y, x]`, matching RITK's contiguous volume contract.
The returned `disp_x` uses the forward-warp convention implemented by the
engine. A registration result is not accepted merely because the call returns
`Ok`: deformable registration examples must compare the output against an
identity baseline and inspect the displacement field.

The produced book figure uses the separate multi-modal CT/MR
mutual-information path because raw MSE between CT and MR intensities is not a
valid cross-modality registration objective. See
[Example: CT/MR Mutual-Information Registration](examples/registration_compare_figure.md)
for the dataset-backed figure. That example uses the RIRE fiducial transform
as its geometric reference, renders an identity baseline beside the registered
overlay, and includes a data-derived MR-change diagnostic; it does not render
the same reference resampling twice.
