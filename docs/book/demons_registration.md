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
`Ok`: the worked example requires the output MSE to improve over the identity
pair and renders the final displacement for inspection.

## Worked figure

The example uses a deterministic translated phantom, so it does not depend on
private patient data or a network download. When executed, it writes four
panels showing the fixed slice, the translated moving slice, the warped
output, and the recovered `x`-component of the displacement field.

![Thirion Demons registration](figures/thirion_demons.svg)

Run it from the repository root with:

```text
cargo run -p ritk-registration --example book_registration -- \
  docs/book/figures/thirion_demons.svg
```

The complete source and the numerical acceptance check are in
[Example: Thirion Demons](examples/thirion_demons.md).
