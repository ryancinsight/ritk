# Example: Deep Learning Registration

Rigid registration using Coeus autodiff and optimization.

## Source

`crates/ritk-registration/examples/dl_registration.rs`

## Description

Demonstrates the Coeus-native registration pipeline: builds a differentiable
registration graph, optimizes rigid transform parameters via gradient descent,
and outputs aligned images. Uses `coeus-autograd` for autodiff and `coeus-optim`
for the optimizer.

## Usage

```bash
cargo run --example dl_registration -- <fixed> <moving> <output>
```

## Verification

- Optimizes rigid transform (translation + rotation)
- Outputs aligned image with matching geometry
- Uses `assert_coeus_matches_coeus` differential test against reference
