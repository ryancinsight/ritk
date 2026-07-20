# Example: Registration Comparison Figure

Visual comparison of CT-to-MR rigid registration: identity (before),
native Coeus registration, and SimpleElastix reference.

## Source

`crates/ritk-registration/examples/registration_compare_figure.rs`

## Description

Renders a mid-axial slice as an RGB overlay (R = CT, G = MR). Aligned
anatomy appears yellow/grey; misalignment appears as red/green fringes.
Compares three approaches: unaligned, Coeus-native classical mutual
information registration, and SimpleElastix reference.

## Usage

```bash
cargo run --example registration_compare_figure -- <ct_file> <mr_file> <output.png>
```

## Verification

- Produces PNG overlay image
- Identity alignment shows red/green fringes
- Coeus registration shows aligned anatomy
- Exercises the `ritk-registration` metric and optimizer boundaries
