# Example: Deep Learning Training

Train a Coeus-native registration model end-to-end.

## Source

`crates/ritk-registration/examples/dl_train.rs`

## Description

Full training loop for a Coeus-native registration network: builds the
autodiff graph, runs forward/backward passes, and reports training metrics.
Uses `coeus-nn` for the neural network layer and `coeus-autograd` for
gradient computation.

## Usage

```bash
cargo run --example dl_train -- <dataset_dir> <checkpoint_dir>
```

## Verification

- Trains for specified number of epochs
- Reports loss convergence
- Saves model checkpoint in Coeus-native format
