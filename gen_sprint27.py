import pathlib

# resample.rs
resample = r'''//! Resample subcommand.
use anyhow::{bail, Result};
use burn::tensor::backend::Backend as BurnBackend;
use burn::tensor::{Shape, Tensor, TensorData};
use clap;
use ritk_core::filter::resample::ResampleImageFilter;
use ritk_core::interpolation::linear::LinearInterpolator;
use ritk_core::interpolation::{BSplineInterpolator, Lanczos4Interpolator, NearestNeighborInterpolator};
use ritk_core::transform::translation::TranslationTransform;
use std::path::PathBuf;
use tracing::info;
use super::{read_image, write_image_inferred, Backend};
fn main() {}
'''
pathlib.Path('test_sprint.rs').write_text(resample)
print('ok')
