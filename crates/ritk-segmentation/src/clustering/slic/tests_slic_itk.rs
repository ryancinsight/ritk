//! Differential tests for [`slic_itk_impl`] against SimpleITK.
//!
//! Goldens are `sitk.SLIC(superGridSize=[g..], spatialProximityWeight=10,
//! maximumNumberOfIterations=it, ...)`. `CORE*` use the deterministic config
//! (enforceConnectivity=False, initializationPerturbation=False); `FULL*` use
//! the sitk DEFAULT (both on). Equality is exact (label-for-label, ITK
//! scan-order numbering) for evenly- and non-evenly-dividing super-grids in 2-D
//! and 3-D. Image and goldens are emitted from one script so they stay
//! consistent.

use super::super::{
    ConnectivityEnforcement, InitializationPerturbation, ItkSlicConfig, ItkSlicFilter,
};
use super::*;
use burn_ndarray::NdArray;
use coeus_core::SequentialBackend;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_image::native::Image as NativeImage;
use ritk_image::test_support::make_image;

type B = NdArray<f32>;

#[rustfmt::skip]
const IMG2D: [i32; 144] = [
    50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
    0, 0, 100, 100, 100, 100, 100, 100, 100, 0, 0, 0, 0, 0, 100, 100, 100, 100, 100, 100, 100, 0, 0, 0,
    0, 0, 100, 100, 100, 100, 100, 100, 100, 0, 0, 0, 0, 0, 100, 100, 100, 100, 100, 100, 100, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 200, 200, 200, 200, 200, 200, 0, 0,
    0, 0, 0, 0, 200, 200, 200, 200, 200, 200, 0, 0, 0, 0, 0, 0, 200, 200, 200, 200, 200, 200, 0, 0,
    0, 0, 0, 0, 200, 200, 200, 200, 200, 200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

#[rustfmt::skip]
const CORE2D: [i32; 144] = [
    0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2,
    3, 3, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 3, 3, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5,
    3, 3, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 3, 3, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5,
    3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 3, 3, 3, 4, 7, 7, 7, 7, 7, 7, 5, 5,
    6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8,
    6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 6, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8,
];

#[rustfmt::skip]
const FULL2D: [i32; 144] = [
    0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2,
    3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5,
    3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5,
    3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 3, 3, 3, 3, 7, 7, 7, 7, 7, 7, 5, 5,
    6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8,
    6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 6, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8,
];

#[rustfmt::skip]
const IMG3D: [i32; 486] = [
    30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
    30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
    30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
    30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
    30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
    30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
    30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 150, 150, 150, 150, 150, 0, 0, 0, 0, 150,
    150, 150, 150, 150, 0, 0, 0, 0, 150, 150, 150, 150, 150, 0, 0, 0, 0, 150, 150, 150, 150, 150, 0, 0,
    0, 0, 150, 150, 150, 150, 150, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 150,
    150, 150, 150, 150, 0, 0, 0, 0, 150, 150, 150, 150, 150, 0, 0, 0, 0, 150, 150, 150, 150, 150, 0, 0,
    0, 80, 80, 80, 80, 150, 150, 0, 0, 0, 80, 80, 80, 80, 150, 150, 0, 0, 0, 80, 80, 80, 80, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 150, 150, 150, 150, 150, 0, 0, 0, 0, 150, 150, 150, 150, 150, 0, 0,
    0, 0, 150, 150, 150, 150, 150, 0, 0, 0, 80, 80, 80, 80, 150, 150, 0, 0, 0, 80, 80, 80, 80, 150,
    150, 0, 0, 0, 80, 80, 80, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 80, 80, 80, 0,
    0, 0, 0, 0, 80, 80, 80, 80, 0, 0, 0, 0, 0, 80, 80, 80, 80, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
];

#[rustfmt::skip]
const CORE3D: [i32; 486] = [
    0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1,
    2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 3, 3, 3, 4, 4, 4, 5, 5, 5, 3, 3, 3,
    4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 6, 6, 6, 7, 7, 7, 8, 8, 8,
    6, 6, 6, 7, 7, 7, 8, 8, 8, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1,
    2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 3, 3, 3,
    4, 4, 4, 5, 5, 5, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8,
    6, 6, 6, 7, 7, 7, 8, 8, 8, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10,
    11, 11, 11, 9, 9, 9, 10, 10, 10, 11, 11, 11, 9, 9, 13, 13, 13, 13, 13, 11, 11, 9, 9, 13,
    13, 13, 13, 13, 14, 14, 12, 12, 13, 13, 13, 13, 13, 14, 14, 12, 12, 13, 13, 13, 13, 13, 14, 14,
    12, 12, 13, 13, 13, 13, 13, 17, 14, 12, 12, 12, 12, 17, 17, 17, 17, 17, 12, 12, 12, 12, 17, 17,
    17, 17, 17, 9, 9, 9, 10, 10, 10, 11, 11, 11, 9, 9, 9, 10, 10, 10, 11, 11, 11, 9, 9, 13,
    13, 13, 13, 13, 11, 11, 9, 9, 13, 13, 13, 13, 13, 14, 14, 12, 12, 13, 13, 13, 13, 13, 14, 14,
    12, 16, 16, 16, 16, 13, 13, 14, 14, 12, 16, 16, 16, 16, 13, 13, 17, 14, 12, 16, 16, 16, 16, 17,
    17, 17, 17, 12, 12, 12, 12, 17, 17, 17, 17, 17, 9, 9, 9, 10, 10, 10, 11, 11, 11, 9, 9, 9,
    10, 10, 10, 11, 11, 11, 9, 9, 13, 13, 13, 13, 13, 11, 11, 9, 9, 13, 13, 13, 13, 13, 14, 14,
    12, 12, 13, 13, 13, 13, 13, 14, 14, 12, 16, 16, 16, 16, 13, 13, 14, 14, 12, 16, 16, 16, 16, 13,
    13, 14, 14, 12, 16, 16, 16, 16, 17, 17, 17, 17, 12, 12, 12, 12, 17, 17, 17, 17, 17, 9, 9, 9,
    10, 10, 10, 11, 11, 11, 9, 9, 9, 10, 10, 10, 11, 11, 11, 9, 9, 9, 10, 10, 10, 11, 11, 11,
    9, 9, 9, 10, 10, 10, 14, 14, 14, 12, 12, 12, 10, 10, 14, 14, 14, 14, 12, 16, 16, 16, 16, 14,
    14, 14, 14, 12, 16, 16, 16, 16, 17, 17, 14, 14, 12, 16, 16, 16, 16, 17, 17, 17, 17, 12, 12, 12,
    12, 17, 17, 17, 17, 17,
];

#[rustfmt::skip]
const FULL3D: [i32; 486] = [
    0, 0, 1, 1, 1, 2, 2, 2, 2, 0, 0, 1, 1, 1, 2, 2, 2, 2, 0, 0, 1, 1, 1, 5,
    5, 5, 5, 3, 3, 4, 4, 4, 5, 5, 5, 5, 3, 3, 4, 4, 4, 5, 5, 5, 5, 3, 3, 4,
    4, 4, 8, 8, 8, 8, 6, 6, 6, 6, 8, 8, 8, 8, 8, 6, 6, 6, 6, 8, 8, 8, 8, 8,
    6, 6, 6, 6, 8, 8, 8, 8, 8, 0, 0, 1, 1, 1, 2, 2, 2, 2, 0, 0, 1, 1, 1, 2,
    2, 2, 2, 0, 0, 1, 1, 1, 5, 5, 5, 5, 3, 3, 4, 4, 4, 5, 5, 5, 5, 3, 3, 4,
    4, 4, 5, 5, 5, 5, 3, 3, 4, 4, 4, 8, 8, 8, 8, 6, 6, 6, 6, 8, 8, 8, 8, 8,
    6, 6, 6, 6, 8, 8, 8, 8, 8, 6, 6, 6, 6, 8, 8, 8, 8, 8, 9, 9, 9, 10, 10, 11,
    11, 11, 11, 9, 9, 9, 10, 10, 11, 11, 11, 11, 9, 9, 13, 13, 13, 13, 13, 14, 14, 12, 12, 13,
    13, 13, 13, 13, 14, 14, 12, 12, 13, 13, 13, 13, 13, 14, 14, 12, 12, 13, 13, 13, 13, 13, 17, 17,
    15, 15, 13, 13, 13, 13, 13, 17, 17, 15, 15, 15, 16, 16, 16, 17, 17, 17, 15, 15, 15, 16, 16, 16,
    16, 17, 17, 9, 9, 9, 10, 10, 11, 11, 11, 11, 9, 9, 9, 10, 10, 11, 11, 11, 11, 9, 9, 13,
    13, 13, 13, 13, 14, 14, 12, 12, 13, 13, 13, 13, 13, 14, 14, 12, 12, 13, 13, 13, 13, 13, 14, 14,
    12, 7, 7, 7, 7, 13, 13, 17, 17, 15, 7, 7, 7, 7, 13, 13, 17, 17, 15, 7, 7, 7, 7, 16,
    17, 17, 17, 15, 15, 15, 16, 16, 16, 16, 17, 17, 9, 9, 10, 10, 10, 11, 11, 11, 11, 9, 9, 10,
    10, 10, 10, 11, 11, 11, 9, 9, 13, 13, 13, 13, 13, 14, 14, 12, 12, 13, 13, 13, 13, 13, 14, 14,
    12, 12, 13, 13, 13, 13, 13, 14, 14, 12, 7, 7, 7, 7, 13, 13, 17, 17, 15, 7, 7, 7, 7, 13,
    13, 17, 17, 15, 7, 7, 7, 7, 16, 17, 17, 17, 15, 15, 15, 16, 16, 16, 17, 17, 17, 9, 9, 10,
    10, 10, 10, 11, 11, 11, 9, 9, 10, 10, 10, 10, 11, 11, 11, 9, 9, 10, 10, 10, 10, 14, 14, 14,
    12, 12, 12, 10, 10, 14, 14, 14, 14, 12, 12, 12, 12, 10, 14, 14, 14, 14, 12, 7, 7, 7, 7, 17,
    17, 17, 17, 15, 7, 7, 7, 7, 16, 17, 17, 17, 15, 7, 7, 7, 7, 16, 17, 17, 17, 15, 15, 15,
    16, 16, 16, 17, 17, 17,
];

#[rustfmt::skip]
const IMGNE: [i32; 384] = [
    30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
    30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
    30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
    30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
    30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
    30, 30, 30, 30, 30, 30, 30, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 150, 150, 150, 150, 0, 0, 0, 0, 150, 150, 150, 150, 0, 0, 0, 0, 150, 150, 150, 150, 0, 0,
    0, 0, 150, 150, 150, 150, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 150, 150, 150, 150, 0, 0,
    0, 0, 150, 150, 150, 150, 0, 0, 0, 80, 80, 80, 80, 150, 0, 0, 0, 80, 80, 80, 80, 150, 0, 0,
    0, 80, 80, 80, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 80, 80, 80, 80, 0, 0, 0, 0, 80, 80, 80, 80, 0, 0, 0, 0, 80, 80, 80, 80, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 80, 80, 80, 0, 0, 0,
    0, 80, 80, 80, 80, 0, 0, 0, 0, 80, 80, 80, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

#[rustfmt::skip]
const CORENE: [i32; 384] = [
    0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
    0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 3, 3, 3, 3, 3,
    2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1,
    0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
    0, 0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 3, 3,
    2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5,
    4, 4, 6, 6, 6, 6, 5, 5, 4, 4, 6, 6, 6, 6, 5, 5, 4, 4, 6, 6, 6, 6, 7, 7,
    4, 4, 6, 6, 6, 6, 7, 7, 2, 2, 7, 7, 7, 7, 7, 7, 2, 2, 7, 7, 7, 7, 7, 7,
    4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 6, 6, 6, 6, 5, 5,
    4, 4, 6, 6, 6, 6, 5, 5, 4, 6, 6, 6, 6, 6, 7, 7, 4, 6, 6, 6, 6, 6, 7, 7,
    2, 6, 6, 6, 6, 7, 7, 7, 2, 2, 7, 7, 7, 7, 7, 7, 4, 4, 4, 4, 5, 5, 5, 5,
    4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5,
    4, 6, 6, 6, 6, 7, 7, 7, 4, 6, 6, 6, 6, 7, 7, 7, 2, 6, 6, 6, 6, 7, 7, 7,
    2, 2, 7, 7, 7, 7, 7, 7, 4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5,
    4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5, 4, 6, 6, 6, 6, 7, 7, 7,
    4, 6, 6, 6, 6, 7, 7, 7, 2, 6, 6, 6, 6, 7, 7, 7, 2, 2, 7, 7, 7, 7, 7, 7,
];

#[rustfmt::skip]
const FULLNE: [i32; 384] = [
    0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
    0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 3, 3, 3, 3, 3, 0, 0, 3, 3, 3, 3, 3, 3,
    2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1,
    0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
    0, 0, 0, 3, 3, 3, 3, 3, 0, 0, 3, 3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 3, 3,
    2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5,
    4, 4, 6, 6, 6, 6, 5, 5, 4, 4, 6, 6, 6, 6, 5, 5, 4, 4, 6, 6, 6, 6, 7, 7,
    2, 2, 6, 6, 6, 6, 7, 7, 2, 2, 2, 7, 7, 7, 7, 7, 2, 2, 2, 7, 7, 7, 7, 7,
    4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 6, 6, 6, 6, 5, 5,
    4, 4, 6, 6, 6, 6, 5, 5, 4, 6, 6, 6, 6, 6, 7, 7, 2, 6, 6, 6, 6, 6, 7, 7,
    2, 6, 6, 6, 6, 7, 7, 7, 2, 2, 2, 7, 7, 7, 7, 7, 4, 4, 4, 4, 5, 5, 5, 5,
    4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5,
    4, 6, 6, 6, 6, 7, 7, 7, 2, 6, 6, 6, 6, 7, 7, 7, 2, 6, 6, 6, 6, 7, 7, 7,
    2, 2, 2, 7, 7, 7, 7, 7, 4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5,
    4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5, 4, 6, 6, 6, 6, 7, 7, 7,
    4, 6, 6, 6, 6, 7, 7, 7, 2, 6, 6, 6, 6, 7, 7, 7, 2, 2, 2, 7, 7, 7, 7, 7,
];

fn f(a: &[i32]) -> Vec<f32> {
    a.iter().map(|&v| v as f32).collect()
}

#[test]
fn core_2d_matches_sitk() {
    assert_eq!(
        slic_itk_impl(&f(&IMG2D), &[12, 12], &[4, 4], 10.0, 10, false, false),
        f(&CORE2D)
    );
}
#[test]
fn full_2d_matches_sitk() {
    assert_eq!(
        slic_itk_impl(&f(&IMG2D), &[12, 12], &[4, 4], 10.0, 10, true, true),
        f(&FULL2D)
    );
}
#[test]
fn core_3d_matches_sitk() {
    assert_eq!(
        slic_itk_impl(&f(&IMG3D), &[6, 9, 9], &[3, 3, 3], 10.0, 5, false, false),
        f(&CORE3D)
    );
}
#[test]
fn full_3d_matches_sitk() {
    assert_eq!(
        slic_itk_impl(&f(&IMG3D), &[6, 9, 9], &[3, 3, 3], 10.0, 5, true, true),
        f(&FULL3D)
    );
}
#[test]
fn core_3d_non_even_matches_sitk() {
    assert_eq!(
        slic_itk_impl(&f(&IMGNE), &[6, 8, 8], &[3, 3, 3], 10.0, 5, false, false),
        f(&CORENE)
    );
}
#[test]
fn full_3d_non_even_matches_sitk() {
    assert_eq!(
        slic_itk_impl(&f(&IMGNE), &[6, 8, 8], &[3, 3, 3], 10.0, 5, true, true),
        f(&FULLNE)
    );
}

#[test]
fn filter_native_legacy_and_sitk_outputs_are_exact() {
    let values = f(&IMG2D);
    let legacy = make_image::<B, 3>(values.clone(), [1, 12, 12]);
    let native = NativeImage::from_flat_on(
        values,
        [1, 12, 12],
        Point::new([2.0, 3.0, 5.0]),
        Spacing::new([0.5, 1.0, 2.0]),
        Direction::identity(),
        &SequentialBackend,
    )
    .unwrap();
    let config = ItkSlicConfig::new(4)
        .unwrap()
        .with_spatial_proximity_weight(10.0)
        .unwrap()
        .with_maximum_iterations(10)
        .unwrap();
    let filter = ItkSlicFilter::new(config);
    let legacy_output = filter.apply(&legacy).unwrap();
    let native_output = filter.apply_native(&native, &SequentialBackend).unwrap();
    assert_eq!(
        legacy_output.data_slice().as_ref(),
        &FULL2D.map(|value| value as f32)
    );
    assert_eq!(
        native_output.data_slice().unwrap(),
        legacy_output.data_slice().as_ref()
    );
    assert_eq!(*native_output.origin(), Point::new([2.0, 3.0, 5.0]));
    assert_eq!(*native_output.spacing(), Spacing::new([0.5, 1.0, 2.0]));
    assert_eq!(*native_output.direction(), Direction::identity());
}

#[test]
fn itk_slic_validation_errors_are_exact() {
    assert_eq!(
        ItkSlicConfig::new(0).unwrap_err().to_string(),
        "ITK SLIC super-grid size must be at least 1, got 0"
    );
    assert_eq!(
        ItkSlicConfig::new(3)
            .unwrap()
            .with_maximum_iterations(0)
            .unwrap_err()
            .to_string(),
        "ITK SLIC maximum iterations must be at least 1, got 0"
    );
    let invalid = make_image::<B, 3>(vec![0.0, f32::NAN], [1, 1, 2]);
    assert_eq!(
        ItkSlicFilter::new(ItkSlicConfig::new(1).unwrap())
            .apply(&invalid)
            .unwrap_err()
            .to_string(),
        "ITK SLIC sample at flat index 1 must be finite, got NaN"
    );
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
        assert_eq!(
            ItkSlicConfig::new(3)
                .unwrap()
                .with_spatial_proximity_weight(value)
                .unwrap_err()
                .to_string(),
            format!("ITK SLIC spatial proximity weight must be finite, nonnegative, and distance-representable, got {value}")
        );
    }
    let native_invalid = NativeImage::from_flat_on(
        vec![0.0, f32::NEG_INFINITY],
        [1, 1, 2],
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
        &SequentialBackend,
    )
    .unwrap();
    assert_eq!(
        ItkSlicFilter::new(ItkSlicConfig::new(1).unwrap())
            .apply_native(&native_invalid, &SequentialBackend)
            .unwrap_err()
            .to_string(),
        "ITK SLIC sample at flat index 1 must be finite, got -inf"
    );
    let empty = NativeImage::from_flat_on(
        Vec::new(),
        [1, 0, 2],
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
        &SequentialBackend,
    )
    .unwrap();
    assert_eq!(
        ItkSlicFilter::new(ItkSlicConfig::new(1).unwrap())
            .apply_native(&empty, &SequentialBackend)
            .unwrap_err()
            .to_string(),
        "ITK SLIC requires nonzero dimensions, got [1, 0, 2]"
    );
}

#[test]
fn itk_slic_policies_and_accessors_route_exactly() {
    let values = f(&IMG2D);
    let image = make_image::<B, 3>(values.clone(), [1, 12, 12]);
    for (perturbation, connectivity) in [
        (
            InitializationPerturbation::Disabled,
            ConnectivityEnforcement::Disabled,
        ),
        (
            InitializationPerturbation::Disabled,
            ConnectivityEnforcement::Enabled,
        ),
        (
            InitializationPerturbation::Enabled,
            ConnectivityEnforcement::Disabled,
        ),
        (
            InitializationPerturbation::Enabled,
            ConnectivityEnforcement::Enabled,
        ),
    ] {
        let config = ItkSlicConfig::new(4)
            .unwrap()
            .with_spatial_proximity_weight(10.0)
            .unwrap()
            .with_maximum_iterations(10)
            .unwrap()
            .with_initialization_perturbation(perturbation)
            .with_connectivity(connectivity);
        assert_eq!(config.super_grid(), 4);
        assert_eq!(config.spatial_proximity_weight(), 10.0);
        assert_eq!(config.maximum_iterations(), 10);
        assert_eq!(config.initialization_perturbation(), perturbation);
        assert_eq!(config.connectivity(), connectivity);
        let actual = ItkSlicFilter::new(config).apply(&image).unwrap();
        let expected = slic_itk_impl(
            &values,
            &[12, 12],
            &[4, 4],
            10.0,
            10,
            perturbation == InitializationPerturbation::Enabled,
            connectivity == ConnectivityEnforcement::Enabled,
        );
        assert_eq!(actual.data_slice().as_ref(), expected);
    }
}

#[test]
fn itk_slic_extreme_representable_weight_keeps_exact_labels() {
    let image = make_image::<B, 3>((0..16).map(|index| index as f32).collect(), [1, 4, 4]);
    let config = ItkSlicConfig::new(2)
        .unwrap()
        .with_spatial_proximity_weight((f64::MAX / 4.0).sqrt())
        .unwrap()
        .with_initialization_perturbation(InitializationPerturbation::Disabled)
        .with_connectivity(ConnectivityEnforcement::Disabled);
    let labels = ItkSlicFilter::new(config).apply(&image).unwrap();
    assert_eq!(
        labels.data_slice().as_ref(),
        &[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 2.0, 2.0, 3.0, 3.0]
    );
}
