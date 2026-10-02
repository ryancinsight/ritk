use super::*;
use crate::test_support::{build_mgh_bytes, make_image, TestBackend, IDENTITY_DIR};
use crate::writer::{write_mgh, write_mgh_series};
use crate::{HEADER_SIZE, MRI_FLOAT, MRI_INT, MRI_SHORT, MRI_UCHAR, SINGLE_FRAME, VERSION};
use anyhow::Result;
use flate2::write::GzEncoder;
use flate2::Compression;
use ritk_codecs::sample::Exact;
use ritk_core::alloc_probe::PeakTrackingAllocator;
use ritk_spatial::{Direction, Point, Spacing};
use std::io::Write;
use tempfile::tempdir;

// `#[global_allocator]` is per binary, so the declaration lives here while the
// mechanism lives in `ritk_core::alloc_probe`.
#[global_allocator]
static ALLOCATOR: PeakTrackingAllocator = PeakTrackingAllocator;

mod datatypes;
mod errors;
mod geometry;
mod gzip;
mod native;
mod roundtrip;
mod series;
mod streaming;
