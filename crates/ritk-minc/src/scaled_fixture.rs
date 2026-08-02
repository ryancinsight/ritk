//! Deterministic foreign-style scaled-integer MINC2 fixture authoring.
//!
//! This module is included only by tests and the executable book example. It
//! uses the generic Consus HDF5 writer so the production RITK writer remains
//! restricted to its documented `f32` contract.

use anyhow::{bail, Context, Result};
use consus_core::{ByteOrder, Datatype, Shape, StringEncoding};
use consus_hdf5::{
    file::writer::{ChildDatasetSpec, ChildGroupSpec, Hdf5FileBuilder},
    property_list::{DatasetCreationProps, FileCreationProps},
};
use core::num::NonZeroUsize;
use std::path::Path;

/// Image-range datasets to include in a scaled integer fixture.
pub(crate) enum ImageRangeFixture<'a> {
    /// Omit both datasets, exercising the MINC default real range `[0, 1]`.
    #[cfg(test)]
    Omitted,
    /// Include matching scalar or per-slice image ranges.
    Complete {
        minima: &'a [f64],
        maxima: &'a [f64],
    },
    /// Include only `image-min` to construct a malformed pair.
    #[cfg(test)]
    MinimumOnly { minima: &'a [f64] },
}

fn encode_f64(values: &[f64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn range_shape(range_count: usize, slice_count: usize) -> Result<Shape> {
    match range_count {
        1 => Ok(Shape::scalar()),
        count if count == slice_count => Ok(Shape::fixed(&[count])),
        count => {
            bail!("fixture image range must be scalar or contain {slice_count} slices, got {count}")
        }
    }
}

/// Write a tiny contiguous `i16` MINC2 file for reader and book verification.
pub(crate) fn write_scaled_integer_fixture(
    path: &Path,
    voxels: &[i16],
    shape: [usize; 3],
    valid_range: [i16; 2],
    image_ranges: ImageRangeFixture<'_>,
) -> Result<()> {
    if shape.contains(&0) {
        bail!("fixture shape must contain only positive dimensions");
    }
    let voxel_count = shape.iter().copied().try_fold(1_usize, |product, extent| {
        product
            .checked_mul(extent)
            .context("fixture shape overflows usize")
    })?;
    if voxels.len() != voxel_count {
        bail!(
            "fixture voxel count mismatch: shape requires {voxel_count}, got {}",
            voxels.len()
        );
    }

    let int16 = Datatype::Integer {
        bits: NonZeroUsize::new(16).context("invariant: 16 is nonzero")?,
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };
    let int32 = Datatype::Integer {
        bits: NonZeroUsize::new(32).context("invariant: 32 is nonzero")?,
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };
    let float64 = Datatype::Float {
        bits: NonZeroUsize::new(64).context("invariant: 64 is nonzero")?,
        byte_order: ByteOrder::LittleEndian,
    };
    let dimorder_bytes = b"zspace,yspace,xspace";
    let dimorder_type = Datatype::FixedString {
        length: dimorder_bytes.len(),
        encoding: StringEncoding::Ascii,
    };

    let scalar_shape = Shape::scalar();
    let triple_shape = Shape::fixed(&[3]);
    let pair_shape = Shape::fixed(&[2]);
    let image_shape = Shape::fixed(&shape);
    let voxel_bytes: Vec<u8> = voxels
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let valid_range_bytes: Vec<u8> = valid_range
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let image_attributes: [(&str, &Datatype, &Shape, &[u8]); 2] = [
        ("dimorder", &dimorder_type, &scalar_shape, dimorder_bytes),
        ("valid_range", &int16, &pair_shape, &valid_range_bytes),
    ];

    let starts = [0.0_f64.to_le_bytes(); 3];
    let steps = [1.0_f64.to_le_bytes(); 3];
    let lengths = [
        i32::try_from(shape[0])
            .context("fixture z length exceeds i32")?
            .to_le_bytes(),
        i32::try_from(shape[1])
            .context("fixture y length exceeds i32")?
            .to_le_bytes(),
        i32::try_from(shape[2])
            .context("fixture x length exceeds i32")?
            .to_le_bytes(),
    ];
    let directions = [
        encode_f64(&[0.0, 0.0, 1.0]),
        encode_f64(&[0.0, 1.0, 0.0]),
        encode_f64(&[1.0, 0.0, 0.0]),
    ];
    let z_attributes = [
        ("start", &float64, &scalar_shape, starts[0].as_slice()),
        ("step", &float64, &scalar_shape, steps[0].as_slice()),
        ("length", &int32, &scalar_shape, lengths[0].as_slice()),
        (
            "direction_cosines",
            &float64,
            &triple_shape,
            directions[0].as_slice(),
        ),
    ];
    let y_attributes = [
        ("start", &float64, &scalar_shape, starts[1].as_slice()),
        ("step", &float64, &scalar_shape, steps[1].as_slice()),
        ("length", &int32, &scalar_shape, lengths[1].as_slice()),
        (
            "direction_cosines",
            &float64,
            &triple_shape,
            directions[1].as_slice(),
        ),
    ];
    let x_attributes = [
        ("start", &float64, &scalar_shape, starts[2].as_slice()),
        ("step", &float64, &scalar_shape, steps[2].as_slice()),
        ("length", &int32, &scalar_shape, lengths[2].as_slice()),
        (
            "direction_cosines",
            &float64,
            &triple_shape,
            directions[2].as_slice(),
        ),
    ];
    let dimension_groups = [
        ChildGroupSpec {
            name: "zspace",
            attributes: &z_attributes,
            datasets: &[],
            sub_groups: &[],
        },
        ChildGroupSpec {
            name: "yspace",
            attributes: &y_attributes,
            datasets: &[],
            sub_groups: &[],
        },
        ChildGroupSpec {
            name: "xspace",
            attributes: &x_attributes,
            datasets: &[],
            sub_groups: &[],
        },
    ];
    let dimensions_group = ChildGroupSpec {
        name: "dimensions",
        attributes: &[],
        datasets: &[],
        sub_groups: &dimension_groups,
    };

    let image_dataset = ChildDatasetSpec {
        name: "image",
        datatype: &int16,
        shape: &image_shape,
        raw_data: &voxel_bytes,
        dcpl: DatasetCreationProps::default(),
        attributes: &image_attributes,
    };
    let mut range_shapes = Vec::new();
    let mut range_bytes = Vec::new();
    let mut image_datasets = vec![image_dataset];
    match image_ranges {
        #[cfg(test)]
        ImageRangeFixture::Omitted => {}
        ImageRangeFixture::Complete { minima, maxima } => {
            if minima.len() != maxima.len() {
                bail!(
                    "fixture image-min/image-max length mismatch: {} versus {}",
                    minima.len(),
                    maxima.len()
                );
            }
            range_shapes.push(range_shape(minima.len(), shape[0])?);
            range_shapes.push(range_shape(maxima.len(), shape[0])?);
            range_bytes.push(encode_f64(minima));
            range_bytes.push(encode_f64(maxima));
            image_datasets.push(ChildDatasetSpec {
                name: "image-min",
                datatype: &float64,
                shape: &range_shapes[0],
                raw_data: &range_bytes[0],
                dcpl: DatasetCreationProps::default(),
                attributes: &[],
            });
            image_datasets.push(ChildDatasetSpec {
                name: "image-max",
                datatype: &float64,
                shape: &range_shapes[1],
                raw_data: &range_bytes[1],
                dcpl: DatasetCreationProps::default(),
                attributes: &[],
            });
        }
        #[cfg(test)]
        ImageRangeFixture::MinimumOnly { minima } => {
            range_shapes.push(range_shape(minima.len(), shape[0])?);
            range_bytes.push(encode_f64(minima));
            image_datasets.push(ChildDatasetSpec {
                name: "image-min",
                datatype: &float64,
                shape: &range_shapes[0],
                raw_data: &range_bytes[0],
                dcpl: DatasetCreationProps::default(),
                attributes: &[],
            });
        }
    }
    let zero_groups = [ChildGroupSpec {
        name: "0",
        attributes: &[],
        datasets: &image_datasets,
        sub_groups: &[],
    }];
    let image_group = ChildGroupSpec {
        name: "image",
        attributes: &[],
        datasets: &[],
        sub_groups: &zero_groups,
    };
    let minc_groups = [dimensions_group, image_group];

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_group_with_children("minc-2.0", &[], &[], &minc_groups)
        .context("build scaled-integer MINC2 hierarchy")?;
    let bytes = builder.finish().context("finish scaled-integer HDF5")?;
    std::fs::write(path, bytes)
        .with_context(|| format!("write scaled-integer MINC2 fixture {path:?}"))
}
