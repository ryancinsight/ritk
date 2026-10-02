//! Typed `<DataArray>` decoding for the VTK XML formats (VTI, VTP, VTU).
//!
//! A `<DataArray>` declares its element type in the `type` attribute. The
//! readers decode the values in that type and then convert them to the `f32`
//! the [`AttributeArray`] domain model stores, under
//! [`Cast`](ritk_codecs::sample::Cast): a `Float64` or `Int32` array whose
//! values `f32` cannot all hold rounds with a `tracing` warning, as at the other
//! `f32` carriers (ADR 0053). Inline base64 (`format="binary"`) and compressed
//! appended data are not implemented and are refused by name.

use crate::domain::vtk_data_object::AttributeArray;
use crate::io::read_helpers::read_ascii_samples;
use crate::io::xml_helpers::{attr_usize, attr_val};
use anyhow::{anyhow, bail, Result};
use ritk_codecs::sample::{Cast, Conversion, SampleBuffer, SampleType};

/// The `type` names of the VTK XML formats and the sample types they store.
///
/// The ten names are the list in the `type` attribute of `DataArray` in the VTK
/// XML file-format documentation (Kitware/VTK at 1fb31d6,
/// `Documentation/docs/vtk_file_formats/vtkxml_file_format.md`, "DataArray"
/// attributes), which also notes that 64-bit integers exist only on platforms
/// with 64-bit ids.
const TYPE_NAMES: [(&str, SampleType); 10] = [
    ("UInt8", SampleType::U8),
    ("Int8", SampleType::I8),
    ("UInt16", SampleType::U16),
    ("Int16", SampleType::I16),
    ("UInt32", SampleType::U32),
    ("Int32", SampleType::I32),
    ("UInt64", SampleType::U64),
    ("Int64", SampleType::I64),
    ("Float32", SampleType::F32),
    ("Float64", SampleType::F64),
];

/// The stored sample type a `<DataArray type="...">` names.
///
/// # Errors
///
/// Returns an error for a name outside the ten numeric types (`String` and
/// `Bit` arrays hold no samples).
pub(crate) fn sample_type_from_name(name: &str) -> Result<SampleType> {
    TYPE_NAMES
        .iter()
        .find(|(known, _)| *known == name)
        .map(|(_, sample_type)| *sample_type)
        .ok_or_else(|| anyhow!("unsupported VTK XML DataArray type: {name}"))
}

/// The sample type the `<DataArray ...>` opening `tag` declares.
///
/// # Errors
///
/// Returns an error when `type` is absent or names no numeric type.
pub(crate) fn declared_sample_type(tag: &str) -> Result<SampleType> {
    let name =
        attr_val(tag, "type").ok_or_else(|| anyhow!("DataArray has no type attribute: {tag}"))?;
    sample_type_from_name(&name)
}

/// The values of `buffer` as `f32` under [`Cast`].
pub(crate) fn attribute_values(buffer: SampleBuffer) -> Result<Vec<f32>> {
    Ok(Cast.convert::<f32>(buffer)?)
}

/// Decode the ASCII `content` of the `<DataArray ...>` opening `tag` into `f32`.
///
/// Every token parses in the declared type, so a malformed token is an error
/// rather than a dropped value. The value count must be a whole number of
/// tuples and, when the tag declares `NumberOfTuples`, equal to
/// `NumberOfTuples * NumberOfComponents`.
///
/// # Errors
///
/// Returns an error for a `format` other than `ascii`, an unknown or missing
/// `type`, a token that is not a value of the declared type, or a value count
/// that contradicts the tag.
pub(crate) fn decode_ascii_attribute(tag: &str, content: &str) -> Result<Vec<f32>> {
    if let Some(format) = attr_val(tag, "format") {
        if format != "ascii" {
            bail!(
                "DataArray format=\"{format}\" is not supported by the ASCII reader \
                 (inline base64 is not implemented): {tag}"
            );
        }
    }
    let sample_type = declared_sample_type(tag)?;
    let count = content.split_whitespace().count();
    check_value_count(tag, count)?;
    let mut text = content.as_bytes();
    attribute_values(read_ascii_samples(&mut text, sample_type, count)?)
}

/// Check that `count` decoded values agree with the `<DataArray ...>` opening
/// `tag`: a whole number of tuples and, when the tag declares `NumberOfTuples`,
/// exactly `NumberOfTuples * NumberOfComponents` values.
///
/// Both the ASCII and the appended readers decode into this check, so the two
/// formats refuse the same contradictions.
///
/// # Errors
///
/// Returns an error naming the declared and the actual counts when they
/// disagree, and for a malformed `NumberOfTuples` or `NumberOfComponents`.
pub(crate) fn check_value_count(tag: &str, count: usize) -> Result<()> {
    let components = component_count(tag)?;
    if !count.is_multiple_of(components) {
        bail!(
            "DataArray holds {count} values, not a whole number of \
             {components}-component tuples: {tag}"
        );
    }
    if attr_val(tag, "NumberOfTuples").is_some() {
        let tuples = attr_usize(tag, "NumberOfTuples")?;
        if tuples.checked_mul(components) != Some(count) {
            bail!(
                "DataArray declares {tuples} tuples of {components} components but holds {count} values: {tag}"
            );
        }
    }
    Ok(())
}

/// The `NumberOfComponents` of the `<DataArray ...>` opening `tag`, `1` when
/// absent.
///
/// # Errors
///
/// Returns an error when the attribute is zero or not an integer.
pub(crate) fn component_count(tag: &str) -> Result<usize> {
    let components = if attr_val(tag, "NumberOfComponents").is_some() {
        attr_usize(tag, "NumberOfComponents")?
    } else {
        1
    };
    if components == 0 {
        bail!("DataArray declares zero components: {tag}");
    }
    Ok(components)
}

/// Interpret `values` as the attribute `name` of `components` per tuple.
///
/// Three components are `Vectors` (`Normals` when the name contains "normal"),
/// two are `TextureCoords`, every other count `Scalars`.
///
/// # Errors
///
/// Returns an error when the count of values is not a whole number of tuples.
pub(crate) fn attribute_from_values(
    name: &str,
    components: usize,
    values: Vec<f32>,
) -> Result<AttributeArray> {
    if !values.len().is_multiple_of(components) {
        bail!(
            "DataArray '{name}' holds {} values, not a whole number of {components}-component tuples",
            values.len()
        );
    }
    Ok(match components {
        3 => {
            let tuples: Vec<[f32; 3]> = values
                .chunks_exact(3)
                .map(|tuple| [tuple[0], tuple[1], tuple[2]])
                .collect();
            if name.to_lowercase().contains("normal") {
                AttributeArray::Normals { values: tuples }
            } else {
                AttributeArray::Vectors { values: tuples }
            }
        }
        2 => AttributeArray::TextureCoords { values, dim: 2 },
        n => AttributeArray::Scalars {
            values,
            num_components: n,
        },
    })
}
