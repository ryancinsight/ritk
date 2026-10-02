//! The legacy VTK `SCALARS` type names and the sample types they store.
//!
//! One table serves the reader and the writer, so a name the writer emits is
//! always a name the reader accepts. The names and widths are those of
//! `vtkDataReader::ReadArray` in Kitware/VTK `IO/Legacy/vtkDataReader.cxx`,
//! which swaps binary data as big-endian in each branch:
//!
//! | name | sample type | width read |
//! | --- | --- | --- |
//! | `unsigned_char` | `u8` | 1 |
//! | `char`, `signed_char` | `i8` | 1 |
//! | `unsigned_short` | `u16` | 2 |
//! | `short` | `i16` | 2 |
//! | `unsigned_int` | `u32` | 4 |
//! | `int`, `vtkidtype` | `i32` | 4 |
//! | `vtktypeuint64` | `u64` | 8 |
//! | `vtktypeint64` | `i64` | 8 |
//! | `float` | `f32` | 4 |
//! | `double` | `f64` | 8 |
//!
//! `vtkidtype` is the name `vtkDataReader::ReadArray` gives `vtkIdTypeArray`
//! (`vtkDataReader.cxx` lines 1890-1916): it reads the data as 4-byte
//! big-endian `int` and widens each value into the `vtkIdType` array, so the
//! stored width is that of `int` whatever the width of `vtkIdType`.
//! `vtkDataWriter` writes every `vtkIdTypeArray` as `int` data under the name
//! `vtkIdType` (`vtkDataWriter.cxx` lines 1288-1296), which the reader accepts
//! case-insensitively. This writer emits the name of the image's own sample
//! type instead, so it never writes `vtkidtype`.
//!
//! `bit` and `long`/`unsigned_long` have no sample type and are refused by name.
//! `vtkDataWriter` still writes `long` and `unsigned_long` data
//! (`vtkDataWriter.cxx` lines 1204-1230), while `vtkDataReader` reads their
//! binary form as `sizeof(long)` bytes (`vtkDataReader.cxx` lines 1950-1993);
//! the file does not record that width, so it cannot be chosen from the file.

use anyhow::{anyhow, bail, Result};
use ritk_codecs::sample::SampleType;

/// Encoding declared on line 3 of a legacy VTK file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VtkEncoding {
    /// Whitespace-separated decimal text.
    Ascii,
    /// Big-endian packed samples.
    Binary,
}

/// The names one stored sample type goes by in a `SCALARS` line.
struct TypeNames {
    sample_type: SampleType,
    /// The name the writer emits.
    canonical: &'static str,
    /// Every other spelling the reader accepts.
    aliases: &'static [&'static str],
}

/// One row per [`SampleType`] variant.
///
/// `char` and `signed_char` both read as `i8`, as `vtkDataReader` accepts both
/// for one `vtkCharArray`; the writer emits `char`. `vtkCharArray` holds a C
/// `char`, whose signedness is implementation-defined, so `i8` is a recorded
/// choice rather than a fact of the format.
const TYPE_NAMES: [TypeNames; 10] = [
    TypeNames {
        sample_type: SampleType::U8,
        canonical: "unsigned_char",
        aliases: &[],
    },
    TypeNames {
        sample_type: SampleType::I8,
        canonical: "char",
        aliases: &["signed_char"],
    },
    TypeNames {
        sample_type: SampleType::U16,
        canonical: "unsigned_short",
        aliases: &[],
    },
    TypeNames {
        sample_type: SampleType::I16,
        canonical: "short",
        aliases: &[],
    },
    TypeNames {
        sample_type: SampleType::U32,
        canonical: "unsigned_int",
        aliases: &[],
    },
    TypeNames {
        sample_type: SampleType::I32,
        canonical: "int",
        aliases: &["vtkidtype"],
    },
    TypeNames {
        sample_type: SampleType::U64,
        canonical: "vtktypeuint64",
        aliases: &[],
    },
    TypeNames {
        sample_type: SampleType::I64,
        canonical: "vtktypeint64",
        aliases: &[],
    },
    TypeNames {
        sample_type: SampleType::F32,
        canonical: "float",
        aliases: &[],
    },
    TypeNames {
        sample_type: SampleType::F64,
        canonical: "double",
        aliases: &[],
    },
];

/// The stored sample type a `SCALARS` type name declares.
///
/// Matching is case-insensitive.
///
/// # Errors
///
/// Returns an error for:
/// - `bit`, which packs eight samples to a byte and has no `Sample` type;
/// - `long` and `unsigned_long`, whose binary width is `sizeof(long)` of the
///   writing machine (4 or 8 bytes, `VTK_SIZEOF_LONG` in `vtkDataReader`), a
///   fact the file does not record, so no width can be chosen without
///   misreading some files (`vtkDataWriter` still writes both,
///   `vtkDataWriter.cxx` lines 1204-1230);
/// - any other name the legacy format does not define for numeric data.
pub(crate) fn sample_type_from_name(name: &str) -> Result<SampleType> {
    let lowered = name.to_ascii_lowercase();
    match lowered.as_str() {
        "bit" => {
            bail!("VTK scalar type 'bit' packs eight samples per byte and is not a sample type")
        }
        "long" | "unsigned_long" => bail!(
            "VTK scalar type '{lowered}' has the width of the writing machine's C `long` \
             (4 or 8 bytes), which the file does not record; rewrite the data as \
             vtktypeint64/vtktypeuint64 or int/unsigned_int"
        ),
        other => TYPE_NAMES
            .iter()
            .find(|names| names.canonical == other || names.aliases.contains(&other))
            .map(|names| names.sample_type)
            .ok_or_else(|| anyhow!("unsupported VTK scalar type: {name}")),
    }
}

/// The `SCALARS` type name that stores `sample_type`.
///
/// Total: the format has a name for every [`SampleType`], so a writer never
/// refuses a sample type.
pub(crate) fn name_for(sample_type: SampleType) -> &'static str {
    TYPE_NAMES
        .iter()
        .find(|names| names.sample_type == sample_type)
        .map(|names| names.canonical)
        .expect("invariant: TYPE_NAMES holds one row per SampleType variant")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The names and widths of `vtkDataReader::ReadArray`, written out
    /// independently of the table.
    const SPECIFICATION: [(&str, SampleType, usize); 11] = [
        ("unsigned_char", SampleType::U8, 1),
        ("char", SampleType::I8, 1),
        ("signed_char", SampleType::I8, 1),
        ("unsigned_short", SampleType::U16, 2),
        ("short", SampleType::I16, 2),
        ("unsigned_int", SampleType::U32, 4),
        ("int", SampleType::I32, 4),
        ("vtktypeuint64", SampleType::U64, 8),
        ("vtktypeint64", SampleType::I64, 8),
        ("float", SampleType::F32, 4),
        ("double", SampleType::F64, 8),
    ];

    #[test]
    fn every_specified_name_maps_to_its_type_and_width() {
        for (name, sample_type, width) in SPECIFICATION {
            let parsed = sample_type_from_name(name).expect("specified name");
            assert_eq!(parsed, sample_type, "{name}");
            assert_eq!(parsed.byte_width(), width, "{name}");
            assert_eq!(
                sample_type_from_name(&name.to_ascii_uppercase()).expect("case-insensitive"),
                sample_type,
                "{name}"
            );
        }
    }

    #[test]
    fn written_names_are_read_back_as_the_same_type() {
        for sample_type in SampleType::ALL {
            assert_eq!(
                sample_type_from_name(name_for(sample_type)).expect("emitted name"),
                sample_type
            );
        }
    }

    #[test]
    fn names_without_a_sample_type_are_refused_by_name() {
        let bit = sample_type_from_name("bit").expect_err("bit has no sample type");
        assert!(bit.to_string().contains("'bit'"), "{bit}");
        for name in ["long", "unsigned_long"] {
            let refused = sample_type_from_name(name).expect_err("platform-width type");
            assert!(refused.to_string().contains(name), "{refused}");
            assert!(refused.to_string().contains("does not record"), "{refused}");
        }
        let unknown = sample_type_from_name("vtkpointer").expect_err("not a scalar name");
        assert_eq!(
            unknown.to_string(),
            "unsupported VTK scalar type: vtkpointer"
        );
    }

    /// `vtkDataReader` reads `vtkidtype` as 4-byte `int`, in any letter case.
    #[test]
    fn id_type_name_reads_as_four_byte_integers() {
        for name in ["vtkidtype", "vtkIdType", "VTKIDTYPE"] {
            let parsed = sample_type_from_name(name).expect("vtkidtype is a defined name");
            assert_eq!(parsed, SampleType::I32, "{name}");
            assert_eq!(parsed.byte_width(), 4, "{name}");
        }
    }
}
