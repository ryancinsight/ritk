//! The NRRD `type` names and the sample types they store.
//!
//! One table serves the reader and the writer, so a name the writer emits is
//! always a name the reader accepts. The first name of each row is the one the
//! writer emits.
//!
//! # Sources
//!
//! - NRRD file format specification (teem.sourceforge.net/nrrd/format.html),
//!   section 5, "Basic Field Specifications", field `type`: the ten numeric
//!   type names and the `block` type.
//! - Teem `src/nrrd/enumsNrrd.c` (the Slicer/teem mirror on GitHub):
//!   `_nrrdTypeStr` holds the ten canonical names, `signed char` through
//!   `double`, and `_nrrdTypeStrEqv` holds every spelling the parser accepts,
//!   from `signed char`/`int8`/`int8_t` through `unsigned long long int`/
//!   `uint64`/`uint64_t`, matched case-insensitively. `TYPE_NAMES` lists those
//!   spellings row for row.

use anyhow::{anyhow, bail, Result};
use ritk_codecs::sample::SampleType;

/// The names one stored sample type goes by in a NRRD `type` field.
struct TypeNames {
    sample_type: SampleType,
    /// The name the writer emits.
    canonical: &'static str,
    /// Every other spelling the specification allows.
    aliases: &'static [&'static str],
}

/// Every numeric NRRD type. `block` is the specification's untyped-record
/// type and has no sample type.
const TYPE_NAMES: [TypeNames; 10] = [
    TypeNames {
        sample_type: SampleType::I8,
        canonical: "signed char",
        aliases: &["int8", "int8_t"],
    },
    TypeNames {
        sample_type: SampleType::U8,
        canonical: "unsigned char",
        aliases: &["uchar", "uint8", "uint8_t"],
    },
    TypeNames {
        sample_type: SampleType::I16,
        canonical: "short",
        aliases: &[
            "short int",
            "signed short",
            "signed short int",
            "int16",
            "int16_t",
        ],
    },
    TypeNames {
        sample_type: SampleType::U16,
        canonical: "unsigned short",
        aliases: &["ushort", "unsigned short int", "uint16", "uint16_t"],
    },
    TypeNames {
        sample_type: SampleType::I32,
        canonical: "int",
        aliases: &["signed int", "int32", "int32_t"],
    },
    TypeNames {
        sample_type: SampleType::U32,
        canonical: "unsigned int",
        aliases: &["uint", "uint32", "uint32_t"],
    },
    TypeNames {
        sample_type: SampleType::I64,
        canonical: "long long int",
        aliases: &[
            "longlong",
            "long long",
            "signed long long",
            "signed long long int",
            "int64",
            "int64_t",
        ],
    },
    TypeNames {
        sample_type: SampleType::U64,
        canonical: "unsigned long long int",
        aliases: &["ulonglong", "unsigned long long", "uint64", "uint64_t"],
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

/// The stored sample type a NRRD `type` field names.
///
/// Matching is case-insensitive and ignores surrounding and repeated
/// whitespace.
///
/// # Errors
///
/// Returns an error for `block`, which stores records rather than numbers, and
/// for any name the specification does not define.
pub(crate) fn sample_type_from_name(name: &str) -> Result<SampleType> {
    let normalised = name
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    if normalised == "block" {
        bail!("NRRD type 'block' stores untyped records, not numeric samples");
    }
    TYPE_NAMES
        .iter()
        .find(|names| names.canonical == normalised || names.aliases.contains(&normalised.as_str()))
        .map(|names| names.sample_type)
        .ok_or_else(|| anyhow!("Unsupported NRRD type: '{name}'"))
}

/// The NRRD `type` name that stores `sample_type`.
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

    /// The specification's alias list, one row per type, written out
    /// independently of the table.
    const SPECIFICATION: [(SampleType, &[&str]); 10] = [
        (SampleType::I8, &["signed char", "int8", "int8_t"]),
        (
            SampleType::U8,
            &["uchar", "unsigned char", "uint8", "uint8_t"],
        ),
        (
            SampleType::I16,
            &[
                "short",
                "short int",
                "signed short",
                "signed short int",
                "int16",
                "int16_t",
            ],
        ),
        (
            SampleType::U16,
            &[
                "ushort",
                "unsigned short",
                "unsigned short int",
                "uint16",
                "uint16_t",
            ],
        ),
        (SampleType::I32, &["int", "signed int", "int32", "int32_t"]),
        (
            SampleType::U32,
            &["uint", "unsigned int", "uint32", "uint32_t"],
        ),
        (
            SampleType::I64,
            &[
                "longlong",
                "long long",
                "long long int",
                "signed long long",
                "signed long long int",
                "int64",
                "int64_t",
            ],
        ),
        (
            SampleType::U64,
            &[
                "ulonglong",
                "unsigned long long",
                "unsigned long long int",
                "uint64",
                "uint64_t",
            ],
        ),
        (SampleType::F32, &["float"]),
        (SampleType::F64, &["double"]),
    ];

    #[test]
    fn every_specified_name_reads_as_its_sample_type() {
        for (expected, names) in SPECIFICATION {
            for name in names {
                assert_eq!(
                    sample_type_from_name(name).expect("specified name"),
                    expected,
                    "{name}"
                );
            }
        }
    }

    #[test]
    fn the_table_holds_exactly_the_specified_names() {
        for (sample_type, names) in SPECIFICATION {
            let row = TYPE_NAMES
                .iter()
                .find(|row| row.sample_type == sample_type)
                .expect("a row per sample type");
            let mut table: Vec<&str> = row.aliases.to_vec();
            table.push(row.canonical);
            table.sort_unstable();
            let mut specified = names.to_vec();
            specified.sort_unstable();
            assert_eq!(table, specified, "{sample_type}");
        }
        assert_eq!(TYPE_NAMES.len(), SampleType::ALL.len());
    }

    #[test]
    fn the_name_a_writer_emits_reads_back_as_the_same_type() {
        for sample_type in SampleType::ALL {
            assert_eq!(
                sample_type_from_name(name_for(sample_type)).expect("emitted name"),
                sample_type
            );
        }
    }

    #[test]
    fn names_ignore_case_and_whitespace() {
        assert_eq!(
            sample_type_from_name("  Unsigned   LONG long  ").expect("normalised"),
            SampleType::U64
        );
    }

    #[test]
    fn block_and_unknown_names_are_refused() {
        let block = sample_type_from_name("block").expect_err("block has no sample type");
        assert!(block.to_string().contains("'block'"), "{block}");
        let unknown = sample_type_from_name("long double").expect_err("not in the specification");
        assert!(
            unknown
                .to_string()
                .contains("Unsupported NRRD type: 'long double'"),
            "{unknown}"
        );
    }
}
