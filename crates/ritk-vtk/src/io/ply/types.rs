//! PLY type system and header structures.

use anyhow::{bail, Context, Result};
use consus_core::{read_integer, ByteOrder, EndianScalar};

// ── Type system ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum PlyType {
    Char,
    Uchar,
    Short,
    Ushort,
    Int,
    Uint,
    Float,
    Double,
}

impl PlyType {
    pub(super) fn from_str(s: &str) -> Result<Self> {
        match s {
            "char" | "int8" => Ok(Self::Char),
            "uchar" | "uint8" => Ok(Self::Uchar),
            "short" | "int16" => Ok(Self::Short),
            "ushort" | "uint16" => Ok(Self::Ushort),
            "int" | "int32" => Ok(Self::Int),
            "uint" | "uint32" => Ok(Self::Uint),
            "float" | "float32" => Ok(Self::Float),
            "double" | "float64" => Ok(Self::Double),
            _ => bail!("unknown PLY scalar type '{}'", s),
        }
    }

    pub(super) fn byte_size(self) -> usize {
        match self {
            Self::Char | Self::Uchar => 1,
            Self::Short | Self::Ushort => 2,
            Self::Int | Self::Uint | Self::Float => 4,
            Self::Double => 8,
        }
    }

    pub(super) fn parse_float_ascii(self, s: &str) -> Result<f32> {
        Ok(match self {
            Self::Float | Self::Double => s
                .parse::<f32>()
                .with_context(|| format!("bad float '{}'", s))?,
            _ => {
                let v: i64 = s.parse().with_context(|| format!("bad integer '{}'", s))?;
                v as f32
            }
        })
    }

    pub(super) fn parse_as_u32(self, s: &str) -> Result<u32> {
        match self {
            Self::Uchar => {
                Ok(s.parse::<u8>()
                    .with_context(|| format!("bad uchar '{}'", s))? as u32)
            }
            Self::Ushort => {
                Ok(s.parse::<u16>()
                    .with_context(|| format!("bad ushort '{}'", s))? as u32)
            }
            Self::Int => Ok(s
                .parse::<i32>()
                .with_context(|| format!("bad int '{}'", s))? as u32),
            Self::Uint => s
                .parse::<u32>()
                .with_context(|| format!("bad uint '{}'", s)),
            _ => bail!("unsupported list count/index type {:?}", self),
        }
    }

    pub(super) fn read_le_float(self, b: &[u8], off: usize) -> f32 {
        let field = &b[off..];
        match self {
            Self::Float => le_field::<f32>(field),
            Self::Double => le_field::<f64>(field) as f32,
            Self::Int => le_field::<i32>(field) as f32,
            Self::Uint => le_field::<u32>(field) as f32,
            Self::Short => f32::from(le_field::<i16>(field)),
            Self::Ushort => f32::from(le_field::<u16>(field)),
            Self::Char => f32::from(le_field::<i8>(field)),
            Self::Uchar => f32::from(le_field::<u8>(field)),
        }
    }

    pub(super) fn read_le_u32(self, b: &[u8], off: usize) -> u32 {
        let field = &b[off..];
        match self {
            Self::Uchar => u32::from(le_field::<u8>(field)),
            Self::Ushort => u32::from(le_field::<u16>(field)),
            Self::Int => le_field::<i32>(field) as u32,
            Self::Uint => le_field::<u32>(field),
            _ => 0,
        }
    }
}

/// Decode the little-endian `T` at the start of a PLY binary body field.
fn le_field<T: EndianScalar>(field: &[u8]) -> T {
    read_integer(field, ByteOrder::LittleEndian)
        .expect("invariant: the body holds every property the header declares")
}

// ── Header model ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum PlyFormat {
    Ascii,
    BinaryLe,
    BinaryBe,
}

pub(super) struct PlyHeader {
    pub(super) format: PlyFormat,
    pub(super) vertex_count: usize,
    pub(super) face_count: usize,
    pub(super) vertex_props: Vec<(String, PlyType)>,
    pub(super) face_count_type: PlyType,
    pub(super) face_index_type: PlyType,
}

impl PlyHeader {
    pub(super) fn find_prop(&self, name: &str) -> Option<usize> {
        self.vertex_props.iter().position(|(n, _)| n == name)
    }

    pub(super) fn prop_type(&self, idx: usize) -> PlyType {
        self.vertex_props[idx].1
    }

    pub(super) fn has_normals(&self) -> bool {
        self.find_prop("nx").is_some()
            && self.find_prop("ny").is_some()
            && self.find_prop("nz").is_some()
    }

    pub(super) fn vertex_byte_size(&self) -> usize {
        self.vertex_props.iter().map(|(_, t)| t.byte_size()).sum()
    }

    pub(super) fn prop_byte_offset(&self, idx: usize) -> usize {
        self.vertex_props[..idx]
            .iter()
            .map(|(_, t)| t.byte_size())
            .sum()
    }
}
