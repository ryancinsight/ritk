use std::borrow::Cow;
use std::fmt::Display;

use anyhow::{anyhow, bail, Result};
use smallvec::SmallVec;

use super::numeric::BinaryNumber;
use super::{DicomNumber, PrimitiveValue};

impl PrimitiveValue {
    pub fn to_str(&self) -> Cow<'_, str> {
        fn joined<T: Display>(values: &[T]) -> String {
            values
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("\\")
        }
        let text = match self {
            Self::Empty => return Cow::Borrowed(""),
            Self::Str(s) => return Cow::Borrowed(s.trim_end_matches([' ', '\0'])),
            Self::Strs(s) if s.len() == 1 => {
                return Cow::Borrowed(s[0].trim_end_matches([' ', '\0']))
            }
            Self::Strs(s) => {
                return Cow::Owned(
                    s.iter()
                        .map(|v| v.trim_end_matches([' ', '\0']))
                        .collect::<Vec<_>>()
                        .join("\\"),
                )
            }
            Self::Bytes(v) => joined(v),
            Self::I16(v) => joined(v),
            Self::U16(v) => joined(v),
            Self::I32(v) => joined(v),
            Self::U32(v) => joined(v),
            Self::I64(v) => joined(v),
            Self::U64(v) => joined(v),
            Self::F32(v) => joined(v),
            Self::F64(v) => joined(v),
            Self::Tags(v) => joined(v),
        };
        Cow::Owned(text)
    }

    fn numbers<T: DicomNumber>(&self, maximum: usize) -> Result<SmallVec<[T; 2]>> {
        fn cast<T: DicomNumber, U: Into<BinaryNumber> + Copy>(
            values: &[U],
            maximum: usize,
        ) -> Result<SmallVec<[T; 2]>> {
            values
                .iter()
                .take(maximum)
                .enumerate()
                .map(|(index, &value)| {
                    T::from_binary(value.into())
                        .ok_or_else(|| anyhow!("DICOM numeric value {index} is out of range"))
                })
                .collect()
        }

        let parse = |(index, text): (usize, &str)| {
            text.trim_matches([' ', '\0'])
                .parse::<T>()
                .map_err(|_| anyhow!("invalid DICOM numeric value {index}: {text:?}"))
        };
        match self {
            Self::Empty => Ok(SmallVec::new()),
            Self::Str(s) => s.split('\\').take(maximum).enumerate().map(parse).collect(),
            Self::Strs(s) => s
                .iter()
                .take(maximum)
                .map(String::as_str)
                .enumerate()
                .map(parse)
                .collect(),
            Self::Bytes(v) => cast(v, maximum),
            Self::I16(v) => cast(v, maximum),
            Self::U16(v) => cast(v, maximum),
            Self::I32(v) => cast(v, maximum),
            Self::U32(v) => cast(v, maximum),
            Self::I64(v) => cast(v, maximum),
            Self::U64(v) => cast(v, maximum),
            Self::F32(v) => cast(v, maximum),
            Self::F64(v) => cast(v, maximum),
            Self::Tags(_) => bail!("DICOM tags are not numeric samples"),
        }
    }

    /// # Errors
    /// The first value must exist and fit the requested numeric type.
    pub fn to_num<T: DicomNumber>(&self) -> Result<T> {
        self.numbers(1)?
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("empty DICOM value"))
    }

    /// # Errors
    /// Each component must fit the requested numeric type.
    pub fn to_numbers<T: DicomNumber>(&self) -> Result<Vec<T>> {
        Ok(self.numbers(usize::MAX)?.into_vec())
    }
}
