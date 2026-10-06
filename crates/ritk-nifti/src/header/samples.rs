use anyhow::{anyhow, Context, Result};
use ritk_codecs::decode_bytes_to_f32;

use super::convert::LabelValue;
use super::validate::checked_lane;
use super::{NiftiDatatype, NiftiHeader};

impl NiftiHeader {
    pub(crate) fn read_f32_samples(&self, raw: &[u8], count: usize) -> Result<Vec<f32>> {
        let (signed, is_float) = match self.datatype {
            NiftiDatatype::Uint8
            | NiftiDatatype::Uint16
            | NiftiDatatype::Uint32
            | NiftiDatatype::Uint64 => (false, false),
            NiftiDatatype::Int8
            | NiftiDatatype::Int16
            | NiftiDatatype::Int32
            | NiftiDatatype::Int64 => (true, false),
            NiftiDatatype::Float32 | NiftiDatatype::Float64 => (false, true),
        };

        decode_bytes_to_f32(
            raw,
            self.datatype.byte_width(),
            signed,
            is_float,
            self.byte_order(),
            count,
            "NIfTI",
        )
    }

    pub(crate) fn read_label_voxel(&self, raw: &[u8]) -> Result<u32> {
        Ok(match self.datatype {
            NiftiDatatype::Uint8 => u32::from(self.read_lane::<u8, 1>(checked_lane::<1>(raw)?)),
            NiftiDatatype::Int8 => {
                positive_label(self.read_lane::<i8, 1>(checked_lane::<1>(raw)?))?
            }
            NiftiDatatype::Uint16 => u32::from(self.read_lane::<u16, 2>(checked_lane::<2>(raw)?)),
            NiftiDatatype::Int16 => {
                positive_label(self.read_lane::<i16, 2>(checked_lane::<2>(raw)?))?
            }
            NiftiDatatype::Uint32 => self.read_lane::<u32, 4>(checked_lane::<4>(raw)?),
            NiftiDatatype::Int32 => {
                positive_label(self.read_lane::<i32, 4>(checked_lane::<4>(raw)?))?
            }
            NiftiDatatype::Uint64 => {
                let value = self.read_lane::<u64, 8>(checked_lane::<8>(raw)?);
                u32::try_from(value)
                    .with_context(|| format!("NIfTI label voxel exceeds u32::MAX, got {value}"))?
            }
            NiftiDatatype::Int64 => {
                positive_label(self.read_lane::<i64, 8>(checked_lane::<8>(raw)?))?
            }
            NiftiDatatype::Float32 => self
                .read_lane::<f32, 4>(checked_lane::<4>(raw)?)
                .to_label_value(),
            NiftiDatatype::Float64 => self
                .read_lane::<f64, 8>(checked_lane::<8>(raw)?)
                .to_label_value(),
        })
    }
}

fn positive_label<T>(value: T) -> Result<u32>
where
    u32: TryFrom<T>,
    T: std::fmt::Display + Copy,
{
    u32::try_from(value)
        .map_err(|_| anyhow!("NIfTI label voxel must be non-negative and fit u32, got {value}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::{HeaderDims, HeaderSpatial};

    #[test]
    fn bulk_decode_maps_every_nifti_scalar_type_to_the_shared_codec() {
        let cases = [
            (NiftiDatatype::Uint8, 42_u8.to_le_bytes().to_vec(), 42.0),
            (NiftiDatatype::Int8, (-42_i8).to_le_bytes().to_vec(), -42.0),
            (
                NiftiDatatype::Uint16,
                40_000_u16.to_le_bytes().to_vec(),
                40_000.0,
            ),
            (
                NiftiDatatype::Int16,
                (-1_234_i16).to_le_bytes().to_vec(),
                -1_234.0,
            ),
            (
                NiftiDatatype::Uint32,
                1_000_000_u32.to_le_bytes().to_vec(),
                1_000_000.0,
            ),
            (
                NiftiDatatype::Int32,
                (-1_000_000_i32).to_le_bytes().to_vec(),
                -1_000_000.0,
            ),
            (
                NiftiDatatype::Uint64,
                16_777_217_u64.to_le_bytes().to_vec(),
                16_777_216.0,
            ),
            (
                NiftiDatatype::Int64,
                (-16_777_217_i64).to_le_bytes().to_vec(),
                -16_777_216.0,
            ),
            (
                NiftiDatatype::Float32,
                1.25_f32.to_le_bytes().to_vec(),
                1.25,
            ),
            (NiftiDatatype::Float64, 1.5_f64.to_le_bytes().to_vec(), 1.5),
        ];

        for (datatype, bytes, expected) in cases {
            let header = NiftiHeader::new_volume(
                HeaderDims {
                    nx: 1,
                    ny: 1,
                    nz: 1,
                },
                datatype,
                HeaderSpatial {
                    pixdim: [1.0; 8],
                    srow_x: [1.0, 0.0, 0.0, 0.0],
                    srow_y: [0.0, 1.0, 0.0, 0.0],
                    srow_z: [0.0, 0.0, 1.0, 0.0],
                },
            )
            .expect("one-voxel header is valid");

            assert_eq!(
                header
                    .read_f32_samples(&bytes, 1)
                    .expect("scalar sample decodes"),
                [expected],
                "NIfTI datatype {datatype:?} must use matching codec flags"
            );
        }
    }
}
