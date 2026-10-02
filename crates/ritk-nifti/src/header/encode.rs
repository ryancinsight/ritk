//! Encoding of NIfTI-1 and NIfTI-2 single-file headers, little-endian.

use super::convert::f64_to_f32;
use super::datatype::{bitpix, datatype_code};
use super::raw::{write_f32x4, write_f64x4, write_field};
use super::types::{
    HeaderVersion, NiftiHeader, NIFTI1_HEADER_LEN, NIFTI1_MAGIC_SINGLE_FILE, NIFTI2_HEADER_LEN,
    NIFTI2_MAGIC_SINGLE_FILE,
};

impl NiftiHeader {
    pub(crate) fn encode(&self) -> Vec<u8> {
        match self.version {
            HeaderVersion::One => self.encode_nifti1().to_vec(),
            HeaderVersion::Two => self.encode_nifti2().to_vec(),
        }
    }

    fn encode_nifti1(&self) -> [u8; NIFTI1_HEADER_LEN] {
        let mut out = [0_u8; NIFTI1_HEADER_LEN];
        write_field::<i32>(&mut out, 0, 348);
        for (index, value) in self.dim.iter().copied().enumerate() {
            write_field::<u16>(
                &mut out,
                40 + index * 2,
                u16::try_from(value)
                    .expect("invariant: NIfTI-1 header dims are validated at construction"),
            );
        }
        write_field::<i16>(&mut out, 70, datatype_code(self.sample_type));
        write_field::<i16>(&mut out, 72, bitpix(self.sample_type));
        for (index, value) in self.pixdim.iter().copied().enumerate() {
            write_field::<f32>(&mut out, 76 + index * 4, f64_to_f32(value, "pixdim"));
        }
        write_field::<f32>(
            &mut out,
            108,
            f64_to_f32(self.vox_offset as f64, "vox_offset"),
        );
        write_field::<f32>(&mut out, 112, f64_to_f32(self.rescale.slope(), "scl_slope"));
        write_field::<f32>(
            &mut out,
            116,
            f64_to_f32(self.rescale.intercept(), "scl_inter"),
        );
        out[123] = u8::try_from(self.xyzt_units)
            .expect("invariant: NIfTI-1 xyzt_units is set to a u8-compatible value");
        write_field::<i16>(
            &mut out,
            252,
            i16::try_from(self.qform_code).expect("invariant: NIfTI-1 qform_code fits i16"),
        );
        write_field::<i16>(
            &mut out,
            254,
            i16::try_from(self.sform_code).expect("invariant: NIfTI-1 sform_code fits i16"),
        );
        write_field::<f32>(&mut out, 256, f64_to_f32(self.quatern_b, "quatern_b"));
        write_field::<f32>(&mut out, 260, f64_to_f32(self.quatern_c, "quatern_c"));
        write_field::<f32>(&mut out, 264, f64_to_f32(self.quatern_d, "quatern_d"));
        write_field::<f32>(&mut out, 268, f64_to_f32(self.quatern_x, "quatern_x"));
        write_field::<f32>(&mut out, 272, f64_to_f32(self.quatern_y, "quatern_y"));
        write_field::<f32>(&mut out, 276, f64_to_f32(self.quatern_z, "quatern_z"));
        write_f32x4(&mut out, 280, self.srow_x);
        write_f32x4(&mut out, 296, self.srow_y);
        write_f32x4(&mut out, 312, self.srow_z);
        out[344..348].copy_from_slice(&NIFTI1_MAGIC_SINGLE_FILE);
        out
    }

    fn encode_nifti2(&self) -> [u8; NIFTI2_HEADER_LEN] {
        let mut out = [0_u8; NIFTI2_HEADER_LEN];
        write_field::<i32>(&mut out, 0, 540);
        out[4..12].copy_from_slice(&NIFTI2_MAGIC_SINGLE_FILE);
        write_field::<i16>(&mut out, 12, datatype_code(self.sample_type));
        write_field::<i16>(&mut out, 14, bitpix(self.sample_type));
        for (index, value) in self.dim.iter().copied().enumerate() {
            write_field::<i64>(
                &mut out,
                16 + index * 8,
                i64::try_from(value)
                    .expect("invariant: NIfTI-2 header dims are validated at construction"),
            );
        }
        for (index, value) in self.pixdim.iter().copied().enumerate() {
            write_field::<f64>(&mut out, 104 + index * 8, value);
        }
        write_field::<i64>(
            &mut out,
            168,
            i64::try_from(self.vox_offset).expect("invariant: vox_offset fits i64"),
        );
        write_field::<f64>(&mut out, 176, self.rescale.slope());
        write_field::<f64>(&mut out, 184, self.rescale.intercept());
        write_field::<i32>(&mut out, 344, self.qform_code);
        write_field::<i32>(&mut out, 348, self.sform_code);
        write_field::<f64>(&mut out, 352, self.quatern_b);
        write_field::<f64>(&mut out, 360, self.quatern_c);
        write_field::<f64>(&mut out, 368, self.quatern_d);
        write_field::<f64>(&mut out, 376, self.quatern_x);
        write_field::<f64>(&mut out, 384, self.quatern_y);
        write_field::<f64>(&mut out, 392, self.quatern_z);
        write_f64x4(&mut out, 400, self.srow_x);
        write_f64x4(&mut out, 432, self.srow_y);
        write_f64x4(&mut out, 464, self.srow_z);
        write_field::<i32>(&mut out, 500, self.xyzt_units);
        out
    }
}
