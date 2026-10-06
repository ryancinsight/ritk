/// A NIfTI scalar decodable from a fixed-width lane of either byte order.
pub(crate) trait NiftiLane<const N: usize>: Sized {
    /// Decodes the value from little-endian bytes.
    fn from_little_endian(raw: [u8; N]) -> Self;
    /// Decodes the value from big-endian bytes.
    fn from_big_endian(raw: [u8; N]) -> Self;
}

macro_rules! nifti_lane {
    ($($scalar:ty => $width:literal),+ $(,)?) => {
        $(impl NiftiLane<$width> for $scalar {
            #[inline]
            fn from_little_endian(raw: [u8; $width]) -> Self {
                Self::from_le_bytes(raw)
            }

            #[inline]
            fn from_big_endian(raw: [u8; $width]) -> Self {
                Self::from_be_bytes(raw)
            }
        })+
    };
}

nifti_lane!(i8 => 1, u8 => 1, i16 => 2, u16 => 2, i32 => 4, u32 => 4, i64 => 8, u64 => 8, f32 => 4, f64 => 8);
