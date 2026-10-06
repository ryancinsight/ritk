/// A NIfTI header scalar decodable from `N` raw bytes of either byte order.
///
/// Sealed by visibility: the set is exactly the scalars `NiftiDatatype` admits,
/// and it grows here when that enum grows.
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

// The impls are byte-for-byte identical past the scalar and its width, and the
// bodies are the std constructors themselves; a declarative macro is the one
// place this file needs to name each scalar, and it is generated once here
// rather than written out four times.
nifti_lane!(i16 => 2, i32 => 4, u32 => 4, f32 => 4);
