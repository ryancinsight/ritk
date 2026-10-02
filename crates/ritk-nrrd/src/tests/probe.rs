//! The ten sample types with the values the typed-sample cases stress them
//! with.

use ritk_codecs::sample::Sample;
use std::fmt::Debug;

/// A sample type with eight values that stress its representation: extremes,
/// signed zero, and magnitudes `f32` cannot hold exactly.
pub(super) trait Probe: Sample + Debug {
    /// The eight volume values.
    const VALUES: [Self; 8];
    /// A specification alias that is not the writer's name.
    const ALIAS: &'static str;
    /// The name the writer emits.
    const CANONICAL: &'static str;
}

macro_rules! probe {
    ($t:ty, $alias:literal, $canonical:literal, $values:expr) => {
        impl Probe for $t {
            const VALUES: [Self; 8] = $values;
            const ALIAS: &'static str = $alias;
            const CANONICAL: &'static str = $canonical;
        }
    };
}

probe!(
    u8,
    "uint8_t",
    "unsigned char",
    [0, 1, 2, 127, 128, 200, 254, 255]
);
probe!(
    i8,
    "int8",
    "signed char",
    [i8::MIN, -100, -1, 0, 1, 2, 100, i8::MAX]
);
probe!(
    u16,
    "ushort",
    "unsigned short",
    [0, 1, 255, 256, 32_768, 40_000, 65_534, u16::MAX]
);
probe!(
    i16,
    "int16_t",
    "short",
    [i16::MIN, -1024, -1, 0, 1, 3071, 30_000, i16::MAX]
);
probe!(
    u32,
    "uint",
    "unsigned int",
    [
        0,
        1,
        65_535,
        65_536,
        16_777_217,
        3_000_000_000,
        u32::MAX - 1,
        u32::MAX
    ]
);
probe!(
    i32,
    "int32",
    "int",
    [
        i32::MIN,
        -16_777_217,
        -1,
        0,
        1,
        16_777_217,
        70_000,
        i32::MAX
    ]
);
probe!(
    u64,
    "ulonglong",
    "unsigned long long int",
    [
        0,
        1,
        4_294_967_295,
        4_294_967_296,
        9_007_199_254_740_993,
        1 << 63,
        u64::MAX - 1,
        u64::MAX
    ]
);
probe!(
    i64,
    "longlong",
    "long long int",
    [
        i64::MIN,
        -9_007_199_254_740_993,
        -1,
        0,
        1,
        4_294_967_296,
        9_007_199_254_740_993,
        i64::MAX
    ]
);
probe!(
    f32,
    "float",
    "float",
    [
        -0.0,
        0.0,
        f32::MIN_POSITIVE,
        1.0 / 7.0,
        -123_456.79,
        f32::MAX,
        f32::MIN,
        f32::EPSILON
    ]
);
probe!(
    f64,
    "double",
    "double",
    [
        0.1,
        -0.0,
        f64::MIN_POSITIVE,
        1.0 / 7.0,
        1e300,
        f64::MAX,
        f64::MIN,
        f64::EPSILON
    ]
);
