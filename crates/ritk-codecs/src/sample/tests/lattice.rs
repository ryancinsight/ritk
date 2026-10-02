use eunomia::CastFrom;

use crate::sample::Sample;

/// Extremes and the first integers past each float significand, cast into `S`.
fn exactness_probes<S>() -> Vec<S>
where
    S: Sample + CastFrom<u64> + CastFrom<i64> + CastFrom<f64>,
{
    let mut probes = vec![
        <S as CastFrom<u64>>::cast_from(u64::MAX),
        <S as CastFrom<i64>>::cast_from(i64::MIN),
        <S as CastFrom<i64>>::cast_from((1 << 53) + 1),
        <S as CastFrom<i64>>::cast_from((1 << 24) + 1),
        <S as CastFrom<i64>>::cast_from(-1),
        <S as CastFrom<f64>>::cast_from(0.1),
        <S as CastFrom<f64>>::cast_from(f64::MAX),
        <S as CastFrom<f64>>::cast_from(f64::MIN_POSITIVE),
        <S as CastFrom<f64>>::cast_from(f64::NAN),
    ];
    for bits in [8, 16, 32] {
        probes.push(<S as CastFrom<u64>>::cast_from(u64::MAX >> (64 - bits)));
        probes.push(<S as CastFrom<i64>>::cast_from(i64::MIN >> (64 - bits)));
    }
    probes
}

/// Whether every probe of `S` reaches `T` with its value: the cast back gives
/// the probe and the sign is kept (NaN to NaN counts as kept).
fn keeps_every_probe<S, T>() -> bool
where
    S: Sample + PartialOrd + CastFrom<T> + CastFrom<u64> + CastFrom<i64> + CastFrom<f64>,
    T: Sample + PartialOrd + CastFrom<S>,
{
    let is_nan = |value: S| value.partial_cmp(&value).is_none();
    let source_zero = S::from_unsigned_sample(0);
    let target_zero = T::from_unsigned_sample(0);
    exactness_probes::<S>().into_iter().all(|stored| {
        let cast = <T as CastFrom<S>>::cast_from(stored);
        let back = <S as CastFrom<T>>::cast_from(cast);
        let kept = back == stored && (stored < source_zero) == (cast < target_zero);
        kept || (is_nan(stored) && is_nan(back))
    })
}

fn widening_matches_the_probes<S, T>()
where
    S: Sample + PartialOrd + CastFrom<T> + CastFrom<u64> + CastFrom<i64> + CastFrom<f64>,
    T: Sample + PartialOrd + CastFrom<S>,
{
    assert_eq!(
        S::TYPE.widens_to(T::TYPE),
        keeps_every_probe::<S, T>(),
        "{} -> {}",
        S::TYPE,
        T::TYPE
    );
}

/// `widens_to` claims exactly the pairs whose casts keep every probe: a
/// true widening can lose nothing, and each lossy pair loses some probe.
#[test]
fn widens_to_is_the_lossless_cast_lattice() {
    macro_rules! each_pair {
        ($($source:ty),+) => {
            $(each_pair!(@row $source; u8, i8, u16, i16, u32, i32, u64, i64, f32, f64);)+
        };
        (@row $source:ty; $($target:ty),+) => {
            $(widening_matches_the_probes::<$source, $target>();)+
        };
    }
    each_pair!(u8, i8, u16, i16, u32, i32, u64, i64, f32, f64);
}
