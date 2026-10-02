use consus_core::ByteOrder;
use eunomia::CastFrom;

use crate::sample::{write_samples, Cast, Conversion, Exact, Sample, SampleBuffer, SampleType};

fn cast<T: Sample>(buffer: SampleBuffer) -> Vec<T> {
    buffer.cast_into_vec().expect("destination allocation")
}

fn conversion_matches_reference<T>()
where
    T: Sample
        + CastFrom<u64>
        + CastFrom<i64>
        + CastFrom<f64>
        + PartialEq
        + PartialOrd
        + std::fmt::Debug,
{
    for value in [0, 1, 2, (1_u64 << 24) + 1, u64::MAX] {
        assert_eq!(
            T::from_unsigned_sample(value),
            <T as CastFrom<u64>>::cast_from(value)
        );
    }
    for value in [i64::MIN, -1, 0, 1, i64::MAX] {
        assert_eq!(
            T::from_signed_sample(value),
            <T as CastFrom<i64>>::cast_from(value)
        );
    }
    for value in [
        f64::NEG_INFINITY,
        -f64::MAX,
        -1.5,
        -0.0,
        0.0,
        1.5,
        f64::MAX,
        f64::INFINITY,
    ] {
        assert_eq!(
            T::from_real_sample(value),
            <T as CastFrom<f64>>::cast_from(value)
        );
    }
    let nan = T::from_real_sample(f64::NAN);
    if T::TYPE.is_float() {
        assert!(
            nan.partial_cmp(&nan).is_none(),
            "NaN conversion for {}",
            T::TYPE
        );
    } else {
        assert_eq!(
            nan,
            T::from_unsigned_sample(0),
            "NaN conversion for {}",
            T::TYPE
        );
    }
}

#[test]
fn sample_conversions_match_the_rust_numeric_cast_rules() {
    conversion_matches_reference::<u8>();
    conversion_matches_reference::<i8>();
    conversion_matches_reference::<u16>();
    conversion_matches_reference::<i16>();
    conversion_matches_reference::<u32>();
    conversion_matches_reference::<i32>();
    conversion_matches_reference::<u64>();
    conversion_matches_reference::<i64>();
    conversion_matches_reference::<f32>();
    conversion_matches_reference::<f64>();
}

/// The conversion NRRD and MetaImage apply: every stored type into `f32`,
/// rounding to nearest where `f32` cannot hold the value.
#[test]
fn cast_into_vec_rounds_every_stored_type_to_nearest() {
    assert_eq!(
        cast::<f32>(SampleBuffer::U8(vec![0, 1, 127, 255])),
        [0.0, 1.0, 127.0, 255.0]
    );
    assert_eq!(
        cast::<f32>(SampleBuffer::I8(vec![i8::MIN, -1, 127])),
        [-128.0, -1.0, 127.0]
    );
    assert_eq!(cast::<f32>(SampleBuffer::U16(vec![u16::MAX])), [65535.0]);
    assert_eq!(
        cast::<f32>(SampleBuffer::I16(vec![-1, i16::MIN])),
        [-1.0, -32768.0]
    );
    assert_eq!(
        cast::<f32>(SampleBuffer::U32(vec![u32::MAX])),
        [4_294_967_296.0]
    );
    assert_eq!(
        cast::<f32>(SampleBuffer::I32(vec![(1 << 24) + 1])),
        [16_777_216.0]
    );
    assert_eq!(
        cast::<f32>(SampleBuffer::U64(vec![u64::MAX])),
        [18_446_744_073_709_551_616.0]
    );
    assert_eq!(
        cast::<f32>(SampleBuffer::I64(vec![-1, i64::MIN])),
        [-1.0, -9_223_372_036_854_775_808.0]
    );
    assert_eq!(
        cast::<f32>(SampleBuffer::F64(vec![1.0 + f64::EPSILON, -2.5])),
        [1.0, -2.5]
    );
    assert_eq!(
        cast::<f64>(SampleBuffer::F32(vec![0.1])),
        [f64::from(0.1_f32)]
    );
}

#[test]
fn into_vec_widens_integers_exactly() {
    assert_eq!(
        SampleBuffer::I32(vec![(1 << 24) + 1, i32::MIN])
            .into_vec::<i64>()
            .expect("i32 values widen exactly"),
        [(1_i64 << 24) + 1, i64::from(i32::MIN)]
    );
    assert_eq!(
        SampleBuffer::U16(vec![u16::MAX])
            .into_vec::<f64>()
            .expect("u16 values widen exactly"),
        [65535.0]
    );
}

/// Narrowing integers truncate, float-to-integer rounds toward zero and
/// saturates with NaN at zero.
#[test]
fn cast_into_vec_applies_the_primitive_cast() {
    assert_eq!(cast::<u8>(SampleBuffer::I8(vec![-1])), [255]);
    assert_eq!(cast::<u8>(SampleBuffer::U16(vec![300])), [44]);
    assert_eq!(cast::<i32>(SampleBuffer::F32(vec![-1.5, 2.9])), [-1, 2]);
    assert_eq!(
        cast::<i32>(SampleBuffer::F64(vec![f64::NAN, 1e10, -1e10])),
        [0, i32::MAX, i32::MIN]
    );
}

/// Asking for the stored type hands back the stored vector itself: a
/// conversion would collect into a new vector sized to its length.
#[test]
fn into_vec_of_the_stored_type_keeps_the_allocation() {
    let mut stored = Vec::with_capacity(16);
    stored.extend_from_slice(&[3_u16, 1, 4]);
    let values = SampleBuffer::U16(stored)
        .into_vec::<u16>()
        .expect("the stored type");
    assert_eq!(values, [3, 1, 4]);
    assert_eq!(values.capacity(), 16);
}

/// Every request whose stored type does not widen to `T` hands the buffer
/// back untouched.
#[test]
fn into_vec_returns_the_buffer_for_every_lossy_request() {
    let lossy = [
        (SampleBuffer::I8(vec![-1]), SampleType::U8),
        (SampleBuffer::U16(vec![300]), SampleType::U8),
        (SampleBuffer::I32(vec![(1 << 24) + 1]), SampleType::F32),
        (SampleBuffer::U64(vec![u64::MAX]), SampleType::F64),
        (SampleBuffer::F64(vec![0.1]), SampleType::F32),
        (SampleBuffer::F32(vec![2.5]), SampleType::I32),
    ];
    for (buffer, requested) in lossy {
        let returned = match requested {
            SampleType::U8 => buffer.clone().into_vec::<u8>().map(drop),
            SampleType::F32 => buffer.clone().into_vec::<f32>().map(drop),
            SampleType::F64 => buffer.clone().into_vec::<f64>().map(drop),
            SampleType::I32 => buffer.clone().into_vec::<i32>().map(drop),
            other => unreachable!("invariant: the table requests only four types, got {other}"),
        };
        let err = returned.expect_err("a lossy request");
        assert_eq!(
            (err.stored(), err.requested()),
            (buffer.sample_type(), requested)
        );
        assert_eq!(err.into_buffer(), buffer);
    }
}

/// Signed zero and NaN payloads survive encoding and decoding in both byte
/// orders when extraction requests the stored floating-point type.
#[test]
fn read_and_write_float_bits_in_both_byte_orders() {
    let single = [(-0.0_f32).to_bits(), 0x7fc0_1234, 0x7f80_0001, 0xff80_0000];
    let single_values = single.map(f32::from_bits);
    for order in [ByteOrder::LittleEndian, ByteOrder::BigEndian] {
        let mut bytes = Vec::new();
        write_samples(&single_values, order, &mut bytes).expect("a vector accepts every byte");
        let read = SampleBuffer::decode(&bytes, SampleType::F32, order).expect("whole samples");
        assert_eq!(
            read.clone()
                .into_vec::<f32>()
                .expect("the stored type")
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>(),
            single
        );
        let wide = read.into_vec::<f64>().expect("f32 widens to f64");
        assert_eq!(wide[0].to_bits(), (-0.0_f64).to_bits());
        assert!(wide[1].is_nan() && wide[2].is_nan());
        assert_eq!(wide[3], f64::NEG_INFINITY);
    }

    let double = [
        (-0.0_f64).to_bits(),
        0x7ff8_0000_0000_1234,
        0x7ff0_0000_0000_0001,
    ];
    let double_values = double.map(f64::from_bits);
    for order in [ByteOrder::LittleEndian, ByteOrder::BigEndian] {
        let mut bytes = Vec::new();
        write_samples(&double_values, order, &mut bytes).expect("a vector accepts every byte");
        let read = SampleBuffer::decode(&bytes, SampleType::F64, order).expect("whole samples");
        assert_eq!(
            read.into_vec::<f64>()
                .expect("the stored type")
                .into_iter()
                .map(f64::to_bits)
                .collect::<Vec<_>>(),
            double
        );
    }
}

#[test]
fn exact_conversion_refuses_what_cast_converts() {
    let stored = SampleBuffer::I32(vec![(1 << 24) + 1, -3]);
    let err = Exact
        .convert::<f32>(stored.clone())
        .expect_err("i32 does not widen to f32");
    assert_eq!(err.into_buffer(), stored);
    let cast_report = Cast.report::<f32>(&stored);
    assert_eq!(cast_report.stored(), SampleType::I32);
    assert_eq!(cast_report.requested(), SampleType::F32);
    assert_eq!(cast_report.sample_count(), 2);
    assert_eq!(
        cast_report.value_change(),
        super::super::ValueChange::Possible
    );
    assert_eq!(
        cast_report.disposition(),
        super::super::ConversionDisposition::Applied
    );
    assert_eq!(
        Cast.convert::<f32>(stored.clone())
            .expect("a cast never refuses"),
        [16_777_216.0, -3.0]
    );
    let exact_report = Exact.report::<f32>(&stored);
    assert_eq!(
        exact_report.value_change(),
        super::super::ValueChange::Possible
    );
    assert_eq!(
        exact_report.disposition(),
        super::super::ConversionDisposition::Refused
    );
    assert_eq!(
        Exact.convert::<f64>(stored).expect("i32 widens to f64"),
        [16_777_217.0, -3.0]
    );
}

#[test]
fn conversion_report_is_type_level_and_covers_empty_buffers() {
    let exactly_representable = SampleBuffer::I32(vec![42]);
    let potentially_rounded = SampleBuffer::I32(vec![(1 << 24) + 1]);

    let exact_widening = Exact.report::<f64>(&exactly_representable);
    assert_eq!(exact_widening.stored(), SampleType::I32);
    assert_eq!(exact_widening.requested(), SampleType::F64);
    assert_eq!(exact_widening.sample_count(), 1);
    assert_eq!(
        exact_widening.value_change(),
        super::super::ValueChange::None
    );
    assert_eq!(
        exact_widening.disposition(),
        super::super::ConversionDisposition::Applied
    );

    let first_cast = Cast.report::<f32>(&exactly_representable);
    let second_cast = Cast.report::<f32>(&potentially_rounded);
    assert_eq!(first_cast, second_cast);
    assert_eq!(
        first_cast.value_change(),
        super::super::ValueChange::Possible
    );

    let empty = SampleBuffer::I32(Vec::new());
    let empty_report = Exact.report::<f32>(&empty);
    assert_eq!(empty_report.sample_count(), 0);
    assert_eq!(
        empty_report.value_change(),
        super::super::ValueChange::Possible
    );
    assert_eq!(
        empty_report.disposition(),
        super::super::ConversionDisposition::Refused
    );
}

/// Counts the warnings emitted while it is the thread's default subscriber.
#[derive(Default)]
struct WarningCount(std::sync::atomic::AtomicUsize);

impl tracing::Subscriber for WarningCount {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        if *event.metadata().level() == tracing::Level::WARN {
            self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

/// Warnings `Cast` emits while converting `samples` to `T`.
fn cast_warnings<T: Sample>(samples: SampleBuffer) -> usize {
    let count = std::sync::Arc::new(WarningCount::default());
    let dispatch = tracing::Dispatch::from(std::sync::Arc::clone(&count));
    tracing::dispatcher::with_default(&dispatch, || {
        let report = Cast.report::<T>(&samples);
        assert_eq!(report.requested(), T::TYPE);
        let _converted = Cast.convert::<T>(samples).expect("a cast never refuses");
    });
    count.0.load(std::sync::atomic::Ordering::Relaxed)
}

/// `Cast` warns exactly when the stored type does not widen to the request.
#[test]
fn cast_warns_only_when_the_stored_type_does_not_widen() {
    assert_eq!(cast_warnings::<f32>(SampleBuffer::I16(vec![-3])), 0);
    assert_eq!(cast_warnings::<f32>(SampleBuffer::F32(vec![0.5])), 0);
    assert_eq!(cast_warnings::<f32>(SampleBuffer::I32(vec![-3])), 1);
    assert_eq!(cast_warnings::<u8>(SampleBuffer::F64(vec![0.5])), 1);
}
