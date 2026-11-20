use ritk_dicom::dataset::PrimitiveValue;
use tracing::info_span;
use tracing_subscriber::fmt::format::FmtSpan;

fn trace_test() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_span_events(FmtSpan::FULL)
            .with_test_writer()
            .finish(),
    )
}

#[test]
fn floating_vrs_check_the_truncated_integer_at_both_bounds() {
    let _trace = trace_test();
    let (lower, upper, wide_upper) =
        info_span!("arrange").in_scope(|| (-129.0_f32, 128.0_f32, 2.0_f64.powi(64)));
    let (accepted, rejected, wide) = info_span!("act").in_scope(|| {
        (
            PrimitiveValue::from([lower.next_up(), upper.next_down()]).to_numbers::<i8>(),
            [lower, upper, f32::NAN, f32::INFINITY]
                .map(|value| PrimitiveValue::from(value).to_num::<i8>()),
            [wide_upper.next_down(), wide_upper]
                .map(|value| PrimitiveValue::from(value).to_num::<u64>()),
        )
    });
    info_span!("assert").in_scope(|| {
        assert_eq!(
            accepted.expect("representable truncated values"),
            [-128, 127]
        );
        for failure in rejected {
            let error = failure.expect_err("out of range or non-finite");
            assert!(error.to_string().contains("out of range"));
        }
        let [below, boundary] = wide;
        assert_eq!(below.expect("predecessor of 2^64"), u64::MAX - 2047);
        assert!(boundary
            .expect_err("2^64 exceeds u64")
            .to_string()
            .contains("out of range"));
    });
}

#[test]
fn unsigned_storage_keeps_its_high_bit_until_target_conversion() {
    let _trace = trace_test();
    let input = info_span!("arrange").in_scope(|| PrimitiveValue::from(u64::MAX));
    let (wide, narrow, rounded) = info_span!("act").in_scope(|| {
        (
            input.to_num::<i128>(),
            input.to_num::<i64>(),
            input.to_num::<f32>(),
        )
    });
    info_span!("assert").in_scope(|| {
        assert_eq!(
            wide.expect("i128 contains unsigned u64"),
            i128::from(u64::MAX)
        );
        assert!(narrow
            .expect_err("signed i64 overflows")
            .to_string()
            .contains("out of range"));
        assert_eq!(
            rounded.expect("IEEE rounding").to_bits(),
            2.0_f32.powi(64).to_bits()
        );
    });
}

#[test]
fn decimal_components_report_the_actual_failed_index() {
    let _trace = trace_test();
    let input = info_span!("arrange").in_scope(|| PrimitiveValue::from(" 7 \\-32768 \\ bad "));
    let (first, all) =
        info_span!("act").in_scope(|| (input.to_num::<i16>(), input.to_numbers::<i16>()));
    info_span!("assert").in_scope(|| {
        assert_eq!(first.expect("first component"), 7);
        assert!(all
            .expect_err("third component fails")
            .to_string()
            .contains("value 2"));
    });
}
