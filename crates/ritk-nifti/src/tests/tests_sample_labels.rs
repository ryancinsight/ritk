//! Label maps read from every stored sample type (ADR 0053): integers that
//! fit `u32` and floats holding whole numbers in its range, in either byte
//! order; anything else is an error, never a rounded or clamped label.

use super::tests_samples::{encode, RawNifti1};
use super::*;
use consus_core::ByteOrder;
use ritk_codecs::sample::Sample;

fn read_labels<T: Sample>(values: &[T], order: ByteOrder) -> Result<Vec<u32>> {
    let dir = tempdir()?;
    let path = dir.path().join("labels.nii");
    let bitpix = i16::try_from(T::TYPE.byte_width() * 8)?;
    let datatype = match T::TYPE {
        SampleType::U8 => 2,
        SampleType::I16 => 4,
        SampleType::I32 => 8,
        SampleType::F32 => 16,
        SampleType::F64 => 64,
        SampleType::I8 => 256,
        SampleType::U16 => 512,
        SampleType::U32 => 768,
        SampleType::I64 => 1024,
        SampleType::U64 => 1280,
    };
    let raw = RawNifti1::new(order, datatype, bitpix, [2, 2, 2]);
    std::fs::write(&path, raw.file(&encode(values, order)))?;
    Ok(read_nifti_labels(&path)?.0)
}

const ORDERS: [ByteOrder; 2] = [ByteOrder::LittleEndian, ByteOrder::BigEndian];

#[test]
fn labels_read_from_every_type_in_both_byte_orders() {
    let small = [0_u8, 1, 2, 3, 4, 5, 250, 255];
    let expected: Vec<u32> = small.map(u32::from).to_vec();
    for order in ORDERS {
        assert_eq!(read_labels(&small, order).expect("uint8"), expected);
        assert_eq!(
            read_labels(&small.map(i16::from), order).expect("int16"),
            expected
        );
        assert_eq!(
            read_labels(&small.map(u16::from), order).expect("uint16"),
            expected
        );
        assert_eq!(
            read_labels(&small.map(i32::from), order).expect("int32"),
            expected
        );
        assert_eq!(
            read_labels(&small.map(u32::from), order).expect("uint32"),
            expected
        );
        assert_eq!(
            read_labels(&small.map(i64::from), order).expect("int64"),
            expected
        );
        assert_eq!(
            read_labels(&small.map(u64::from), order).expect("uint64"),
            expected
        );
        assert_eq!(
            read_labels(&small.map(f32::from), order).expect("float32"),
            expected
        );
        assert_eq!(
            read_labels(&small.map(f64::from), order).expect("float64"),
            expected
        );
    }
    // int8 holds labels to 127.
    let signed = [0_i8, 1, 2, 3, 4, 5, 100, 127];
    for order in ORDERS {
        assert_eq!(
            read_labels(&signed, order).expect("int8"),
            signed.map(|label| u32::from(label.unsigned_abs())).to_vec()
        );
    }
    // The largest label: u32::MAX is a whole f64, and a uint32 label reads as
    // itself.
    let top = f64::from(u32::MAX);
    assert_eq!(
        read_labels(
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, top],
            ByteOrder::BigEndian
        )
        .expect("float64 whole numbers")[7],
        u32::MAX
    );
}

/// A label of `minus_one` among zeros is rejected in either byte order.
fn negative_label_is_rejected<T: Sample>(minus_one: T) {
    let mut values = [T::zero(); 8];
    values[3] = minus_one;
    for order in ORDERS {
        let err = read_labels(&values, order).expect_err("negative label");
        assert!(format!("{err:#}").contains("got -1"), "{} {err:#}", T::TYPE);
    }
}

#[test]
fn negative_labels_are_rejected_in_every_signed_type() {
    negative_label_is_rejected::<i8>(-1);
    negative_label_is_rejected::<i16>(-1);
    negative_label_is_rejected::<i32>(-1);
    negative_label_is_rejected::<i64>(-1);
    negative_label_is_rejected::<f32>(-1.0);
    negative_label_is_rejected::<f64>(-1.0);
}

#[test]
fn float_labels_must_be_whole_numbers_in_the_label_range() {
    for bad in [0.5_f32, 2.49, f32::NAN, f32::INFINITY, 4_294_967_296.0] {
        let values = [0.0, 1.0, 2.0, bad, 4.0, 5.0, 6.0, 7.0];
        let err = read_labels(&values, ByteOrder::LittleEndian).expect_err("not a label");
        assert!(
            format!("{err:#}").contains("whole number"),
            "{bad}: {err:#}"
        );
    }
    let err = read_labels(
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.5_f64],
        ByteOrder::BigEndian,
    )
    .expect_err("fractional float64 label");
    assert!(format!("{err:#}").contains("got 1.5"), "{err:#}");
}

#[test]
fn labels_past_the_label_range_are_rejected() {
    let beyond = u64::from(u32::MAX) + 1;
    let err = read_labels(&[0_u64, 1, 2, 3, 4, 5, 6, beyond], ByteOrder::LittleEndian)
        .expect_err("label past u32");
    assert!(format!("{err:#}").contains(&beyond.to_string()), "{err:#}");
}
