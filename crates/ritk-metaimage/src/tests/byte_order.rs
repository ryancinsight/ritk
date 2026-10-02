//! `BinaryDataByteOrderMSB` parsing: a case-insensitive `True` is big endian,
//! any other value little endian.

use crate::reader::parse_byte_order_msb;
use consus_core::ByteOrder;

#[test]
fn byte_order_msb_true_in_any_case_is_big_endian() {
    for value in ["True", "TRUE", "true", "tRuE"] {
        assert_eq!(
            parse_byte_order_msb(value),
            ByteOrder::BigEndian,
            "{value:?}"
        );
    }
}

#[test]
fn byte_order_msb_any_other_value_is_little_endian() {
    for value in [
        "False", "FALSE", "fAlSe", "", "yes", "1", "on", "true ", " true", "True.", "big",
    ] {
        assert_eq!(
            parse_byte_order_msb(value),
            ByteOrder::LittleEndian,
            "{value:?}"
        );
    }
}
