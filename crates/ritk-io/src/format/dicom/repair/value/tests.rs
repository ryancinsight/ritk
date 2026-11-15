//! Value rewrites against PS3.5 Table 6.2-1 and section 9.1.

use super::{date, time, uid};

#[test]
fn only_the_acr_nema_date_spelling_is_rewritten() {
    let cases = [
        ("1994.07.21", Some("19940721")),
        ("19940721", None),
        ("1994.13.21", None),
        ("1994.07.32", None),
        ("94.07.21", None),
        ("1994.7.21", None),
        ("1994.07.21.1", None),
        ("1994-07-21", None),
        ("", None),
    ];
    for (input, expected) in cases {
        assert_eq!(date(input).as_deref(), expected, "date {input:?}");
    }
}

#[test]
fn only_the_acr_nema_time_spelling_is_rewritten() {
    let cases = [
        ("10:20:30.5", Some("102030.5")),
        ("10:20:30", Some("102030")),
        ("10:20", Some("1020")),
        ("23:59:60", Some("235960")),
        ("102030.5", None),
        ("24:00:00", None),
        ("10:60:00", None),
        ("1:20:30", None),
        ("10:20.5", None),
        ("10:20:30.1234567", None),
        ("10:20:30:40", None),
        ("", None),
    ];
    for (input, expected) in cases {
        assert_eq!(time(input).as_deref(), expected, "time {input:?}");
    }
}

#[test]
fn a_uid_loses_surrounding_spaces_and_leading_zeros_and_nothing_else() {
    let too_long = format!("1.{}", "02".repeat(40));
    let cases = [
        ("1.02.3", Some("1.2.3")),
        ("1.00.3", Some("1.0.3")),
        ("0001", Some("1")),
        (" 1.2.3 ", Some("1.2.3")),
        ("1.2.3", None),
        ("1.0.3", None),
        ("1..3", None),
        ("1.2a.3", None),
        (too_long.as_str(), None),
        ("", None),
    ];
    for (input, expected) in cases {
        assert_eq!(uid(input).as_deref(), expected, "uid {input:?}");
    }
}
