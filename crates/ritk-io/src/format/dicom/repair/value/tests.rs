//! Value rewrites against PS3.5 Table 6.2-1 and section 9.1.

use super::{date, date_is_valid, time, time_is_valid, uid};

#[test]
fn only_the_acr_nema_date_spelling_is_rewritten() {
    let cases = [
        ("1994.07.21", Some("19940721")),
        ("19940721", None),
        ("1994.13.21", None),
        ("1994.07.32", None),
        ("1994.07.31", Some("19940731")),
        ("1994.12.01", Some("19941201")),
        ("1994.00.21", None),
        ("1994.07.00", None),
        ("1994.02.29", None),
        ("1996.02.29", Some("19960229")),
        ("1900.02.29", None),
        ("2000.02.29", Some("20000229")),
        ("1994.04.31", None),
        ("199.407.21", None),
        ("1994.07.2\u{e9}", None),
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
        ("23:59:61", None),
        ("1:020:30", None),
        ("10:20:30.", None),
        ("10:20:30.123456", Some("102030.123456")),
        ("1\u{e9}:20", None),
        ("", None),
    ];
    for (input, expected) in cases {
        assert_eq!(time(input).as_deref(), expected, "time {input:?}");
    }
}

#[test]
fn a_uid_loses_surrounding_spaces_and_leading_zeros_and_nothing_else() {
    let too_long = format!("1.{}", "02".repeat(40));
    let sixty_five = format!("1.{}", "2".repeat(63));
    let sixty_four = format!("1.{}", "2".repeat(62));
    let shrinks_to_sixty_four = format!("01.0{}", "2".repeat(62));
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
        (sixty_five.as_str(), None),
        (shrinks_to_sixty_four.as_str(), Some(sixty_four.as_str())),
        ("1.2.3\0", Some("1.2.3")),
        ("000", Some("0")),
        ("", None),
    ];
    for (input, expected) in cases {
        assert_eq!(uid(input).as_deref(), expected, "uid {input:?}");
    }
}

#[test]
fn the_conformant_date_and_time_grammars_are_recognised_exactly() {
    for (input, valid) in [
        ("19940721", true),
        ("19940229", false),
        ("19941301", false),
        ("1994072", false),
        ("199407211", false),
        ("1994072a", false),
        ("\u{661}\u{669}\u{669}\u{664}", false),
    ] {
        assert_eq!(date_is_valid(input), valid, "date {input:?}");
    }
    for (input, valid) in [
        ("10", true),
        ("1020", true),
        ("102030", true),
        ("102030.5", true),
        ("235960", true),
        ("24", false),
        ("1060", false),
        ("102061", false),
        ("10203", false),
        ("1020.5", false),
        ("102030.", false),
        ("102030.1234567", false),
        (".5", false),
        ("\u{e9}0", false),
    ] {
        assert_eq!(time_is_valid(input), valid, "time {input:?}");
    }
}
