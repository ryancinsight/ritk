use super::*;

/// An excerpt shaped like `FreeSurferColorLUT.txt`: comments, blank lines, a
/// seventh tissue-type column on one entry, and irregular spacing.
const SAMPLE: &str = "\
#$Id: FreeSurferColorLUT.txt $

#No. Label Name:                            R   G   B   A

0   Unknown                                 0   0   0   0
2   Left-Cerebral-White-Matter              245 245 245 0
17  Left-Hippocampus                        220 216 20  0   1
1035 ctx-lh-insula                          255 192 32  0
";

fn color(red: u8, green: u8, blue: u8, transparency: u8) -> LutColor {
    LutColor {
        red,
        green,
        blue,
        transparency,
    }
}

fn entry(label: u32, name: &str, color: LutColor) -> LutEntry {
    LutEntry::new(label, name.to_owned(), color).expect("valid entry")
}

#[test]
fn a_sample_parses_to_exactly_its_entries() {
    let lut = ColorLut::parse(SAMPLE.as_bytes()).expect("valid table");

    assert_eq!(
        lut.entries(),
        &[
            entry(0, "Unknown", color(0, 0, 0, 0)),
            entry(2, "Left-Cerebral-White-Matter", color(245, 245, 245, 0)),
            entry(17, "Left-Hippocampus", color(220, 216, 20, 0)),
            entry(1035, "ctx-lh-insula", color(255, 192, 32, 0)),
        ]
    );
}

#[test]
fn lookup_and_region_names_agree_with_the_entries() {
    let lut = ColorLut::parse(SAMPLE.as_bytes()).expect("valid table");

    assert_eq!(lut.get(17).map(LutEntry::name), Some("Left-Hippocampus"));
    assert_eq!(lut.get(3), None);
    assert_eq!(
        lut.region_names(),
        vec![
            (0, "Unknown".to_owned()),
            (2, "Left-Cerebral-White-Matter".to_owned()),
            (17, "Left-Hippocampus".to_owned()),
            (1035, "ctx-lh-insula".to_owned()),
        ]
    );
}

/// FreeSurfer stores the fourth component as transparency and derives
/// `alpha = 255 - t`; the annotation value packs red, green, blue low to high.
#[test]
fn color_derivations_follow_freesurfer() {
    let insula = color(255, 192, 32, 64);
    assert_eq!(insula.alpha(), 191);
    assert_eq!(insula.annotation_value(), 255 + 192 * 256 + 32 * 65_536);
}

#[test]
fn a_written_table_parses_back_to_itself() {
    let lut = ColorLut::parse(SAMPLE.as_bytes()).expect("valid table");
    let mut text = Vec::new();
    lut.write(&mut text).expect("writes");
    assert_eq!(ColorLut::parse(text.as_slice()).expect("reparses"), lut);
}

#[test]
fn entries_are_ordered_by_label_whatever_the_input_order() {
    let lut = ColorLut::new([
        entry(9, "b", LutColor::default()),
        entry(4, "a", LutColor::default()),
    ])
    .expect("unique");
    let labels: Vec<u32> = lut.entries().iter().map(LutEntry::label).collect();
    assert_eq!(labels, vec![4, 9]);
}

fn malformed_line(text: &str) -> usize {
    match ColorLut::parse(text.as_bytes()) {
        Err(FreeSurferError::Malformed {
            field: "line",
            index,
            ..
        }) => index,
        other => panic!("expected a malformed line, got {other:?}"),
    }
}

#[test]
fn an_incomplete_entry_is_rejected_at_its_line() {
    assert_eq!(
        malformed_line("# header\n0 Unknown 0 0 0 0\n5 Short 1 2\n"),
        3
    );
    assert_eq!(malformed_line("7\n"), 1);
}

#[test]
fn a_component_outside_a_byte_is_rejected() {
    assert_eq!(malformed_line("1 Region 256 0 0 0\n"), 1);
    assert_eq!(malformed_line("1 Region 0 0 -1 0\n"), 1);
}

#[test]
fn a_negative_label_is_rejected() {
    assert_eq!(malformed_line("-4 Region 0 0 0 0\n"), 1);
}

#[test]
fn a_repeated_label_is_rejected() {
    let error = ColorLut::parse("3 A 0 0 0 0\n3 B 1 1 1 0\n".as_bytes())
        .expect_err("invalid input must be rejected");
    assert!(
        matches!(
            error,
            FreeSurferError::Malformed {
                field: "label",
                index: 3,
                ..
            }
        ),
        "got {error}"
    );
}

#[test]
fn a_table_without_entries_is_rejected() {
    let error = ColorLut::parse("# only a comment\n\n".as_bytes())
        .expect_err("invalid input must be rejected");
    assert!(
        matches!(error, FreeSurferError::InvalidCount { count: 0, .. }),
        "got {error}"
    );
}

#[test]
fn a_name_the_text_format_cannot_hold_is_rejected() {
    for name in ["", "two words"] {
        let error = LutEntry::new(1, name.to_owned(), LutColor::default())
            .expect_err("invalid input must be rejected");
        assert!(
            matches!(error, FreeSurferError::Malformed { index: 1, .. }),
            "{name:?}: got {error}"
        );
    }
}
