use ritk_annotation::LabelEntry;

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

#[test]
fn a_sample_reads_to_exactly_its_entries() {
    let table = read(SAMPLE.as_bytes()).expect("valid table");

    // Transparency 0 is opaque: alpha 255.
    assert_eq!(
        table.entries(),
        &[
            LabelEntry::new(0, "Unknown", RgbaBytes::new(0, 0, 0, 255)),
            LabelEntry::new(
                2,
                "Left-Cerebral-White-Matter",
                RgbaBytes::new(245, 245, 245, 255)
            ),
            LabelEntry::new(17, "Left-Hippocampus", RgbaBytes::new(220, 216, 20, 255)),
            LabelEntry::new(1035, "ctx-lh-insula", RgbaBytes::new(255, 192, 32, 255)),
        ]
    );
}

#[test]
fn lookup_and_region_names_agree_with_the_entries() {
    let table = read(SAMPLE.as_bytes()).expect("valid table");

    assert_eq!(
        table.get_label(17).map(|entry| entry.name.as_str()),
        Some("Left-Hippocampus")
    );
    assert_eq!(table.get_label(3), None);
    assert_eq!(
        region_names(&table),
        vec![
            (0, "Unknown".to_owned()),
            (2, "Left-Cerebral-White-Matter".to_owned()),
            (17, "Left-Hippocampus".to_owned()),
            (1035, "ctx-lh-insula".to_owned()),
        ]
    );
}

/// FreeSurfer stores the fourth component as transparency and derives
/// `alpha = 255 - t`; the conversion is exact in both directions for every
/// stored value.
#[test]
fn transparency_and_alpha_convert_exactly_both_ways() {
    assert_eq!(
        color_from_stored([255, 192, 32, 64]),
        RgbaBytes::new(255, 192, 32, 191)
    );
    for transparency in 0..=u8::MAX {
        let stored = [1, 2, 3, transparency];
        assert_eq!(stored_components(color_from_stored(stored)), stored);
    }
}

#[test]
fn a_written_table_reads_back_to_itself() {
    let table = read(SAMPLE.as_bytes()).expect("valid table");
    let mut text = Vec::new();
    write(&table, &mut text).expect("writes");
    assert_eq!(read(text.as_slice()).expect("rereads"), table);
}

/// Entries keep file order, so a table round-trips line for line.
#[test]
fn entries_keep_file_order() {
    let table = read("9 b 0 0 0 0\n4 a 0 0 0 0\n".as_bytes()).expect("unique labels");
    let labels: Vec<u32> = table.entries().iter().map(|entry| entry.id.0).collect();
    assert_eq!(labels, vec![9, 4]);
}

fn malformed_line(text: &str) -> usize {
    match read(text.as_bytes()) {
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

/// A repeated label is reported at the line that repeats it.
#[test]
fn a_repeated_label_is_rejected_at_its_line() {
    assert_eq!(malformed_line("3 A 0 0 0 0\n3 B 1 1 1 0\n"), 2);
}

#[test]
fn a_table_without_entries_is_rejected() {
    let error =
        read("# only a comment\n\n".as_bytes()).expect_err("invalid input must be rejected");
    assert!(
        matches!(error, FreeSurferError::InvalidCount { count: 0, .. }),
        "got {error}"
    );
}

#[test]
fn a_name_the_text_format_cannot_hold_is_rejected_on_write() {
    for name in ["", "two words"] {
        let mut table = LabelTable::new();
        table
            .add_label(1, name, RgbaBytes::default())
            .expect("unique label");
        let error = write(&table, Vec::new()).expect_err("invalid input must be rejected");
        assert!(
            matches!(
                error,
                FreeSurferError::Malformed {
                    field: "name of label",
                    index: 1,
                    ..
                }
            ),
            "{name:?}: got {error}"
        );
    }
}
