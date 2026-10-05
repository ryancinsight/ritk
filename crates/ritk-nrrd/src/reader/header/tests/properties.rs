use std::io::Cursor;

use proptest::prelude::*;

use super::super::{
    parse_nrrd_header_from_reader, NrrdHeaderError, MAX_HEADER_BYTES, MAX_HEADER_ENTRIES,
};

proptest! {
    #[test]
    fn bounded_arbitrary_header_lines_preserve_parser_invariants(
        lines in proptest::collection::vec(
            (
                0_u8..8,
                proptest::collection::vec(any::<u8>(), 0..=64),
                proptest::collection::vec(32_u8..=126, 0..=64),
            ),
            0..=64,
        )
    ) {
        let mut input = b"NRRD0005\n".to_vec();
        for (kind, arbitrary, ascii) in lines {
            let (prefix, payload) = match kind {
                0 => (&b""[..], arbitrary.as_slice()),
                1 => (b"content: ".as_slice(), arbitrary.as_slice()),
                2 => (b"custom:=".as_slice(), arbitrary.as_slice()),
                3 => (b"# ".as_slice(), arbitrary.as_slice()),
                4 => (b"content: ".as_slice(), ascii.as_slice()),
                5 => (b"custom:=".as_slice(), ascii.as_slice()),
                6 => (b"# ".as_slice(), ascii.as_slice()),
                _ => (b"dimension: ".as_slice(), ascii.as_slice()),
            };
            input.extend_from_slice(prefix);
            input.extend_from_slice(payload);
            input.push(b'\n');
        }
        input.push(b'\n');
        prop_assert!(input.len() < MAX_HEADER_BYTES);

        let mut reader = Cursor::new(input);
        let result = parse_nrrd_header_from_reader(&mut reader);
        let header_is_bounded = !result
            .as_ref()
            .err()
            .is_some_and(|error| matches!(error, NrrdHeaderError::HeaderTooLarge { .. }));
        prop_assert!(header_is_bounded);
        let Ok(header) = result else {
            return Ok(());
        };

        prop_assert_eq!(header.format_version(), 5);
        prop_assert!(header.comments().iter().all(|comment| comment.is_ascii()));
        prop_assert!(header
            .fields()
            .iter()
            .all(|(key, value)| key.is_ascii() && value.is_ascii()));
        prop_assert!(header
            .key_values()
            .iter()
            .all(|(key, value)| key.is_ascii() && value.is_ascii()));
        let records_are_ascii = header
            .key_value_records()
            .iter()
            .all(|record| record.key().is_ascii() && record.value().is_ascii());
        prop_assert!(records_are_ascii);
        let entry_count = header
            .fields()
            .len()
            .checked_add(header.comments().len())
            .and_then(|count| count.checked_add(header.key_value_records().len()));
        prop_assert!(entry_count.is_some_and(|count| count <= MAX_HEADER_ENTRIES));
    }
}
