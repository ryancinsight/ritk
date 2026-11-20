#[test]
fn bounded_metadata_text_preserves_utf8_prefix_without_panicking() {
    use super::super::super::types::{
        cs_to_arraystring, da_to_arraystring, tm_to_arraystring, truncate_arraystring,
        uid_to_arraystring,
    };
    // Byte limits deliberately bisect the two- and three-byte code points.
    let text = "123456789012345éXYZ";
    assert_eq!(truncate_arraystring::<16>(text).as_str(), "123456789012345");
    assert_eq!(cs_to_arraystring(text).as_str(), "123456789012345");
    assert_eq!(tm_to_arraystring(text).as_str(), "123456789012345");
    assert_eq!(da_to_arraystring("1234567é").as_str(), "1234567");
    assert_eq!(
        truncate_arraystring::<16>("界界界界界界").as_str(),
        "界界界界界"
    );
    assert_eq!(truncate_arraystring::<0>("界").as_str(), "");
    assert_eq!(truncate_arraystring::<2>("界").as_str(), "");
    assert_eq!(truncate_arraystring::<3>("界").as_str(), "界");
    let prefix = "1".repeat(63);
    assert_eq!(
        uid_to_arraystring(&format!("{prefix}é"))
            .expect("bounded UID representation")
            .as_str(),
        prefix
    );
}
