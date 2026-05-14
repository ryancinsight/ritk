//! XML parsing helpers shared by the VTI ASCII and binary-appended readers.

/// Return the opening tag string for the first occurrence of `<tag ...>` or
/// `<tag>` in `s`, including the closing `>`.
pub(crate) fn find_tag(s: &str, tag: &str) -> Option<String> {
    let open = format!("<{}", tag);
    let start = s.find(&open)?;
    let end = s[start..].find('>')? + 1;
    Some(s[start..start + end].to_string())
}

/// Return the substring from the first `<tag` to the matching `</tag>` (inclusive).
pub(crate) fn find_section(s: &str, tag: &str) -> Option<String> {
    let open = format!("<{}", tag);
    let close = format!("</{}>", tag);
    let start = s.find(&open)?;
    let end_offset = s[start..].find(&close)? + close.len();
    Some(s[start..start + end_offset].to_string())
}

/// Parse the `name="value"` attribute from an XML tag string.
pub(crate) fn attr_val(tag: &str, name: &str) -> Option<String> {
    let dq = char::from(34u8); // "
    let mut pat = name.to_string();
    pat.push(char::from(61u8)); // =
    pat.push(dq);
    let start = tag.find(&pat)? + pat.len();
    let rest = &tag[start..];
    let end = rest.find(dq)?;
    Some(rest[..end].to_string())
}

/// Parse space/newline-delimited f32 values from a string slice.
pub(crate) fn parse_floats(s: &str) -> Vec<f32> {
    s.split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect()
}

/// Parse space/newline-delimited f64 values from a string slice.
pub(crate) fn parse_f64s(s: &str) -> Vec<f64> {
    s.split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect()
}

/// Parse space/newline-delimited i64 values from a string slice.
pub(crate) fn parse_i64s(s: &str) -> Vec<i64> {
    s.split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect()
}

/// Extract the text content of the first `<DataArray ...>...</DataArray>` in `section`.
///
/// Specified as a required helper for VTI format support; available for
/// future extensions that need to extract individual named DataArray content.
#[allow(dead_code)]
pub(crate) fn extract_da_content(section: &str) -> String {
    let da_start = match section.find("<DataArray") {
        Some(p) => p,
        None => return String::new(),
    };
    let rest = &section[da_start..];
    let gt = match rest.find('>') {
        Some(p) => p + 1,
        None => return String::new(),
    };
    let lt = rest[gt..].find("</").map(|p| gt + p).unwrap_or(rest.len());
    rest[gt..lt].trim().to_string()
}

/// Find a named `<DataArray Name="name" ...>...</DataArray>` within `section`.
///
/// Specified as a required helper for VTI format support; available for
/// future extensions that need to locate a specific named DataArray by name.
#[allow(dead_code)]
pub(crate) fn named_da(section: &str, name: &str) -> Option<String> {
    let dq = char::from(34u8);
    let mut np = String::from("Name=");
    np.push(dq);
    np.push_str(name);
    np.push(dq);
    let attr_pos = section.find(&np)?;
    let da_start = section[..attr_pos].rfind("<DataArray")?;
    let rest = &section[da_start..];
    let close = "</DataArray>";
    let end = rest.find(close)? + close.len();
    Some(rest[..end].to_string())
}
