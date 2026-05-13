//! JPEG-LS marker constants.

/// JPEG Start of Image marker.
pub(crate) const SOI: u16 = 0xFFD8;
/// JPEG-LS Start of Frame marker, ISO 14495-1 SOF55.
pub(crate) const SOF55: u16 = 0xFFF7;
/// Start of Scan marker.
pub(crate) const SOS: u16 = 0xFFDA;
/// Define Number of Lines marker.
pub(crate) const DNL: u16 = 0xFFDC;
/// Define Restart Interval marker.
pub(crate) const DRI: u16 = 0xFFDD;
/// JPEG-LS preset parameter marker.
pub(crate) const LSE: u16 = 0xFFF8;
/// End of Image marker.
pub(crate) const EOI: u16 = 0xFFD9;
