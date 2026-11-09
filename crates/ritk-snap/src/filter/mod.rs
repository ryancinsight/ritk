#[cfg_attr(
    not(test),
    expect(dead_code, reason = "ratchet RITK-SNAP-NATIVE-FILTERS-001")
)]
pub(crate) mod catalog;
pub mod filter_kind;
pub mod serde_helper;

pub use filter_kind::FilterKind;
pub use serde_helper::BedSeparationConfigSerde;
