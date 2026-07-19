use ritk_filter::{BedSeparationConfig, ComponentPolicy};
use serde::{Deserialize, Serialize};

// â”€â”€ Remote serde helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// `BedSeparationConfig` is defined in `ritk-core` and does not derive
// `Serialize`/`Deserialize`. The `#[serde(remote = "...")]` pattern generates
// `BedSeparationConfigSerde::serialize` and `BedSeparationConfigSerde::deserialize`
// static methods that serde dispatches to via `#[serde(with = "BedSeparationConfigSerde")]`
// on the `FilterKind::BedSeparation` field.
#[derive(Serialize, Deserialize)]
#[serde(remote = "BedSeparationConfig")]
pub struct BedSeparationConfigSerde {
    pub body_threshold: f32,
    pub background_threshold: f32,
    pub component_policy: ComponentPolicy,
    pub closing_radius: usize,
    pub opening_radius: usize,
    pub outside_value: f32,
}
