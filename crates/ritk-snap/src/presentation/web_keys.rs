//! Map browser keyboard codes onto RITK's shared virtual-key contract.

#[cfg(any(target_arch = "wasm32", test))]
use crate::app::action_adapter::{
    VIRTUAL_KEY_ARROW_DOWN, VIRTUAL_KEY_ARROW_UP, VIRTUAL_KEY_CINE_FPS_DOWN,
    VIRTUAL_KEY_CINE_FPS_UP, VIRTUAL_KEY_CINE_TOGGLE, VIRTUAL_KEY_CROSSHAIR_TOGGLE,
    VIRTUAL_KEY_END, VIRTUAL_KEY_HOME, VIRTUAL_KEY_PAGE_DOWN, VIRTUAL_KEY_PAGE_UP,
};
#[cfg(any(target_arch = "wasm32", test))]
use crate::ui::tool_shortcuts::{
    VIRTUAL_KEY_HU_POINT, VIRTUAL_KEY_LABEL_PAINT, VIRTUAL_KEY_MEASURE_ANGLE,
    VIRTUAL_KEY_MEASURE_LENGTH, VIRTUAL_KEY_PAN, VIRTUAL_KEY_ROI_ELLIPSE, VIRTUAL_KEY_ROI_RECT,
    VIRTUAL_KEY_WINDOW_LEVEL, VIRTUAL_KEY_ZOOM,
};

/// Returns the shared virtual-key value for a bounded browser key snapshot.
///
/// `code` is preferred because it identifies the physical key independently
/// of the active layout. The `key` fallback preserves the symbol shortcuts
/// when a host supplies an empty or non-standard code value.
#[must_use]
#[cfg(any(target_arch = "wasm32", test))]
pub(crate) fn virtual_key_for_browser(key: &str, code: &str) -> Option<u32> {
    let mapped = match code {
        "Space" => Some(VIRTUAL_KEY_CINE_TOGGLE),
        "KeyX" => Some(VIRTUAL_KEY_CROSSHAIR_TOGGLE),
        "Equal" => Some(VIRTUAL_KEY_CINE_FPS_UP),
        "Minus" => Some(VIRTUAL_KEY_CINE_FPS_DOWN),
        "ArrowUp" => Some(VIRTUAL_KEY_ARROW_UP),
        "ArrowDown" => Some(VIRTUAL_KEY_ARROW_DOWN),
        "PageUp" => Some(VIRTUAL_KEY_PAGE_UP),
        "PageDown" => Some(VIRTUAL_KEY_PAGE_DOWN),
        "Home" => Some(VIRTUAL_KEY_HOME),
        "End" => Some(VIRTUAL_KEY_END),
        "KeyA" => Some(VIRTUAL_KEY_MEASURE_ANGLE),
        "KeyB" => Some(VIRTUAL_KEY_LABEL_PAINT),
        "KeyE" => Some(VIRTUAL_KEY_ROI_ELLIPSE),
        "KeyH" => Some(VIRTUAL_KEY_HU_POINT),
        "KeyL" => Some(VIRTUAL_KEY_MEASURE_LENGTH),
        "KeyP" => Some(VIRTUAL_KEY_PAN),
        "KeyR" => Some(VIRTUAL_KEY_ROI_RECT),
        "KeyW" => Some(VIRTUAL_KEY_WINDOW_LEVEL),
        "KeyZ" => Some(VIRTUAL_KEY_ZOOM),
        _ => None,
    };
    mapped.or(match key {
        " " => Some(VIRTUAL_KEY_CINE_TOGGLE),
        "+" | "=" => Some(VIRTUAL_KEY_CINE_FPS_UP),
        "-" | "_" => Some(VIRTUAL_KEY_CINE_FPS_DOWN),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::{virtual_key_for_browser, VIRTUAL_KEY_CINE_FPS_DOWN, VIRTUAL_KEY_CINE_FPS_UP};
    use crate::app::action_adapter::{VIRTUAL_KEY_CINE_TOGGLE, VIRTUAL_KEY_CROSSHAIR_TOGGLE};

    #[test]
    fn browser_code_maps_to_shared_virtual_key() {
        assert_eq!(
            virtual_key_for_browser("+", "Equal"),
            Some(VIRTUAL_KEY_CINE_FPS_UP)
        );
        assert_eq!(
            virtual_key_for_browser("-", "Minus"),
            Some(VIRTUAL_KEY_CINE_FPS_DOWN)
        );
        assert_eq!(
            virtual_key_for_browser(" ", "Space"),
            Some(VIRTUAL_KEY_CINE_TOGGLE)
        );
        assert_eq!(
            virtual_key_for_browser("x", "KeyX"),
            Some(VIRTUAL_KEY_CROSSHAIR_TOGGLE)
        );
    }

    #[test]
    fn browser_key_fallback_handles_symbols() {
        assert_eq!(
            virtual_key_for_browser("+", ""),
            Some(VIRTUAL_KEY_CINE_FPS_UP)
        );
        assert_eq!(
            virtual_key_for_browser("_", ""),
            Some(VIRTUAL_KEY_CINE_FPS_DOWN)
        );
        assert_eq!(virtual_key_for_browser("unmapped", "Unknown"), None);
    }
}
