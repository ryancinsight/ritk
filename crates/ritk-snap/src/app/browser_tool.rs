//! Typed browser access to the RITK interaction-tool contract.

use super::SnapApp;
use crate::tools::kind::ToolKind;

/// Failure to select or inspect a browser interaction tool.
#[derive(Debug, thiserror::Error, PartialEq)]
pub(crate) enum BrowserToolError {
    /// No browser viewer is mounted.
    #[cfg(target_arch = "wasm32")]
    #[error("RITK browser viewer is not mounted")]
    ViewerNotMounted,
    /// Another browser callback currently owns the viewer state.
    #[cfg(target_arch = "wasm32")]
    #[error("RITK browser viewer is handling another callback")]
    ViewerBusy,
    /// The mounted viewer has not loaded a study.
    #[error("RITK browser viewer has no loaded study")]
    StudyNotLoaded,
    /// The requested tool index is outside the browser tool table.
    #[error("browser tool index {index} is outside 0..{tool_count}")]
    IndexOutOfRange {
        /// Rejected zero-based tool index.
        index: usize,
        /// Number of tools in the table.
        tool_count: usize,
    },
    /// A JavaScript number is not finite.
    #[error("browser tool index {value} must be finite")]
    CoordinateNotFinite {
        /// Rejected JavaScript number.
        value: f64,
    },
    /// A JavaScript number does not denote an integer.
    #[error("browser tool index {value} must be an integer")]
    CoordinateNotInteger {
        /// Rejected JavaScript number.
        value: f64,
    },
    /// A JavaScript integer cannot be represented by the WASM ABI.
    #[error("browser tool index {value} is outside 0..={maximum}")]
    CoordinateOutOfRange {
        /// Rejected JavaScript number.
        value: f64,
        /// Inclusive upper bound of the WASM ABI integer representation.
        maximum: u32,
    },
}

/// Validates a JavaScript number before narrowing it to a tool index.
pub(crate) fn parse_browser_tool_request(value: f64) -> Result<usize, BrowserToolError> {
    if !value.is_finite() {
        return Err(BrowserToolError::CoordinateNotFinite { value });
    }
    if value.fract() != 0.0 {
        return Err(BrowserToolError::CoordinateNotInteger { value });
    }
    if value < 0.0 || value > f64::from(u32::MAX) {
        return Err(BrowserToolError::CoordinateOutOfRange {
            value,
            maximum: u32::MAX,
        });
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "finite integral input is bounded to the complete u32 range above"
    )]
    let value = value as u32;
    usize::try_from(value).map_err(|_| BrowserToolError::CoordinateOutOfRange {
        value: f64::from(value),
        maximum: u32::MAX,
    })
}

impl SnapApp {
    /// Returns the stable browser interaction-tool count.
    pub(crate) fn browser_tool_count() -> usize {
        ToolKind::all().len()
    }

    /// Returns one stable browser interaction-tool label.
    pub(crate) fn browser_tool_name(index: usize) -> Result<&'static str, BrowserToolError> {
        ToolKind::all()
            .get(index)
            .map(ToolKind::label)
            .ok_or(BrowserToolError::IndexOutOfRange {
                index,
                tool_count: Self::browser_tool_count(),
            })
    }

    /// Selects one loaded-study interaction tool without changing pixels.
    pub(crate) fn select_browser_tool(&mut self, index: usize) -> Result<bool, BrowserToolError> {
        if self.loaded.is_none() {
            return Err(BrowserToolError::StudyNotLoaded);
        }
        let tool = *ToolKind::all()
            .get(index)
            .ok_or(BrowserToolError::IndexOutOfRange {
                index,
                tool_count: Self::browser_tool_count(),
            })?;
        if self.active_tool == tool {
            return Ok(false);
        }
        self.active_tool = tool;
        self.tool_state = crate::tools::interaction::ToolState::Idle;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::tests::test_volume;

    #[test]
    fn parser_rejects_non_integral_and_unrepresentable_indices() {
        assert_eq!(
            parse_browser_tool_request(1.5),
            Err(BrowserToolError::CoordinateNotInteger { value: 1.5 })
        );
        assert!(matches!(
            parse_browser_tool_request(f64::NAN),
            Err(BrowserToolError::CoordinateNotFinite { value }) if value.is_nan()
        ));
        assert_eq!(
            parse_browser_tool_request(-1.0),
            Err(BrowserToolError::CoordinateOutOfRange {
                value: -1.0,
                maximum: u32::MAX,
            })
        );
    }

    #[test]
    fn invalid_tool_selection_leaves_state_unchanged() {
        let mut app = SnapApp::default();
        app.loaded = Some(test_volume([4, 3, 2]));
        let active = app.active_tool;
        assert_eq!(
            app.select_browser_tool(usize::MAX),
            Err(BrowserToolError::IndexOutOfRange {
                index: usize::MAX,
                tool_count: ToolKind::all().len(),
            })
        );
        assert_eq!(app.active_tool, active);
        assert!(app.tool_state.is_idle());
    }

    #[test]
    fn valid_tool_selection_is_input_sensitive_and_clears_gesture() {
        let mut app = SnapApp::default();
        app.loaded = Some(test_volume([4, 3, 2]));
        app.tool_state = crate::tools::interaction::ToolState::MeasureLength1 {
            p1: crate::tools::interaction::ImagePoint::new(1.0, 2.0),
        };
        assert_eq!(app.select_browser_tool(0), Ok(true));
        assert_eq!(app.active_tool, ToolKind::Pan);
        assert!(app.tool_state.is_idle());
        assert_eq!(app.select_browser_tool(0), Ok(false));
        assert_eq!(SnapApp::browser_tool_name(0), Ok("Pan"));
    }

    #[test]
    fn unloaded_tool_selection_is_rejected_without_mutation() {
        let mut app = SnapApp::default();
        let active = app.active_tool;
        assert_eq!(
            app.select_browser_tool(0),
            Err(BrowserToolError::StudyNotLoaded)
        );
        assert_eq!(app.active_tool, active);
    }
}
