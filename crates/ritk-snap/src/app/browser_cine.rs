//! Typed browser controls for host-neutral cine playback.

use super::SnapApp;

const MIN_CINE_FPS: u8 = 1;
const MAX_CINE_FPS: u8 = 60;

/// Failure to change browser cine state without mutating the viewer.
#[derive(Debug, thiserror::Error, PartialEq)]
pub(crate) enum BrowserCineControlError {
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
    /// A JavaScript number is not finite.
    #[error("browser cine rate {value} must be finite")]
    RateNotFinite {
        /// Rejected JavaScript number.
        value: f64,
    },
    /// A JavaScript number does not denote an integer.
    #[error("browser cine rate {value} must be an integer")]
    RateNotInteger {
        /// Rejected JavaScript number.
        value: f64,
    },
    /// A JavaScript integer is outside the supported playback range.
    #[error("browser cine rate {value} is outside {minimum}..={maximum} FPS")]
    RateOutOfRange {
        /// Rejected JavaScript number.
        value: f64,
        /// Inclusive lower bound in frames per second.
        minimum: u8,
        /// Inclusive upper bound in frames per second.
        maximum: u8,
    },
}

/// Validates a JavaScript cine rate before narrowing it to the browser API.
pub(crate) fn parse_browser_cine_rate_request(value: f64) -> Result<u8, BrowserCineControlError> {
    if !value.is_finite() {
        return Err(BrowserCineControlError::RateNotFinite { value });
    }
    if value.fract() != 0.0 {
        return Err(BrowserCineControlError::RateNotInteger { value });
    }
    if value < f64::from(MIN_CINE_FPS) || value > f64::from(MAX_CINE_FPS) {
        return Err(BrowserCineControlError::RateOutOfRange {
            value,
            minimum: MIN_CINE_FPS,
            maximum: MAX_CINE_FPS,
        });
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "finite integral input is bounded to the complete supported u8 rate range above"
    )]
    let rate = value as u8;
    Ok(rate)
}

impl SnapApp {
    /// Toggles cine playback for the loaded study.
    pub(crate) fn toggle_browser_cine(&mut self) -> Result<bool, BrowserCineControlError> {
        if self.loaded.is_none() {
            return Err(BrowserCineControlError::StudyNotLoaded);
        }
        Ok(self.toggle_cine())
    }

    /// Sets the exact bounded browser cine rate.
    pub(crate) fn set_browser_cine_rate(
        &mut self,
        rate: u8,
    ) -> Result<bool, BrowserCineControlError> {
        if self.loaded.is_none() {
            return Err(BrowserCineControlError::StudyNotLoaded);
        }
        if !(MIN_CINE_FPS..=MAX_CINE_FPS).contains(&rate) {
            return Err(BrowserCineControlError::RateOutOfRange {
                value: f64::from(rate),
                minimum: MIN_CINE_FPS,
                maximum: MAX_CINE_FPS,
            });
        }
        let next = f32::from(rate);
        if self.cine.fps.to_bits() == next.to_bits() {
            return Ok(false);
        }
        self.cine.set_fps(next);
        self.status_message = format!("Cine playback rate: {rate} FPS.");
        Ok(true)
    }

    /// Returns whether browser cine playback is currently enabled.
    #[must_use]
    pub(crate) const fn browser_cine_enabled(&self) -> bool {
        self.cine.enabled
    }

    /// Returns the current browser cine playback rate.
    #[must_use]
    pub(crate) const fn browser_cine_rate(&self) -> f32 {
        self.cine.fps
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::tests::test_volume;

    #[test]
    fn parser_accepts_only_integer_rates_in_the_supported_range() {
        assert_eq!(parse_browser_cine_rate_request(1.0), Ok(1));
        assert_eq!(parse_browser_cine_rate_request(60.0), Ok(60));
        assert_eq!(
            parse_browser_cine_rate_request(12.5),
            Err(BrowserCineControlError::RateNotInteger { value: 12.5 })
        );
        assert!(matches!(
            parse_browser_cine_rate_request(f64::NAN),
            Err(BrowserCineControlError::RateNotFinite { value }) if value.is_nan()
        ));
        assert_eq!(
            parse_browser_cine_rate_request(0.0),
            Err(BrowserCineControlError::RateOutOfRange {
                value: 0.0,
                minimum: MIN_CINE_FPS,
                maximum: MAX_CINE_FPS,
            })
        );
        assert_eq!(
            parse_browser_cine_rate_request(61.0),
            Err(BrowserCineControlError::RateOutOfRange {
                value: 61.0,
                minimum: MIN_CINE_FPS,
                maximum: MAX_CINE_FPS,
            })
        );
    }

    #[test]
    fn unloaded_controls_fail_without_mutating_playback() {
        let mut app = SnapApp::default();
        let enabled = app.browser_cine_enabled();
        let rate = app.browser_cine_rate();
        assert_eq!(
            app.toggle_browser_cine(),
            Err(BrowserCineControlError::StudyNotLoaded)
        );
        assert_eq!(
            app.set_browser_cine_rate(24),
            Err(BrowserCineControlError::StudyNotLoaded)
        );
        assert_eq!(app.browser_cine_enabled(), enabled);
        assert_eq!(app.browser_cine_rate(), rate);
    }

    #[test]
    fn loaded_controls_change_only_the_requested_playback_state() {
        let mut app = SnapApp::default();
        app.loaded = Some(test_volume([4, 3, 2]));
        assert!(!app.browser_cine_enabled());
        assert_eq!(app.set_browser_cine_rate(24), Ok(true));
        assert_eq!(app.browser_cine_rate(), 24.0);
        assert_eq!(app.set_browser_cine_rate(24), Ok(false));
        assert_eq!(app.toggle_browser_cine(), Ok(true));
        assert!(app.browser_cine_enabled());
        assert_eq!(app.toggle_browser_cine(), Ok(false));
        assert!(!app.browser_cine_enabled());
    }
}
