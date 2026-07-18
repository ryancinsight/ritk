//! Cine playback timing state for slice navigation.
//!
//! Keeps frame-rate timing logic isolated from app-shell rendering code.

/// Playback state for cine loop over the active axis.
#[derive(Debug, Clone, PartialEq)]
pub struct CinePlayback {
    /// True while playback is active.
    pub enabled: bool,
    /// Target frames per second. Clamped to [1, 60].
    pub fps: f32,
    last_tick_seconds: Option<f64> }

impl Default for CinePlayback {
    fn default() -> Self {
        Self {
            enabled: false,
            fps: 12.0,
            last_tick_seconds: None }
    }
}

impl CinePlayback {
    /// Enable or disable playback at the current wall-clock time.
    pub fn set_enabled(&mut self, enabled: bool, now_seconds: f64) {
        if enabled {
            self.enabled = true;
            self.last_tick_seconds = Some(now_seconds);
        } else {
            self.stop();
        }
    }

    /// Stop playback and clear accumulated timing state.
    pub fn stop(&mut self) {
        self.enabled = false;
        self.last_tick_seconds = None;
    }

    /// Update FPS while preserving bounded domain.
    pub fn set_fps(&mut self, fps: f32) {
        self.fps = fps.clamp(1.0, 60.0);
    }

    /// Restore state from a saved session.
    pub fn restore(&mut self, enabled: bool, fps: f32) {
        self.fps = fps.clamp(1.0, 60.0);
        self.enabled = enabled;
        self.last_tick_seconds = None;
    }

    /// Return the number of frame steps that should be advanced at `now_seconds`.
    ///
    /// The result is capped to avoid huge jumps after long pauses.
    pub fn consume_steps(&mut self, now_seconds: f64) -> u32 {
        if !self.enabled {
            return 0;
        }

        let frame_dt = 1.0 / self.fps.clamp(1.0, 60.0) as f64;
        let Some(last) = self.last_tick_seconds else {
            self.last_tick_seconds = Some(now_seconds);
            return 0;
        };

        let elapsed = (now_seconds - last).max(0.0);
        if elapsed < frame_dt {
            return 0;
        }

        let steps = (elapsed / frame_dt).floor() as u32;
        let capped_steps = steps.min(64);
        self.last_tick_seconds = Some(last + capped_steps as f64 * frame_dt);
        capped_steps
    }
}

#[cfg(test)]
mod tests {
    use super::CinePlayback;

    #[test]
    fn default_is_paused_with_12_fps() {
        let cine = CinePlayback::default();
        assert!(!cine.enabled);
        assert_eq!(cine.fps, 12.0);
    }

    #[test]
    fn consume_steps_returns_zero_while_paused() {
        let mut cine = CinePlayback::default();
        assert_eq!(cine.consume_steps(10.0), 0);
    }

    #[test]
    fn consume_steps_advances_on_frame_boundary() {
        let mut cine = CinePlayback::default();
        cine.set_fps(10.0);
        cine.set_enabled(true, 0.0);

        assert_eq!(cine.consume_steps(0.09), 0);
        assert_eq!(cine.consume_steps(0.10), 1);
        assert_eq!(cine.consume_steps(0.31), 2);
    }

    #[test]
    fn consume_steps_caps_large_catch_up() {
        let mut cine = CinePlayback::default();
        cine.set_fps(60.0);
        cine.set_enabled(true, 0.0);

        let steps = cine.consume_steps(10.0);
        assert_eq!(steps, 64);
    }

    #[test]
    fn set_fps_clamps_to_supported_range() {
        let mut cine = CinePlayback::default();
        cine.set_fps(0.5);
        assert_eq!(cine.fps, 1.0);
        cine.set_fps(120.0);
        assert_eq!(cine.fps, 60.0);
    }
}
