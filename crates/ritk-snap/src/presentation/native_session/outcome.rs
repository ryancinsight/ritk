//! Observable results from an interactive native viewer session.\n\n/// Observable result of an interactive native viewer session.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NativeViewerOutcome {
    pub(crate) surface_width: u32,
    pub(crate) surface_height: u32,
    pub(crate) initial_frame_width: u32,
    pub(crate) initial_frame_height: u32,
    pub(crate) view_count: usize,
    pub(crate) presented_frames: usize,
    pub(crate) event_batches: usize,
    pub(crate) translated_events: usize,
    pub(crate) frame_generations: usize,
    pub(crate) last_slice: usize,
    pub(crate) zoom: f32,
    pub(crate) dpi: u32,
    pub(crate) minimized: bool,
    pub(crate) destroyed: bool,
}

impl NativeViewerOutcome {
    /// Final native surface width in client pixels.
    #[must_use]
    pub const fn surface_width(self) -> u32 {
        self.surface_width
    }

    /// Final native surface height in client pixels.
    #[must_use]
    pub const fn surface_height(self) -> u32 {
        self.surface_height
    }

    /// Width of the decoded RITK slice before host scaling.
    #[must_use]
    pub const fn initial_frame_width(self) -> u32 {
        self.initial_frame_width
    }

    /// Height of the decoded RITK slice before host scaling.
    #[must_use]
    pub const fn initial_frame_height(self) -> u32 {
        self.initial_frame_height
    }

    /// Number of orthogonal RITK views composed into each native frame.
    #[must_use]
    pub const fn view_count(self) -> usize {
        self.view_count
    }

    /// Number of framebuffer presentations requested by the host loop.
    #[must_use]
    pub const fn presented_frames(self) -> usize {
        self.presented_frames
    }

    /// Number of bounded event batches consumed by the viewer.
    #[must_use]
    pub const fn event_batches(self) -> usize {
        self.event_batches
    }

    /// Number of native events translated at the RITK boundary.
    #[must_use]
    pub const fn translated_events(self) -> usize {
        self.translated_events
    }

    /// Number of decoded frames rendered after state transitions.
    #[must_use]
    pub const fn frame_generations(self) -> usize {
        self.frame_generations
    }

    /// Final slice index along the active orthogonal axis.
    #[must_use]
    pub const fn last_slice(self) -> usize {
        self.last_slice
    }

    /// Final RITK zoom value.
    #[must_use]
    pub const fn zoom(self) -> f32 {
        self.zoom
    }

    /// Final display DPI reported by the host.
    #[must_use]
    pub const fn dpi(self) -> u32 {
        self.dpi
    }

    /// Whether the last host event left the surface minimized.
    #[must_use]
    pub const fn minimized(self) -> bool {
        self.minimized
    }

    /// Whether the host reported destruction rather than an orderly close.
    #[must_use]
    pub const fn destroyed(self) -> bool {
        self.destroyed
    }
}
