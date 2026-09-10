//! Framework-independent points used by viewer interaction state.

/// A point in the displayed slice's image-pixel coordinate system.
///
/// The x component is the column and the y component is the row. The point is
/// kept independent of a presentation toolkit so a host can translate its
/// coordinates into viewer actions without storing GUI values in RITK state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImagePoint {
    x: f32,
    y: f32,
}

impl ImagePoint {
    /// Construct an image-space point.
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Return the column coordinate.
    #[must_use]
    pub const fn x(self) -> f32 {
        self.x
    }

    /// Return the row coordinate.
    #[must_use]
    pub const fn y(self) -> f32 {
        self.y
    }
}

/// A viewport pan offset measured in screen pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewportOffset {
    x: f32,
    y: f32,
}

impl ViewportOffset {
    /// Construct a viewport offset.
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Return the horizontal offset.
    #[must_use]
    pub const fn x(self) -> f32 {
        self.x
    }

    /// Return the vertical offset.
    #[must_use]
    pub const fn y(self) -> f32 {
        self.y
    }
}
