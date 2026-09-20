//! Orientation as a value: the rotation step count and the transform that
//! composes it with the two flips.
//!
//! The specification these satisfy is in [`super`]; this module is the state,
//! and [`super::image_ops`] is what applies it to pixels.

#[cfg(feature = "eframe-shell")]
use egui::Pos2;
/// Number of 90° clockwise rotation steps (0=0°, 1=90°, 2=180°, 3=270°).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum RotationSteps {
    #[default]
    Zero,
    Ninety,
    OneEighty,
    TwoSeventy,
}

impl RotationSteps {
    /// Advance by one 90° clockwise step.
    pub fn rotate_cw(self) -> Self {
        match self {
            Self::Zero => Self::Ninety,
            Self::Ninety => Self::OneEighty,
            Self::OneEighty => Self::TwoSeventy,
            Self::TwoSeventy => Self::Zero,
        }
    }

    /// Reverse by one 90° step (counter-clockwise).
    pub fn rotate_ccw(self) -> Self {
        match self {
            Self::Zero => Self::TwoSeventy,
            Self::Ninety => Self::Zero,
            Self::OneEighty => Self::Ninety,
            Self::TwoSeventy => Self::OneEighty,
        }
    }
}

/// Viewport image orientation state.
///
/// Encodes the sequence of flip and rotation transforms applied to each
/// rendered slice before display. The transform is stateless and deterministic:
/// equal `ViewTransform` values produce identical output for identical input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct ViewTransform {
    /// Mirror the image about its vertical axis (left↔right).
    pub flip_h: bool,
    /// Mirror the image about its horizontal axis (up↔down).
    pub flip_v: bool,
    /// Clockwise rotation applied after flips.
    pub rotation: RotationSteps,
}

impl ViewTransform {
    /// True when the transform is the identity (no flip, no rotation).
    pub fn is_identity(self) -> bool {
        !self.flip_h && !self.flip_v && self.rotation == RotationSteps::Zero
    }

    /// Toggle horizontal flip.
    pub fn toggle_flip_h(self) -> Self {
        Self {
            flip_h: !self.flip_h,
            ..self
        }
    }

    /// Toggle vertical flip.
    pub fn toggle_flip_v(self) -> Self {
        Self {
            flip_v: !self.flip_v,
            ..self
        }
    }

    /// Advance rotation by one 90° clockwise step.
    pub fn rotate_cw(self) -> Self {
        Self {
            rotation: self.rotation.rotate_cw(),
            ..self
        }
    }

    /// Advance rotation by one 90° counter-clockwise step.
    pub fn rotate_ccw(self) -> Self {
        Self {
            rotation: self.rotation.rotate_ccw(),
            ..self
        }
    }

    /// Reset to identity.
    pub fn reset(self) -> Self {
        Self::default()
    }

    /// Return the displayed `(width, height)` for a source image size.
    ///
    /// Pixel transforms use source coordinates at pixel centres, while the
    /// continuous point methods below use image-edge coordinates. Keeping the
    /// dimensions here makes both paths share the same rotation contract.
    #[must_use]
    pub fn output_size(self, source_size: [usize; 2]) -> [usize; 2] {
        match self.rotation {
            RotationSteps::Zero | RotationSteps::OneEighty => source_size,
            RotationSteps::Ninety | RotationSteps::TwoSeventy => [source_size[1], source_size[0]],
        }
    }

    /// Map a source image-edge point `(x=column, y=row)` to display space.
    ///
    /// Coordinates are measured from the outer image edges, so a source image
    /// of size `[width, height]` occupies `[0,width] × [0,height]`. This is the
    /// continuous counterpart of the pixel mapping used by
    /// [`crate::ui::view_transform::apply_to_image`], and therefore maps
    /// annotation geometry without a
    /// half-pixel drift.
    #[must_use]
    #[cfg(feature = "eframe-shell")]
    pub fn source_to_output(self, point: Pos2, source_size: [usize; 2]) -> Pos2 {
        let [x, y] = self
            .source_to_output_coordinates([f64::from(point.x), f64::from(point.y)], source_size);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the eframe point contract is f32 and the caller supplied f32 coordinates"
        )]
        Pos2::new(x as f32, y as f32)
    }

    /// Map a source image-edge point `(x=column, y=row)` to display space.
    ///
    /// This host-neutral form keeps the orientation equation available to the
    /// native Métis compositor as well as the eframe shell. Coordinates are
    /// measured from the outer image edges, so `[width, height]` occupies
    /// `[0,width] × [0,height]`.
    #[must_use]
    pub fn source_to_output_coordinates(
        self,
        point: [f64; 2],
        source_size: [usize; 2],
    ) -> [f64; 2] {
        let [width, height] = source_size;
        let width = width as f64;
        let height = height as f64;
        let mut x = point[0];
        let mut y = point[1];
        if self.flip_h {
            x = width - x;
        }
        if self.flip_v {
            y = height - y;
        }
        match self.rotation {
            RotationSteps::Zero => [x, y],
            RotationSteps::Ninety => [height - y, x],
            RotationSteps::OneEighty => [width - x, height - y],
            RotationSteps::TwoSeventy => [y, width - x],
        }
    }

    /// Map a display image-edge point back to source coordinates.
    ///
    /// This is the exact inverse of [`Self::source_to_output`] for every
    /// transform and source size, including non-square images.
    #[must_use]
    #[cfg(feature = "eframe-shell")]
    pub fn output_to_source(self, point: Pos2, source_size: [usize; 2]) -> Pos2 {
        let [x, y] = self
            .output_to_source_coordinates([f64::from(point.x), f64::from(point.y)], source_size);
        Pos2::new(x as f32, y as f32)
    }

    /// Map a display image-edge point back to source coordinates without
    /// narrowing the coordinate arithmetic.
    ///
    /// The returned array uses `f64` so a host can subtract a large viewport
    /// origin and apply its scale before the viewer's `f32` image-space model
    /// is reached. The caller must provide finite image-edge coordinates.
    #[must_use]
    pub fn output_to_source_coordinates(
        self,
        point: [f64; 2],
        source_size: [usize; 2],
    ) -> [f64; 2] {
        let [width, height] = source_size;
        let width = width as f64;
        let height = height as f64;
        let (mut x, mut y) = match self.rotation {
            RotationSteps::Zero => (point[0], point[1]),
            RotationSteps::Ninety => (point[1], height - point[0]),
            RotationSteps::OneEighty => (width - point[0], height - point[1]),
            RotationSteps::TwoSeventy => (width - point[1], point[0]),
        };
        if self.flip_h {
            x = width - x;
        }
        if self.flip_v {
            y = height - y;
        }
        [x, y]
    }
}
