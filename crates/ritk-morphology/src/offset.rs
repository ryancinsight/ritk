//! 3-D integer offset type for structuring elements.
//!
//! See the `ritk_morphology` crate root for the umbrella module and
//! [`StructuringElement`](crate::StructuringElement) for the
//! consumer type that stores collections of [`Offset3D`].

use std::fmt;

/// A 3-D integer offset `(Δz, Δy, Δx)`.
///
/// `Offset3D` is a `#[repr(transparent)]` newtype over `[i32; 3]` to enforce
/// domain separation from other 3-element integer arrays while preserving
/// ABI compatibility with `[i32; 3]` for FFI/transmutation use cases.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct Offset3D(pub [i32; 3]);

impl Offset3D {
    /// Construct a new offset from `(Δz, Δy, Δx)` components.
    #[inline]
    pub const fn new(iz: i32, iy: i32, ix: i32) -> Self {
        Self([iz, iy, ix])
    }

    /// The z-axis offset.
    #[inline]
    pub const fn iz(self) -> i32 {
        self.0[0]
    }

    /// The y-axis offset.
    #[inline]
    pub const fn iy(self) -> i32 {
        self.0[1]
    }

    /// The x-axis offset.
    #[inline]
    pub const fn ix(self) -> i32 {
        self.0[2]
    }
}

impl From<[i32; 3]> for Offset3D {
    #[inline]
    fn from(v: [i32; 3]) -> Self {
        Self(v)
    }
}

impl From<Offset3D> for [i32; 3] {
    #[inline]
    fn from(o: Offset3D) -> Self {
        o.0
    }
}

impl fmt::Display for Offset3D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Offset3D(Δz={}, Δy={}, Δx={})",
            self.iz(),
            self.iy(),
            self.ix()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Offset3D` accessors return the corresponding components.
    #[test]
    fn offset3d_accessors() {
        let o = Offset3D::new(-3, 7, 11);
        assert_eq!(o.iz(), -3);
        assert_eq!(o.iy(), 7);
        assert_eq!(o.ix(), 11);
    }

    /// `Offset3D` is `#[repr(transparent)]` — same ABI as `[i32; 3]`.
    #[test]
    fn offset3d_has_transparent_layout() {
        assert_eq!(
            std::mem::size_of::<Offset3D>(),
            std::mem::size_of::<[i32; 3]>(),
            "Offset3D must be layout-compatible with [i32; 3]"
        );
    }

    /// `Offset3D` round-trips through `[i32; 3]` via `From`/`Into`.
    #[test]
    fn offset3d_from_into_array() {
        let o = Offset3D::new(1, 2, 3);
        let arr: [i32; 3] = o.into();
        assert_eq!(arr, [1, 2, 3]);
        let back: Offset3D = arr.into();
        assert_eq!(back, Offset3D::new(1, 2, 3));
    }

    /// `Display` includes the three components.
    #[test]
    fn offset3d_display() {
        let s = format!("{}", Offset3D::new(-1, 2, -3));
        assert!(s.contains("Δz=-1"), "display must mention Δz: {s}");
        assert!(s.contains("Δy=2"), "display must mention Δy: {s}");
        assert!(s.contains("Δx=-3"), "display must mention Δx: {s}");
    }
}
