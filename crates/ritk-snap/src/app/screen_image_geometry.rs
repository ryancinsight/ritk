//! Checked affine mapping between host coordinates and a raster.

use crate::presentation::ViewportPoint;

/// Affine screen mapping for a finite raster whose coordinates use image edges.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ScreenImageGeometry {
    origin: [f64; 2],
    pixel_size: [f64; 2],
    dimensions: [u32; 2],
    far_edge: [f64; 2],
}

impl ScreenImageGeometry {
    /// Validate a displayed raster rectangle and retain its checked far edge.
    pub(crate) fn try_new(
        origin: [f64; 2],
        pixel_size: [f64; 2],
        dimensions: [u32; 2],
    ) -> Option<Self> {
        if !origin.into_iter().all(f64::is_finite)
            || !pixel_size.into_iter().all(f64::is_finite)
            || pixel_size.into_iter().any(|value| value <= 0.0)
            || dimensions.contains(&0)
        {
            return None;
        }
        let far_edge = std::array::from_fn(|axis| {
            f64::from(dimensions[axis]).mul_add(pixel_size[axis], origin[axis])
        });
        if far_edge
            .into_iter()
            .zip(origin)
            .any(|(edge, start)| !edge.is_finite() || edge <= start)
        {
            return None;
        }
        Some(Self {
            origin,
            pixel_size,
            dimensions,
            far_edge,
        })
    }

    /// Return the checked pixel spacing in host coordinates.
    #[cfg(test)]
    pub(crate) const fn pixel_size(self) -> [f64; 2] {
        self.pixel_size
    }

    /// Return the checked half-open image rectangle as `[left, right, top, bottom]`.
    #[cfg(test)]
    pub(crate) const fn bounds(self) -> [f64; 4] {
        [
            self.origin[0],
            self.far_edge[0],
            self.origin[1],
            self.far_edge[1],
        ]
    }

    /// Map a host point into continuous image-edge coordinates.
    pub(crate) fn map_edge(self, point: ViewportPoint) -> Option<[f64; 2]> {
        let position = [point.x(), point.y()];
        if !position.into_iter().all(f64::is_finite)
            || (0..2).any(|axis| {
                position[axis] < self.origin[axis] || position[axis] >= self.far_edge[axis]
            })
        {
            return None;
        }
        let mut pixel = std::array::from_fn(|axis| {
            let distance = position[axis] - self.origin[axis];
            if distance.is_finite() {
                distance / self.pixel_size[axis]
            } else {
                position[axis] / self.pixel_size[axis] - self.origin[axis] / self.pixel_size[axis]
            }
        });
        for axis in 0..2 {
            let coordinate = pixel[axis];
            let extent = f64::from(self.dimensions[axis]);
            if !coordinate.is_finite() || coordinate < 0.0 || coordinate > extent {
                return None;
            }
            if coordinate == extent {
                pixel[axis] = extent.next_down();
            }
        }
        Some(pixel)
    }

    /// Map a host point into continuous pixel-centre coordinates.
    ///
    /// The first and last half-pixels clamp to their edge pixel, while the
    /// half-open far edge remains outside the raster. Exact rendered centres
    /// are matched before inversion because large host-coordinate origins can
    /// round away part of the pixel spacing.
    #[cfg(test)]
    pub(crate) fn map_pixel_center(self, point: ViewportPoint) -> Option<[f64; 2]> {
        let edge = self.map_edge(point)?;
        let position = [point.x(), point.y()];
        Some(std::array::from_fn(|axis| {
            let coordinate = edge[axis] - 0.5;
            let lower = coordinate.floor();
            let upper = coordinate.ceil();
            let exact_center = [lower, upper].into_iter().find(|candidate| {
                *candidate >= 0.0
                    && *candidate < f64::from(self.dimensions[axis])
                    && (*candidate + 0.5).mul_add(self.pixel_size[axis], self.origin[axis])
                        == position[axis]
            });
            match exact_center {
                Some(pixel) => pixel,
                None => coordinate.clamp(0.0, f64::from(self.dimensions[axis] - 1)),
            }
        }))
    }

    /// Map a continuous pixel-centre coordinate to host coordinates.
    #[cfg(test)]
    pub(crate) fn pixel_center_to_screen(self, pixel: [f64; 2]) -> Option<[f64; 2]> {
        if pixel.into_iter().enumerate().any(|(axis, coordinate)| {
            !coordinate.is_finite()
                || coordinate < 0.0
                || coordinate > f64::from(self.dimensions[axis] - 1)
        }) {
            return None;
        }
        let screen = std::array::from_fn(|axis| {
            (pixel[axis] + 0.5).mul_add(self.pixel_size[axis], self.origin[axis])
        });
        if screen.into_iter().enumerate().any(|(axis, coordinate)| {
            !coordinate.is_finite()
                || coordinate <= self.origin[axis]
                || coordinate >= self.far_edge[axis]
        }) {
            return None;
        }
        Some(screen)
    }

    /// Return the validated raster dimensions.
    pub(crate) const fn dimensions(self) -> [u32; 2] {
        self.dimensions
    }
}
