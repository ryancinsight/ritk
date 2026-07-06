//! Object-erosion morphology filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! Ports `itk::ErodeObjectMorphologyImageFilter` (box structuring element). The
//! output starts as a copy of the input. Every voxel equal to `object_value`
//! that is **on the object boundary** — i.e. at least one of its `3×3×3`
//! immediate neighbours is not `object_value`, with out-of-image neighbours
//! treated as non-object (ITK's erode boundary condition `= max`) — paints its
//! entire `(2r+1)^3` box footprint in the output with `background_value`.
//!
//! Equivalently, a shell of thickness `r` is removed from every object surface,
//! including surfaces that abut the image border. This differs from grayscale
//! erosion only at the image edge (grayscale erosion does not erode border-
//! touching objects); the interior is identical.
//!
//! # ITK parity
//!
//! Corresponds to `itk::ErodeObjectMorphologyImageFilter` with a box kernel
//! (`sitk.ErodeObjectMorphology(..., sitkBox, objectValue, backgroundValue)`),
//! default radius `[1, 1, 1]`, `object_value = 1`, `background_value = 0`.

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Object-erosion morphology filter (`itk::ErodeObjectMorphologyImageFilter`, box SE).
#[derive(Debug, Clone, Copy)]
pub struct ErodeObjectMorphologyFilter {
    /// Per-axis box radii `[rz, ry, rx]`. ITK default `[1, 1, 1]`.
    pub radius: [usize; 3],
    /// Pixel value identifying the object. ITK default `1.0`.
    pub object_value: f32,
    /// Value written to eroded voxels. ITK default `0.0`.
    pub background_value: f32,
}

impl Default for ErodeObjectMorphologyFilter {
    fn default() -> Self {
        Self {
            radius: [1, 1, 1],
            object_value: 1.0,
            background_value: 0.0,
        }
    }
}

impl ErodeObjectMorphologyFilter {
    /// Construct with explicit radius and object/background values.
    pub fn new(radius: [usize; 3], object_value: f32, background_value: f32) -> Self {
        Self {
            radius,
            object_value,
            background_value,
        }
    }

    /// Erode the object surface by the box structuring element.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (vals, dims) = extract_vec_infallible(image);
        let [nz, ny, nx] = dims;
        let [rz, ry, rx] = self.radius;
        let obj = self.object_value;
        let bg = self.background_value;
        let mut out = vals.clone();
        let idx = |z: usize, y: usize, x: usize| (z * ny + y) * nx + x;

        // Boundary-scan radius per axis: 0 for a size-1 (z=1-promoted 2-D) axis
        // so its absent border is not mistaken for a non-object neighbour — a
        // genuine 3-D axis (size > 1) keeps radius 1, where out-of-image still
        // counts as non-object (ITK's erode boundary condition).
        let bz = if nz > 1 { 1i32 } else { 0 };
        let by = if ny > 1 { 1i32 } else { 0 };
        let bx = if nx > 1 { 1i32 } else { 0 };

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    if vals[idx(z, y, x)] != obj {
                        continue;
                    }
                    // Boundary: any neighbour in the (existing) 3×3×3 window is
                    // non-object, with out-of-image neighbours counted as
                    // non-object.
                    let mut boundary = false;
                    'scan: for dz in -bz..=bz {
                        for dy in -by..=by {
                            for dx in -bx..=bx {
                                let nz_ = z as i32 + dz;
                                let ny_ = y as i32 + dy;
                                let nx_ = x as i32 + dx;
                                if nz_ < 0
                                    || ny_ < 0
                                    || nx_ < 0
                                    || nz_ >= nz as i32
                                    || ny_ >= ny as i32
                                    || nx_ >= nx as i32
                                    || vals[idx(nz_ as usize, ny_ as usize, nx_ as usize)] != obj
                                {
                                    boundary = true;
                                    break 'scan;
                                }
                            }
                        }
                    }
                    if !boundary {
                        continue;
                    }
                    // Paint the (2r+1)^3 box footprint with the background value.
                    let z0 = z.saturating_sub(rz);
                    let z1 = (z + rz).min(nz - 1);
                    let y0 = y.saturating_sub(ry);
                    let y1 = (y + ry).min(ny - 1);
                    let x0 = x.saturating_sub(rx);
                    let x1 = (x + rx).min(nx - 1);
                    for pz in z0..=z1 {
                        for py in y0..=y1 {
                            for px in x0..=x1 {
                                out[idx(pz, py, px)] = bg;
                            }
                        }
                    }
                }
            }
        }
        rebuild(out, dims, image)
    }
}

#[cfg(test)]
#[path = "tests_erode_object.rs"]
mod tests_erode_object;
