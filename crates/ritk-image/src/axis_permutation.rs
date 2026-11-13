//! Signed axis permutations between two orientations of one `[z, y, x]` grid.
//!
//! A direction whose columns are each a signed world basis vector (one entry
//! exactly `±1.0`, the others exactly `0.0`) is a signed axis permutation.
//! [`SignedAxisPermutation::between`] finds the relabeling between two such
//! directions and [`SignedAxisPermutation::reexpress`] applies it to the voxel
//! buffer, shape, spacing and origin together, so every voxel keeps its world
//! point `p = origin + D · diag(spacing) · index`. The decode is exact: an
//! entry one rounding step from `0` or `±1` is not a signed permutation, so a
//! caller that cannot store an oblique direction refuses it instead of
//! snapping it. A caller that does snap an oblique direction to its nearest
//! axes builds the relabeling with [`SignedAxisPermutation::from_axes`].

use std::borrow::Cow;

use ritk_spatial::{Direction, Point, Spacing, Vector};

/// Relabeling of a `[z, y, x]` grid's tensor axes, each possibly reversed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SignedAxisPermutation {
    /// `axis[o]` is the input tensor axis that output tensor axis `o` reads.
    axis: [usize; 3],
    /// `reversed[o]` walks that input axis from its last index.
    reversed: [bool; 3],
}

/// A grid re-expressed by a [`SignedAxisPermutation`].
#[derive(Debug, Clone, PartialEq)]
pub struct PermutedGrid<'a, T: Clone> {
    /// Voxels in output `[z, y, x]` order; borrowed when the permutation is the
    /// identity.
    pub values: Cow<'a, [T]>,
    /// Output tensor extents `[nz, ny, nx]`.
    pub dims: [usize; 3],
    /// World point of output index `(0, 0, 0)`.
    pub origin: Point<3>,
    /// Output spacing in tensor-axis order.
    pub spacing: Spacing<3>,
}

/// World axis and sign of each column, when every column is a signed basis vector.
fn signed_columns(direction: &Direction<3>) -> Option<[(usize, bool); 3]> {
    let mut columns = [(0, false); 3];
    let mut seen = [false; 3];
    for (column, slot) in columns.iter_mut().enumerate() {
        let mut found = None;
        for row in 0..3 {
            let entry = direction[(row, column)];
            if entry == 0.0 {
                continue;
            }
            if found.is_some() || (entry != 1.0 && entry != -1.0) {
                return None;
            }
            found = Some((row, entry < 0.0));
        }
        let (row, negative) = found?;
        if seen[row] {
            return None;
        }
        seen[row] = true;
        *slot = (row, negative);
    }
    Some(columns)
}

/// Voxel count of a grid with extents `dims`, or `None` when it exceeds
/// `usize`.
///
/// A zero extent makes the grid empty, so the count is zero before any
/// product is formed; a running product would otherwise overflow or not
/// depending on where the zero sits.
fn voxel_count(dims: [usize; 3]) -> Option<usize> {
    if dims.contains(&0) {
        return Some(0);
    }
    dims.iter()
        .try_fold(1usize, |count, &extent| count.checked_mul(extent))
}

impl SignedAxisPermutation {
    /// The relabeling that turns a grid with direction `from` into the same
    /// physical grid with direction `to`.
    ///
    /// Returns `None` unless both directions are exact signed axis
    /// permutations.
    pub fn between(from: &Direction<3>, to: &Direction<3>) -> Option<Self> {
        let source = signed_columns(from)?;
        let target = signed_columns(to)?;
        let mut axis = [0; 3];
        let mut reversed = [false; 3];
        for (output, &(world, negative)) in target.iter().enumerate() {
            let input = source.iter().position(|&(row, _)| row == world)?;
            axis[output] = input;
            reversed[output] = source[input].1 != negative;
        }
        Some(Self { axis, reversed })
    }

    /// The relabeling whose output tensor axis `o` reads input tensor axis
    /// `axis[o]`, walked from its last index when `reversed[o]`.
    ///
    /// Returns `None` unless `axis` is a permutation of `[0, 1, 2]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ritk_image::SignedAxisPermutation;
    /// use ritk_spatial::Direction;
    ///
    /// let identity = Direction::<3>::identity();
    /// assert_eq!(
    ///     SignedAxisPermutation::from_axes([0, 1, 2], [false; 3]),
    ///     SignedAxisPermutation::between(&identity, &identity),
    /// );
    /// // Two output axes cannot read one input axis.
    /// assert_eq!(SignedAxisPermutation::from_axes([0, 0, 1], [false; 3]), None);
    /// ```
    pub fn from_axes(axis: [usize; 3], reversed: [bool; 3]) -> Option<Self> {
        let mut seen = [false; 3];
        for &input in &axis {
            let slot = seen.get_mut(input)?;
            if *slot {
                return None;
            }
            *slot = true;
        }
        Some(Self { axis, reversed })
    }

    /// Whether every output axis reads the same input axis forward.
    fn is_identity(&self) -> bool {
        self.axis == [0, 1, 2] && self.reversed == [false; 3]
    }

    /// Re-express a grid under this relabeling.
    ///
    /// `values` is the flat `[z, y, x]` buffer of extents `dims`; `origin`,
    /// `spacing` and `direction` are the input geometry, which need not be a
    /// signed permutation. The output origin is the world point of the input
    /// voxel that becomes output `(0, 0, 0)`, evaluated as
    /// `origin + D · (S · index)` in the same operation order as
    /// [`ritk_spatial::CartesianGridGeometry::point`].
    ///
    /// # Panics
    ///
    /// Panics if `values.len()` is not the product of `dims`, or if no extent
    /// is zero and computing that product overflows `usize`. A grid with a
    /// zero extent is empty, so its product is zero whatever its other
    /// extents are.
    #[track_caller]
    pub fn reexpress<'a, T: Clone>(
        &self,
        values: &'a [T],
        dims: [usize; 3],
        origin: &Point<3>,
        spacing: &Spacing<3>,
        direction: &Direction<3>,
    ) -> PermutedGrid<'a, T> {
        assert_eq!(
            voxel_count(dims),
            Some(values.len()),
            "voxel count must match the grid extents"
        );
        let out_dims = self.axis.map(|input| dims[input]);
        let out_spacing = Spacing::new(self.axis.map(|input| spacing[input]));

        let mut corner = [0usize; 3];
        for (output, &input) in self.axis.iter().enumerate() {
            if self.reversed[output] {
                corner[input] = dims[input].saturating_sub(1);
            }
        }
        let mut scaled = Vector::zeros();
        for (input, &index) in corner.iter().enumerate() {
            scaled[input] = <f64 as eunomia::FloatElement>::from_count(index) * spacing[input];
        }
        let out_origin = *origin + *direction * scaled;

        let values = if self.is_identity() {
            Cow::Borrowed(values)
        } else {
            Cow::Owned(self.reindex(values, dims, out_dims))
        };
        PermutedGrid {
            values,
            dims: out_dims,
            origin: out_origin,
            spacing: out_spacing,
        }
    }

    /// Gather `values` into output order, walking the output buffer once.
    fn reindex<T: Clone>(&self, values: &[T], dims: [usize; 3], out_dims: [usize; 3]) -> Vec<T> {
        if values.is_empty() {
            // Past here every extent is at least 1, so `extent - 1` cannot underflow.
            return Vec::new();
        }
        let stride = [dims[1] * dims[2], dims[2], 1];
        // Input offset of output index (0, 0, 0) and the signed input step per
        // output axis.
        let mut start = 0usize;
        let mut forward = [0usize; 3];
        for (output, &input) in self.axis.iter().enumerate() {
            forward[output] = stride[input];
            if self.reversed[output] {
                start += (dims[input] - 1) * stride[input];
            }
        }
        let offset = |output: usize, index: usize, base: usize| {
            if self.reversed[output] {
                base - index * forward[output]
            } else {
                base + index * forward[output]
            }
        };
        let mut out = Vec::with_capacity(values.len());
        for z in 0..out_dims[0] {
            let plane = offset(0, z, start);
            for y in 0..out_dims[1] {
                let row = offset(1, y, plane);
                for x in 0..out_dims[2] {
                    out.push(values[offset(2, x, row)].clone());
                }
            }
        }
        out
    }
}

#[cfg(test)]
#[path = "tests_axis_permutation.rs"]
mod tests;
