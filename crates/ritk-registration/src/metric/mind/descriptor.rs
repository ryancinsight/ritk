//! Six-neighbour self-similarity context descriptor construction and packing.

use eunomia::CastFrom;

use super::config::DescriptorGeometry;
use super::MindSscError;

pub(super) const DESCRIPTOR_COMPONENTS: usize = 12;
pub(super) const BITS_PER_COMPONENT: usize = 5;
pub(super) const DESCRIPTOR_BITS: usize = DESCRIPTOR_COMPONENTS * BITS_PER_COMPONENT;

const PAIRS: [(usize, usize); DESCRIPTOR_COMPONENTS] = [
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (2, 4),
    (2, 5),
    (3, 4),
    (3, 5),
];

pub(super) fn descriptor_at(
    center: [usize; 3],
    geometry: DescriptorGeometry,
    mut sample: impl FnMut([usize; 3]) -> Result<f32, MindSscError>,
) -> Result<u64, MindSscError> {
    let radius = geometry.patch_radius().map(|value| {
        isize::try_from(value).expect("invariant: MindSscConfig validates signed patch radii")
    });
    let dilation = geometry.neighbour_dilation().map(|value| {
        isize::try_from(value).expect("invariant: MindSscConfig validates signed dilation")
    });
    let neighbours = [
        [-dilation[0], 0, 0],
        [dilation[0], 0, 0],
        [0, -dilation[1], 0],
        [0, dilation[1], 0],
        [0, 0, -dilation[2]],
        [0, 0, dilation[2]],
    ];
    let mut distances = [0.0_f32; DESCRIPTOR_COMPONENTS];

    for dz in -radius[0]..=radius[0] {
        for dy in -radius[1]..=radius[1] {
            for dx in -radius[2]..=radius[2] {
                let patch = [dz, dy, dx];
                let mut values = [0.0_f32; 6];
                for (neighbour, value) in neighbours.iter().zip(values.iter_mut()) {
                    let mut support = [0_usize; 3];
                    for axis in 0..3 {
                        let displacement = neighbour[axis]
                            .checked_add(patch[axis])
                            .ok_or(MindSscError::IndexOverflow)?;
                        support[axis] = center[axis]
                            .checked_add_signed(displacement)
                            .ok_or(MindSscError::IndexOverflow)?;
                    }
                    *value = sample(support)?;
                }
                for (distance, &(left, right)) in distances.iter_mut().zip(PAIRS.iter()) {
                    let difference = values[left] - values[right];
                    *distance = difference.mul_add(difference, *distance);
                }
            }
        }
    }
    pack_distances(distances)
}

pub(super) fn pack_distances(distances: [f32; DESCRIPTOR_COMPONENTS]) -> Result<u64, MindSscError> {
    if distances.iter().any(|distance| !distance.is_finite()) {
        return Err(MindSscError::NonFinitePatchDistance);
    }
    let minimum = distances
        .iter()
        .copied()
        .min_by(f32::total_cmp)
        .expect("invariant: SSC has twelve distances");
    let variance = distances.iter().sum::<f32>() / 12.0;
    if !variance.is_finite() {
        return Err(MindSscError::NonFinitePatchDistance);
    }
    if variance == 0.0 {
        return Ok((1_u64 << DESCRIPTOR_BITS) - 1);
    }

    let mut packed = 0_u64;
    for (component, distance) in distances.into_iter().enumerate() {
        let response = (-(distance - minimum) / variance).exp().clamp(0.0, 1.0);
        let level = u32::cast_from((response * 5.0).round());
        let unary = (1_u64 << level) - 1;
        packed |= unary << (component * BITS_PER_COMPONENT);
    }
    Ok(packed)
}

#[cfg(test)]
pub(super) fn quantized_levels(packed: u64) -> [u32; DESCRIPTOR_COMPONENTS] {
    std::array::from_fn(|component| {
        let mask = (1_u64 << BITS_PER_COMPONENT) - 1;
        ((packed >> (component * BITS_PER_COMPONENT)) & mask).count_ones()
    })
}
