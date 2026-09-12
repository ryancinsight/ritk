//! GPU volume-buffer size validation.

use std::mem::size_of;

/// Return the voxel count and byte size required for a scalar volume buffer.
pub(super) fn volume_storage_size(shape: [usize; 3]) -> Option<(usize, u64)> {
    let voxel_count = shape.into_iter().try_fold(1usize, usize::checked_mul)?;
    let byte_count = u64::try_from(voxel_count)
        .ok()?
        .checked_mul(u64::try_from(size_of::<f32>()).ok()?)?;
    Some((voxel_count, byte_count))
}

/// Return whether a scalar volume fits the device's storage-buffer limits.
pub(super) fn volume_fits_device(shape: [usize; 3], limits: &wgpu::Limits) -> bool {
    let Some((_, byte_count)) = volume_storage_size(shape) else {
        return false;
    };
    byte_count <= limits.max_buffer_size
        && byte_count <= u64::from(limits.max_storage_buffer_binding_size)
}

#[cfg(test)]
mod tests {
    use super::{volume_fits_device, volume_storage_size};

    #[test]
    fn volume_storage_size_matches_public_ct_dimensions() {
        assert_eq!(
            volume_storage_size([409, 512, 512]),
            Some((107_216_896, 428_867_584))
        );
    }

    #[test]
    fn volume_storage_size_rejects_overflow() {
        assert_eq!(volume_storage_size([usize::MAX, 2, 2]), None);
    }

    #[test]
    fn volume_fits_device_rejects_storage_limit() {
        let limits = wgpu::Limits::default();
        assert!(!volume_fits_device([409, 512, 512], &limits));
        assert!(volume_fits_device([8, 8, 8], &limits));
    }
}
