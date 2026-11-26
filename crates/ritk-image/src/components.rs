//! Channel-major component storage for multi-component volumes.
//!
//! [`ComponentBuffers`] owns the per-channel samples of one volume in a single
//! contiguous block, so a consumer reads a component as a slice instead of
//! following one heap block per channel.

use anyhow::{bail, Context, Result};

/// `C` scalar component buffers in one channel-major contiguous block.
///
/// Component `k` occupies `[k·voxels, (k+1)·voxels)`. The channel count is a
/// compile-time constant, so selecting a component needs no offset table, hash
/// or per-channel heap node: the whole payload is one allocation and one slice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComponentBuffers<T, const C: usize> {
    data: Vec<T>,
    voxels: usize,
}

impl<T, const C: usize> ComponentBuffers<T, C> {
    /// Build from `C` owned channel buffers, moving each into one block.
    ///
    /// # Errors
    /// Returns `Err` when `C == 0`, channel lengths differ, or the total length
    /// overflows `usize`.
    pub fn from_channels(channels: [Vec<T>; C]) -> Result<Self> {
        if C == 0 {
            bail!("component buffer channel count must be positive");
        }
        let voxels = channels[0].len();
        let total = voxels
            .checked_mul(C)
            .context("component buffer length overflows usize")?;
        for (k, buf) in channels.iter().enumerate().skip(1) {
            if buf.len() != voxels {
                bail!(
                    "component buffer channel {k} has length {}, expected {voxels}",
                    buf.len()
                );
            }
        }
        let mut data = Vec::with_capacity(total);
        for buf in channels {
            data.extend(buf);
        }
        Ok(Self { data, voxels })
    }

    /// Allocate one default-filled block of `C · voxels` values.
    ///
    /// # Errors
    /// Returns `Err` when the total length overflows `usize`.
    pub fn zeros(voxels: usize) -> Result<Self>
    where
        T: Clone + Default,
    {
        let total = voxels
            .checked_mul(C)
            .context("component buffer length overflows usize")?;
        Ok(Self {
            data: vec![T::default(); total],
            voxels,
        })
    }

    /// Deinterleave `C` channels from a voxel-major sample block.
    ///
    /// The input holds `voxels · C` samples in voxel-major order. The block is
    /// written channel-major, so component `k` occupies
    /// `[k·voxels, (k+1)·voxels)`.
    ///
    /// # Errors
    /// Returns `Err` when the sample count does not match `voxels · C` or the
    /// product overflows `usize`.
    pub fn from_interleaved(interleaved: &[T], voxels: usize) -> Result<Self>
    where
        T: Clone + Default,
    {
        let total = voxels
            .checked_mul(C)
            .context("component buffer length overflows usize")?;
        if interleaved.len() != total {
            bail!(
                "interleaved sample count {} does not match {voxels} voxels with {C} channels",
                interleaved.len()
            );
        }
        let mut buffers = Self::zeros(voxels)?;
        if voxels > 0 {
            for (channel, component) in buffers.data.chunks_exact_mut(voxels).enumerate() {
                for (voxel, value) in component.iter_mut().enumerate() {
                    *value = interleaved[voxel * C + channel].clone();
                }
            }
        }
        Ok(buffers)
    }

    /// Compile-time channel count.
    #[must_use]
    pub const fn channels(&self) -> usize {
        C
    }

    /// Number of voxels in each component.
    #[must_use]
    pub const fn voxels(&self) -> usize {
        self.voxels
    }

    /// Borrow one component's contiguous values.
    ///
    /// # Panics
    /// Panics when `channel >= C`.
    #[must_use]
    pub fn component(&self, channel: usize) -> &[T] {
        assert!(
            channel < C,
            "component channel {channel} out of range for {C}"
        );
        let start = channel * self.voxels;
        &self.data[start..start + self.voxels]
    }

    /// Mutably borrow one component's contiguous values.
    ///
    /// # Panics
    /// Panics when `channel >= C`.
    pub fn component_mut(&mut self, channel: usize) -> &mut [T] {
        assert!(
            channel < C,
            "component channel {channel} out of range for {C}"
        );
        let start = channel * self.voxels;
        &mut self.data[start..start + self.voxels]
    }

    /// Borrow the whole channel-major block.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Take one component's values, dropping the other channels.
    ///
    /// # Panics
    /// Panics when `channel >= C`.
    #[must_use]
    pub fn into_channel(self, channel: usize) -> Vec<T>
    where
        T: Clone,
    {
        assert!(
            channel < C,
            "component channel {channel} out of range for {C}"
        );
        let start = channel * self.voxels;
        self.data[start..start + self.voxels].to_vec()
    }
}
