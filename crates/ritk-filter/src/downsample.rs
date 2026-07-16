use ritk_image::native::Image;

/// Downsample filter.
///
/// Reduces the image size by integer factors by keeping every Nth pixel.
/// Updates spacing to reflect the new resolution.
pub struct DownsampleFilter {
    factors: Vec<usize>,
}

impl DownsampleFilter {
    /// Create a new downsample filter.
    ///
    /// # Arguments
    /// * `factors` - Downsampling factor for each dimension (must be >= 1).
    pub fn new(factors: Vec<usize>) -> Self {
        Self { factors }
    }

    /// Apply the filter to a Coeus-native image.
    pub fn apply<B>(
        &self,
        image: &Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let dims = image.shape();
        let mut data = image.data_slice()?.to_vec();
        let mut spacing = *image.spacing();

        for d in 0..3 {
            let factor = if d < self.factors.len() {
                self.factors[d]
            } else {
                self.factors[0]
            };

            if factor <= 1 {
                continue;
            }

            data = downsample_3d_axis(&data, dims, d, factor);
            spacing[d] *= factor as f64;
        }

        Image::from_flat_on(
            data,
            dims,
            *image.origin(),
            spacing,
            *image.direction(),
            backend,
        )
    }
}

/// Downsample a flat z-major `[nz, ny, nx]` volume along `axis` by keeping
/// every `factor`-th sample starting at index 0.
fn downsample_3d_axis(data: &[f32], dims: [usize; 3], axis: usize, factor: usize) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let new_dims = [
        if axis == 0 { (nz + factor - 1) / factor } else { nz },
        if axis == 1 { (ny + factor - 1) / factor } else { ny },
        if axis == 2 { (nx + factor - 1) / factor } else { nx },
    ];
    let [nz2, ny2, nx2] = new_dims;
    let mut out = Vec::with_capacity(nz2 * ny2 * nx2);

    for iz in 0..nz2 {
        for iy in 0..ny2 {
            for ix in 0..nx2 {
                let src_iz = if axis == 0 { iz * factor } else { iz };
                let src_iy = if axis == 1 { iy * factor } else { iy };
                let src_ix = if axis == 2 { ix * factor } else { ix };
                let src_idx = src_iz * ny * nx + src_iy * nx + src_ix;
                out.push(data[src_idx]);
            }
        }
    }

    out
}

#[cfg(test)]
#[path = "tests_downsample.rs"]
mod tests;
