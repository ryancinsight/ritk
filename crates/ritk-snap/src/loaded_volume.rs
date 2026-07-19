use arrayvec::ArrayString;

/// Type-erased loaded volume for viewer use (avoids propagating `<B: Backend>` through UI).
///
/// Data is stored in row-major `[depth, rows, cols, channels]` order with
/// interleaved channel samples. Voxel values are in HU for CT or relative
/// intensity for other modalities.
#[derive(Debug, Clone)]
pub struct LoadedVolume {
    /// Pixel data in row-major [depth, rows, cols, channels] order, f32.
    pub data: std::sync::Arc<Vec<f32>>,
    /// Image shape [depth, rows, cols] (spatial dimensions only).
    pub shape: [usize; 3],
    /// Number of interleaved channels per voxel (1 = scalar, 3 = RGB).
    pub channels: u8,
    /// Voxel spacing [dz, dy, dx] in mm/pixel.
    pub spacing: [f64; 3],
    /// Image origin in physical space.
    pub origin: [f64; 3],
    /// Direction cosine matrix (row-major 3x3 flattened).
    pub direction: [f64; 9],
    /// Optional DICOM metadata.
    pub metadata: Option<Box<ritk_io::DicomReadMetadata>>,
    /// Source path.
    pub source: Option<std::path::PathBuf>,
    /// DICOM modality string (CS VR, max 16 chars).
    pub modality: Option<ArrayString<16>>,
    /// Patient name from metadata.
    pub patient_name: Option<String>,
    /// Patient ID from metadata.
    pub patient_id: Option<String>,
    /// Study date from metadata (DA VR, 8 chars).
    pub study_date: Option<ArrayString<8>>,
    /// Series description from metadata.
    pub series_description: Option<String>,
    /// Series time from metadata (TM VR, max 16 chars).
    pub series_time: Option<ArrayString<16>>,
    /// Patient weight in kg (for PET SUV computation).
    pub patient_weight_kg: Option<f64>,
    /// Injected radionuclide total dose in Bq (for PET SUV computation).
    pub injected_dose_bq: Option<f64>,
    /// Radionuclide physical half-life in seconds (for PET SUV computation).
    pub radionuclide_half_life_s: Option<f64>,
    /// Radiopharmaceutical start time (TM VR, max 16 chars).
    pub radiopharmaceutical_start_time: Option<ArrayString<16>>,
    /// Pixel decay-correction mode from (0054,1102) (CS VR, max 16 chars).
    pub decay_correction: Option<ArrayString<16>>,
}

impl LoadedVolume {
    /// Get the first channel value at voxel position (d, r, c).
    ///
    /// For multi-channel volumes (e.g. RGB), returns the first channel.
    /// Returns `0.0` when any index exceeds the corresponding dimension bound.
    pub fn pixel_at(&self, d: usize, r: usize, c: usize) -> f32 {
        let [depth, rows, cols] = self.shape;
        let ch = self.channels as usize;
        if d >= depth || r >= rows || c >= cols {
            return 0.0;
        }
        self.data[((d * rows + r) * cols + c) * ch]
    }

    /// Get all channel values at voxel position (d, r, c).
    ///
    /// Returns a slice of length `self.channels`. For scalar volumes this
    /// is a single-element slice.
    pub fn pixel_channels(&self, d: usize, r: usize, c: usize) -> &[f32] {
        let [depth, rows, cols] = self.shape;
        let ch = self.channels as usize;
        if d >= depth || r >= rows || c >= cols {
            return &[];
        }
        let base = ((d * rows + r) * cols + c) * ch;
        &self.data[base..base + ch]
    }

    /// Extract a 2D slice as a flat `Vec<f32>` in row-major order.
    ///
    /// For multi-channel volumes, extracts only the first channel.
    ///
    /// # Axis semantics
    /// - `axis = 0` â€” axial (fixed depth index `d`): output shape `[rows, cols]`,
    ///   returns `(pixels, cols, rows)`.
    /// - `axis = 1` â€” coronal (fixed row index `r`): output shape `[depth, cols]`,
    ///   returns `(pixels, cols, depth)`.
    /// - `axis = 2` â€” sagittal (fixed column index `c`): output shape `[depth, rows]`,
    ///   returns `(pixels, rows, depth)`.
    ///
    /// An out-of-range `index` is silently clamped to the last valid position.
    /// An unknown `axis` returns an empty result `(vec![], 0, 0)`.
    pub fn extract_slice(&self, axis: usize, index: usize) -> (Vec<f32>, usize, usize) {
        let [depth, rows, cols] = self.shape;
        let ch = self.channels as usize;
        match axis {
            0 => {
                // Axial: fixed d, contiguous rowsÃ—colsÃ—channels slice;
                // take only the first channel of each voxel.
                let d = index.min(depth.saturating_sub(1));
                let stride = rows * cols * ch;
                let offset = d * stride;
                let mut pixels = Vec::with_capacity(rows * cols);
                for voxel in 0..rows * cols {
                    pixels.push(self.data[offset + voxel * ch]);
                }
                (pixels, cols, rows)
            }
            1 => {
                // Coronal: fixed r, strided per-depth access.
                let r = index.min(rows.saturating_sub(1));
                let mut pixels = Vec::with_capacity(depth * cols);
                for d in 0..depth {
                    let base = d * rows * cols * ch + r * cols * ch;
                    for c in 0..cols {
                        pixels.push(self.data[base + c * ch]);
                    }
                }
                (pixels, cols, depth)
            }
            2 => {
                // Sagittal: fixed c, strided access.
                let c = index.min(cols.saturating_sub(1));
                let mut pixels = Vec::with_capacity(depth * rows);
                for d in 0..depth {
                    for r in 0..rows {
                        pixels.push(self.data[(d * rows + r) * cols * ch + c * ch]);
                    }
                }
                (pixels, rows, depth)
            }
            _ => (vec![], 0, 0),
        }
    }

    /// Extract a 2-D slice into a pre-allocated buffer, returning `(width, height)`.
    ///
    /// This is the zero-allocation variant of [`Self::extract_slice`]: the caller
    /// supplies `out`, which is resized (never shrunk in capacity) to exactly the
    /// number of pixels in the slice before being filled. Capacity is reused when
    /// `out` already has sufficient capacity, eliminating the per-call heap alloc
    /// that `extract_slice` incurs.
    ///
    /// # Axis semantics
    ///
    /// Identical to [`Self::extract_slice`]:
    /// - `axis = 0` â€” axial (fixed `d`): returns `(cols, rows)`.
    /// - `axis = 1` â€” coronal (fixed `r`): returns `(cols, depth)`.
    /// - `axis = 2` â€” sagittal (fixed `c`): returns `(rows, depth)`.
    ///
    /// Unknown axes clear `out` and return `(0, 0)`.
    pub fn extract_slice_into(
        &self,
        out: &mut Vec<f32>,
        axis: usize,
        index: usize,
    ) -> (usize, usize) {
        let [depth, rows, cols] = self.shape;
        let ch = self.channels as usize;
        match axis {
            0 => {
                let d = index.min(depth.saturating_sub(1));
                let n = rows * cols;
                let stride = rows * cols * ch;
                let offset = d * stride;
                out.resize(n, 0.0);
                for (i, dst) in out.iter_mut().enumerate() {
                    *dst = self.data[offset + i * ch];
                }
                (cols, rows)
            }
            1 => {
                let r = index.min(rows.saturating_sub(1));
                let n = depth * cols;
                out.resize(n, 0.0);
                let mut pos = 0;
                for d in 0..depth {
                    let base = d * rows * cols * ch + r * cols * ch;
                    for c in 0..cols {
                        out[pos] = self.data[base + c * ch];
                        pos += 1;
                    }
                }
                (cols, depth)
            }
            2 => {
                let c = index.min(cols.saturating_sub(1));
                let n = depth * rows;
                out.resize(n, 0.0);
                let mut pos = 0;
                for d in 0..depth {
                    for r in 0..rows {
                        out[pos] = self.data[(d * rows + r) * cols * ch + c * ch];
                        pos += 1;
                    }
                }
                (rows, depth)
            }
            _ => {
                out.clear();
                (0, 0)
            }
        }
    }
}
