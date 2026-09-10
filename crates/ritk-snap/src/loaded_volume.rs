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
    /// - `axis = 0` — axial (fixed depth index `d`): output shape `[rows, cols]`,
    ///   returns `(pixels, cols, rows)`.
    /// - `axis = 1` — coronal (fixed row index `r`): output shape `[depth, cols]`,
    ///   returns `(pixels, cols, depth)`.
    /// - `axis = 2` — sagittal (fixed column index `c`): output shape `[depth, rows]`,
    ///   returns `(pixels, rows, depth)`.
    ///
    /// An out-of-range `index` is silently clamped to the last valid position.
    /// An unknown `axis` returns an empty result `(vec![], 0, 0)`.
    pub fn extract_slice(&self, axis: usize, index: usize) -> (Vec<f32>, usize, usize) {
        if self.channels == 0 {
            return (Vec::new(), 0, 0);
        }
        let (width, height) = self.slice_dimensions(axis);
        let mut pixels = Vec::with_capacity(width * height);
        self.visit_slice_offsets(axis, index, |offset| {
            pixels.push(self.data[offset]);
        });
        (pixels, width, height)
    }

    /// Extract a 2-D slice while retaining every interleaved channel.
    ///
    /// The returned samples use row-major pixel order with channel-fastest
    /// storage: `[pixel_0_channel_0, ..., pixel_0_channel_n, pixel_1_channel_0, ...]`.
    /// RGB DICOM slices therefore retain the decoded red, green, and blue
    /// samples for every axial, coronal, and sagittal view.
    ///
    /// # Axis semantics
    ///
    /// The dimensions and clamping rules are identical to [`Self::extract_slice`].
    pub fn extract_slice_channels(&self, axis: usize, index: usize) -> (Vec<f32>, usize, usize) {
        if self.channels == 0 {
            return (Vec::new(), 0, 0);
        }
        let (width, height) = self.slice_dimensions(axis);
        let channels = usize::from(self.channels);
        let mut samples = Vec::with_capacity(width * height * channels);
        if channels != 0 {
            self.visit_slice_offsets(axis, index, |offset| {
                samples.extend_from_slice(&self.data[offset..offset + channels]);
            });
        }
        (samples, width, height)
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
    /// - `axis = 0` — axial (fixed `d`): returns `(cols, rows)`.
    /// - `axis = 1` — coronal (fixed `r`): returns `(cols, depth)`.
    /// - `axis = 2` — sagittal (fixed `c`): returns `(rows, depth)`.
    ///
    /// Unknown axes clear `out` and return `(0, 0)`.
    pub fn extract_slice_into(
        &self,
        out: &mut mnemosyne::AlignedVec<f32>,
        axis: usize,
        index: usize,
    ) -> (usize, usize) {
        if self.channels == 0 {
            out.clear();
            return (0, 0);
        }
        let (width, height) = self.slice_dimensions(axis);
        out.resize(width * height, 0.0);
        let mut position = 0;
        self.visit_slice_offsets(axis, index, |offset| {
            out[position] = self.data[offset];
            position += 1;
        });
        (width, height)
    }

    /// Extract every channel into a caller-supplied scratch buffer.
    ///
    /// The output layout matches [`Self::extract_slice_channels`]. The buffer
    /// is resized to `width × height × channels`, preserving capacity between
    /// frames so RGB rendering does not allocate in its hot path.
    pub fn extract_slice_channels_into(
        &self,
        out: &mut mnemosyne::AlignedVec<f32>,
        axis: usize,
        index: usize,
    ) -> (usize, usize) {
        if self.channels == 0 {
            out.clear();
            return (0, 0);
        }
        let (width, height) = self.slice_dimensions(axis);
        let channels = usize::from(self.channels);
        out.resize(width * height * channels, 0.0);
        let mut position = 0;
        self.visit_slice_offsets(axis, index, |offset| {
            out[position..position + channels]
                .copy_from_slice(&self.data[offset..offset + channels]);
            position += channels;
        });
        (width, height)
    }

    fn slice_dimensions(&self, axis: usize) -> (usize, usize) {
        match axis {
            0 => (self.shape[2], self.shape[1]),
            1 => (self.shape[2], self.shape[0]),
            2 => (self.shape[1], self.shape[0]),
            _ => (0, 0),
        }
    }

    fn visit_slice_offsets<F>(&self, axis: usize, index: usize, mut visit: F)
    where
        F: FnMut(usize),
    {
        let [depth, rows, cols] = self.shape;
        let channels = usize::from(self.channels);
        match axis {
            0 => {
                let depth_index = index.min(depth.saturating_sub(1));
                let offset = depth_index * rows * cols * channels;
                for voxel in 0..rows * cols {
                    visit(offset + voxel * channels);
                }
            }
            1 => {
                let row = index.min(rows.saturating_sub(1));
                for depth_index in 0..depth {
                    let base = depth_index * rows * cols * channels + row * cols * channels;
                    for column in 0..cols {
                        visit(base + column * channels);
                    }
                }
            }
            2 => {
                let column = index.min(cols.saturating_sub(1));
                for depth_index in 0..depth {
                    for row in 0..rows {
                        visit((depth_index * rows + row) * cols * channels + column * channels);
                    }
                }
            }
            _ => {}
        }
    }
}
