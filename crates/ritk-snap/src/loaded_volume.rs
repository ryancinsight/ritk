use aequitas::systems::si::{quantities::Length, units::Millimeter};
use arrayvec::ArrayString;
use ritk_spatial::Vector;

pub(crate) const READER_ORIENTATION_DIRECTION_TOLERANCE: f64 = 1e-3;
const READER_NONUNIFORM_SPACING_THRESHOLD: f64 = 0.01;

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
    pub patient_weight_kg: Option<aequitas::systems::si::quantities::Mass>,
    /// Injected radionuclide total dose in Bq (for PET SUV computation).
    pub injected_dose_bq: Option<ritk_io::Radioactivity>,
    /// Radionuclide physical half-life in seconds (for PET SUV computation).
    pub radionuclide_half_life_s: Option<aequitas::systems::si::quantities::Time>,
    /// Radiopharmaceutical start time (TM VR, max 16 chars).
    pub radiopharmaceutical_start_time: Option<ArrayString<16>>,
    /// Pixel decay-correction mode from (0054,1102) (CS VR, max 16 chars).
    pub decay_correction: Option<ArrayString<16>>,
}

impl LoadedVolume {
    /// Replace the active PET reconstruction and its SUV and geometry together.
    pub fn select_enhanced_pet_volume(&mut self, index: usize) -> Option<()> {
        let metadata = self.metadata.as_deref_mut()?;
        let volume = metadata.enhanced_pet_selection.as_mut()?.select(index)?;
        self.data = std::sync::Arc::clone(&volume.data);
        self.shape = volume.shape;
        self.origin = volume.origin.to_array();
        self.spacing = volume.spacing.to_array();
        self.direction = volume.direction.to_row_major();
        metadata.dimensions = [self.shape[1], self.shape[2], self.shape[0]];
        metadata.origin = self.origin.map(Length::from_unit::<Millimeter>);
        metadata.spacing = self.spacing.map(Length::from_unit::<Millimeter>);
        metadata.direction = volume.direction.to_column_major();
        metadata.enhanced_pet_suv = volume.suv_bw.clone();
        Some(())
    }

    /// Label a reconstructed plane on the original DICOM stack axis.
    #[must_use]
    pub fn enhanced_pet_slice_label(&self, axis: usize, slice: usize) -> Option<&'static str> {
        use ritk_io::SliceProvenance;

        if axis != 0 {
            return None;
        }
        let selection = self.metadata.as_ref()?.enhanced_pet_selection.as_ref()?;
        let volume = selection.volumes().get(selection.selected_index())?;
        match volume.slice_provenance.as_ref()?.get(slice)? {
            SliceProvenance::Acquired => None,
            SliceProvenance::Interpolated => Some("Interpolated PET slice"),
            SliceProvenance::Copied => Some("Copied PET slice"),
        }
    }

    /// Read an Enhanced PET quantitative voxel in grams per millilitre.
    #[must_use]
    pub fn enhanced_pet_suv_at(&self, [depth, row, col]: [usize; 3]) -> Option<f64> {
        let [planes, rows, cols] = self.shape;
        if depth >= planes || row >= rows || col >= cols {
            return None;
        }
        let index = depth
            .checked_mul(rows)?
            .checked_add(row)?
            .checked_mul(cols)?
            .checked_add(col)?;
        self.metadata
            .as_ref()?
            .enhanced_pet_suv
            .as_ref()?
            .at(index)
            .map(|value| value.in_unit::<aequitas::systems::si::units::GramPerCubicCentimeter>())
    }

    /// Returns source-qualified DICOM in-plane spacing for measurement admission.
    ///
    /// Every retained DICOM slice must supply the same PixelSpacing; absence or
    /// conflict returns `None` instead of using display fallback geometry.
    /// Non-DICOM volumes use their format geometry.
    pub(crate) fn measurement_pixel_spacing(&self) -> Option<[f64; 2]> {
        let Some(metadata) = self.metadata.as_deref() else {
            return Some([self.spacing[1], self.spacing[2]]);
        };
        let mut slices = metadata.slices.iter();
        let spacing = slices.next()?.pixel_spacing?;
        slices
            .all(|slice| slice.pixel_spacing == Some(spacing))
            .then(|| spacing.map(|length| length.in_unit::<Millimeter>()))
    }

    pub(crate) fn measurement_spacing(&self) -> Option<([f64; 2], Option<f64>)> {
        let slice = self
            .metadata
            .is_none()
            .then_some(self.spacing[0])
            .or_else(|| self.measurement_slice_spacing());
        Some((self.measurement_pixel_spacing()?, slice))
    }

    /// Returns the source-qualified distance between adjacent slice centres.
    ///
    /// Slice bases and step directions follow the reader's 1e-3 Euclidean
    /// direction-admission policy. This inherited approximation limit is not a
    /// floating-point error bound. Projected gaps use the reader's median and 1%
    /// maximum-relative-deviation policy; display spacing is never a fallback.
    pub(crate) fn measurement_slice_spacing(&self) -> Option<f64> {
        let metadata = self.metadata.as_deref()?;
        let basis = |orientation: [f64; 6]| {
            let column =
                Vector::new([orientation[0], orientation[1], orientation[2]]).normalized()?;
            let row = Vector::new([orientation[3], orientation[4], orientation[5]]).normalized()?;
            Some((column, row, column.cross(&row).normalized()?))
        };
        let (column, row, normal) = basis(metadata.slices.first()?.image_orientation_patient?)?;
        let positions = metadata
            .slices
            .iter()
            .map(|slice| {
                let (candidate_column, candidate_row, _) = basis(slice.image_orientation_patient?)?;
                let position = Vector::new(
                    slice
                        .image_position_patient?
                        .map(|length| length.in_unit::<Millimeter>()),
                );
                ((candidate_column - column).norm() <= READER_ORIENTATION_DIRECTION_TOLERANCE
                    && (candidate_row - row).norm() <= READER_ORIENTATION_DIRECTION_TOLERANCE)
                    .then_some(position)
            })
            .collect::<Option<Vec<_>>>()?;
        let mut distances = positions
            .windows(2)
            .map(|pair| {
                let step = pair[1] - pair[0];
                let projection = normal.dot(&step);
                let expected = if projection.is_sign_negative() {
                    -normal
                } else {
                    normal
                };
                ((step.normalized()? - expected).norm() <= READER_ORIENTATION_DIRECTION_TOLERANCE)
                    .then_some(projection.abs())
            })
            .collect::<Option<Vec<_>>>()?;
        distances.sort_by(f64::total_cmp);
        let middle = distances.len() / 2;
        let (lower, upper) = distances.split_at(middle);
        let spacing = if distances.len().is_multiple_of(2) {
            lower.last()?.midpoint(*upper.first()?)
        } else {
            *upper.first()?
        };
        (spacing.is_finite()
            && spacing > 0.0
            && distances.iter().all(|distance| {
                ((distance - spacing) / spacing).abs() <= READER_NONUNIFORM_SPACING_THRESHOLD
            }))
        .then_some(spacing)
    }

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
