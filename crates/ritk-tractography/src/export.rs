use crate::types::TractographyResult;

impl TractographyResult {
    /// Export streamlines as a [`ritk_trk::TrkTractogram`] suitable for
    /// writing to a DSI Studio / TrackVis `.trk` file.
    ///
    /// Tractography points are already in physical millimetre coordinates,
    /// so the header uses an **identity** affine.  Callers that need a
    /// voxel‑to‑RAS mapping should set `dim` and `voxel_size` to match the
    /// reference image and optionally replace
    /// [`ritk_trk::TrkHeader::vox_to_ras`]
    /// on the returned tractogram.
    ///
    /// Per‑point scalars and per‑streamline properties are not populated.
    #[must_use]
    pub fn to_trk(&self, dim: [i16; 3], voxel_size: [f32; 3]) -> ritk_trk::TrkTractogram {
        self.to_trk_header(dim, voxel_size, None)
    }

    /// Export streamlines as a [`ritk_trk::TrkTractogram`] with a custom
    /// `vox_to_ras` affine.
    ///
    /// When `vox_to_ras` is `None` the identity affine is used (same as
    /// [`Self::to_trk`]).  This is useful when the reference image has a
    /// non‑trivial voxel‑to‑world mapping — the affine is embedded in
    /// the `.trk` header so DSI Studio can display streamlines in the
    /// correct anatomical space.
    ///
    /// Per‑point scalars are not populated; use
    /// [`Self::to_trk_with_scalars`] and replace the header affine on the
    /// returned tractogram, or combine `to_trk_header` with manual
    /// scalar construction.
    #[must_use]
    pub fn to_trk_header(
        &self,
        dim: [i16; 3],
        voxel_size: [f32; 3],
        vox_to_ras: Option<[[f32; 4]; 4]>,
    ) -> ritk_trk::TrkTractogram {
        let count = self.streamlines.len() as i32;
        let mut header = ritk_trk::TrkHeader {
            dim,
            voxel_size,
            n_count: count,
            ..Default::default()
        };
        if let Some(affine) = vox_to_ras {
            header.vox_to_ras = affine;
        }
        let streamlines: Vec<gaia::Polyline<f64>> = self
            .streamlines
            .iter()
            .map(|s| s.geometry().clone())
            .collect();
        ritk_trk::TrkTractogram {
            header,
            streamlines,
            scalars: Vec::new(),
            properties: Vec::new(),
        }
    }

    /// Export streamlines as a [`ritk_trk::TrkTractogram`] with per‑point
    /// scalar values (e.g. FA, MD) for DSI Studio colour‑coding.
    ///
    /// `scalar_names` sets the null‑terminated, space‑separated
    /// `scalar_name` field in the header (e.g. `"FA MD"`).
    /// `scalars` must have the same length as the number of streamlines;
    /// each inner `Box<[f32]>` must contain `n_points × n_scalars` values
    /// in row‑major order (per‑point scalar stride).
    ///
    /// # Panics
    ///
    /// Panics in debug when `scalars.len() != streamlines_generated()` or
    /// any inner scalar slice has the wrong length for its streamline.
    #[must_use]
    pub fn to_trk_with_scalars(
        &self,
        dim: [i16; 3],
        voxel_size: [f32; 3],
        scalar_names: &[&str],
        scalars: Vec<Box<[f32]>>,
    ) -> ritk_trk::TrkTractogram {
        let n_scalars = scalar_names.len();

        if n_scalars > 0 {
            debug_assert_eq!(
                scalars.len(),
                self.streamlines.len(),
                "scalars vec length must match streamline count"
            );

            debug_assert!(
                scalars
                    .iter()
                    .zip(self.streamlines.iter())
                    .all(|(s, streamline)| s.len() == streamline.geometry().len() * n_scalars)
            );
        }

        let mut trk = self.to_trk_header(dim, voxel_size, None);
        trk.header.n_scalars = n_scalars as i16;

        let mut scalar_name = [0u8; 200];
        let joined = scalar_names.join(" ");
        let name_bytes = joined.as_bytes();
        let copy_len = name_bytes.len().min(199);
        scalar_name[..copy_len].copy_from_slice(&name_bytes[..copy_len]);
        trk.header.scalar_name = scalar_name;
        trk.scalars = scalars;

        trk
    }

    /// Export streamlines as a [`ritk_tck::TckTractogram`] suitable for
    /// writing to an MRtrix3 `.tck` file.
    ///
    /// Tractography points are already in scanner-space millimetre
    /// coordinates, which is the native `.tck` coordinate system — no
    /// affine conversion is needed.  The header uses
    /// [`ritk_tck::TckDatatype::Float32LE`]
    /// by default.
    ///
    /// Per‑point scalars and per‑streamline properties are not stored
    /// natively in the `.tck` format.
    #[must_use]
    pub fn to_tck(&self) -> ritk_tck::TckTractogram {
        self.to_tck_header(None, None, None)
    }

    /// Export streamlines as a [`ritk_tck::TckTractogram`] with custom
    /// header fields.
    ///
    /// All three parameters are optional; `None` leaves the field unset
    /// in the header.  This is useful for embedding provenance (version
    /// and comments) or attaching a voxel‑to‑scanner transform for
    /// downstream tools that need it.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let tck = result.to_tck_header(
    ///     Some("3.0.4".into()),
    ///     Some("RITK Euler tractography".into()),
    ///     Some(transform_matrix),
    /// );
    /// ```
    #[must_use]
    pub fn to_tck_header(
        &self,
        mrtrix_version: Option<String>,
        comments: Option<String>,
        transform: Option<[[f64; 4]; 4]>,
    ) -> ritk_tck::TckTractogram {
        let mut header = ritk_tck::TckHeader::default();
        if let Some(v) = mrtrix_version {
            header.mrtrix_version = Some(v);
        }
        if let Some(c) = comments {
            header.comments = Some(c);
        }
        if let Some(t) = transform {
            header.transform = Some(t);
        }
        let streamlines: Vec<gaia::Polyline<f64>> = self
            .streamlines
            .iter()
            .map(|s| s.geometry().clone())
            .collect();
        ritk_tck::TckTractogram {
            header,
            streamlines,
        }
    }

    /// Export streamlines as a [`ritk_trx::TrxTractogram`] suitable for
    /// writing to a TRX (Tractography Reference eXchange) file.
    ///
    /// Tractography points are already in physical millimetre coordinates,
    /// which is the native TRX coordinate system.  The header uses `"float32"`
    /// dtype by default.
    ///
    /// Per-vertex and per-streamline data arrays are not populated.
    #[must_use]
    pub fn to_trx(&self) -> ritk_trx::TrxTractogram {
        self.to_trx_with_dpv(std::collections::HashMap::new())
    }

    /// Export streamlines as a [`ritk_trx::TrxTractogram`] with
    /// per‑vertex data arrays (DPV) such as FA and MD.
    ///
    /// `dpv_data` keys must match entries the caller adds to the returned
    /// header's `dpv` map.  Each value is the raw encoded byte buffer
    /// for the corresponding array; the dtype and `n_components` are
    /// declared in the [`ritk_trx::TrxHeader::dpv`] entry.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use std::collections::HashMap;
    /// let mut trx = result.to_trx_with_dpv(HashMap::new());
    /// trx.header.dpv.insert(
    ///     "FA".into(),
    ///     ritk_trx::TrxArrayDef { dtype: "float32".into(), n_components: 1 },
    /// );
    /// trx.dpv_data.insert("FA".into(), fa_bytes);
    /// ```
    #[must_use]
    pub fn to_trx_with_dpv(
        &self,
        dpv_data: std::collections::HashMap<String, Vec<u8>>,
    ) -> ritk_trx::TrxTractogram {
        let header = ritk_trx::TrxHeader::default();
        let streamlines: Vec<gaia::Polyline<f64>> = self
            .streamlines
            .iter()
            .map(|s| s.geometry().clone())
            .collect();
        ritk_trx::TrxTractogram {
            header,
            streamlines,
            dpv_data,
        }
    }
}
