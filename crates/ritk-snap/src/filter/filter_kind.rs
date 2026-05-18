use crate::filter::serde_helper::BedSeparationConfigSerde;
use ritk_core::filter::BedSeparationConfig;
use serde::{Deserialize, Serialize};

#[doc = include_str!("variant_docs/enum_level.md")]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FilterKind {
    #[doc = include_str!("variant_docs/bed_separation.md")]
    BedSeparation(#[serde(with = "BedSeparationConfigSerde")] BedSeparationConfig),

    #[doc = include_str!("variant_docs/gaussian.md")]
    Gaussian {
        /// Standard deviation in physical units (mm).
        sigma: f32,
    },

    #[doc = include_str!("variant_docs/median.md")]
    Median {
        /// Neighbourhood half-width in voxels.
        radius: usize,
    },

    #[doc = include_str!("variant_docs/clahe.md")]
    Clahe {
        /// `[n_tiles_rows, n_tiles_cols]` per axial slice. Default `[8, 8]`.
        tile_grid_size: [usize; 2],
        /// Clip limit factor (dimensionless). Default 40.0.
        clip_limit: f32,
    },

    #[doc = include_str!("variant_docs/hist_eq.md")]
    HistEq {
        /// Number of histogram bins. Default 256.
        bins: usize,
    },

    #[doc = include_str!("variant_docs/unsharp_mask.md")]
    UnsharpMask {
        /// Gaussian standard deviation in physical units (mm). Broadcast across all dims.
        sigma: f32,
        /// Sharpening strength. Typical range [0.0, 5.0]. ITK default: 0.5.
        amount: f32,
        /// Minimum absolute mask value to trigger sharpening. Default: 0.0.
        threshold: f32,
        /// Whether to clamp output to the input intensity range. Default: true.
        clamp: bool,
    },

    #[doc = include_str!("variant_docs/gradient_aniso_diff.md")]
    GradientAnisotropicDiffusion {
        /// Number of explicit Euler iterations. ITK default: 5.
        iterations: u32,
        /// Time step Δt. Must satisfy Δt ≤ 1/6. ITK default: 0.125.
        time_step: f32,
        /// Conductance K. Larger K → more isotropic smoothing. ITK default: 1.0.
        conductance: f32,
    },

    #[doc = include_str!("variant_docs/connected_components.md")]
    ConnectedComponents {
        /// Use 26-connectivity instead of the default 6-connectivity.
        connectivity_26: bool,
        /// Value designating background pixels. ITK default: 0.0.
        background_value: f32,
    },

    #[doc = include_str!("variant_docs/relabel_components.md")]
    RelabelComponents {
        /// Discard components smaller than this voxel count. Default: 0 (retain all).
        minimum_object_size: u32,
    },

    #[doc = include_str!("variant_docs/multi_otsu.md")]
    MultiOtsuThreshold {
        /// Number of intensity classes to segment into. Must be ≥ 2. ITK default: 3.
        num_classes: u32,
    },

    #[doc = include_str!("variant_docs/binary_erode.md")]
    BinaryErode {
        /// Structuring element half-width in voxels.
        radius: usize,
        /// Voxel intensity treated as foreground. Default: 1.0.
        foreground_value: f32,
    },

    #[doc = include_str!("variant_docs/binary_dilate.md")]
    BinaryDilate {
        /// Structuring element half-width in voxels.
        radius: usize,
        /// Voxel intensity treated as foreground. Default: 1.0.
        foreground_value: f32,
    },

    #[doc = include_str!("variant_docs/binary_closing.md")]
    BinaryClosing {
        /// Structuring element half-width in voxels.
        radius: usize,
        /// Voxel intensity treated as foreground. Default: 1.0.
        foreground_value: f32,
    },

    #[doc = include_str!("variant_docs/binary_opening.md")]
    BinaryOpening {
        /// Structuring element half-width in voxels.
        radius: usize,
        /// Voxel intensity treated as foreground. Default: 1.0.
        foreground_value: f32,
    },

    #[doc = include_str!("variant_docs/binary_fillhole.md")]
    BinaryFillhole {
        /// Voxel intensity treated as foreground. Default: 1.0.
        foreground_value: f32,
    },

    #[doc = include_str!("variant_docs/grayscale_closing.md")]
    GrayscaleClosing {
        /// Structuring element half-width in voxels.
        radius: usize,
    },

    #[doc = include_str!("variant_docs/grayscale_opening.md")]
    GrayscaleOpening {
        /// Structuring element half-width in voxels.
        radius: usize,
    },

    #[doc = include_str!("variant_docs/grayscale_fillhole.md")]
    GrayscaleFillhole,

    #[doc = include_str!("variant_docs/abs.md")]
    Abs,

    #[doc = include_str!("variant_docs/invert_intensity.md")]
    InvertIntensity {
        /// Fixed inversion maximum. `None` → computed from image.
        maximum: Option<f32>,
    },

    #[doc = include_str!("variant_docs/normalize_intensity.md")]
    NormalizeIntensity,

    #[doc = include_str!("variant_docs/square.md")]
    Square,

    #[doc = include_str!("variant_docs/sqrt.md")]
    Sqrt,

    #[doc = include_str!("variant_docs/log.md")]
    Log,

    #[doc = include_str!("variant_docs/exp.md")]
    Exp,

    #[doc = include_str!("variant_docs/morphological_gradient.md")]
    MorphologicalGradient {
        /// Structuring element half-width in voxels.
        radius: usize,
    },

    #[doc = include_str!("variant_docs/distance_transform.md")]
    DistanceTransform {
        /// Intensity threshold separating background from foreground. Default: 0.5.
        threshold: f32,
    },

    #[doc = include_str!("variant_docs/signed_distance_transform.md")]
    SignedDistanceTransform {
        /// Intensity threshold. Default: 0.5.
        threshold: f32,
    },

    /// Flip image along the Z axis (ITK `FlipImageFilter` with axis 0).
    FlipZ,
    /// Flip image along the Y axis (ITK `FlipImageFilter` with axis 1).
    FlipY,
    /// Flip image along the X axis (ITK `FlipImageFilter` with axis 2).
    FlipX,

    #[doc = include_str!("variant_docs/mask_threshold.md")]
    MaskThreshold {
        /// Intensity threshold; voxels ≤ threshold are zeroed.
        threshold: f32,
    },

    #[doc = include_str!("variant_docs/geodesic_dilation_self.md")]
    GeodesicDilationSelf,
    #[doc = include_str!("variant_docs/geodesic_erosion_self.md")]
    GeodesicErosionSelf,

    #[doc = include_str!("variant_docs/shift_scale.md")]
    ShiftScale {
        /// Value added to each voxel before multiplication.
        shift: f32,
        /// Scale factor applied after the shift.
        scale: f32,
    },

    #[doc = include_str!("variant_docs/zero_crossing.md")]
    ZeroCrossing {
        /// Value assigned to zero-crossing voxels.
        foreground_value: f32,
        /// Value assigned to non-crossing voxels.
        background_value: f32,
    },

    /// Crop a rectangular sub-volume (ITK `RegionOfInterestImageFilter`).
    RegionOfInterest {
        /// Start index in Z (slowest axis).
        start_z: usize,
        /// Start index in Y.
        start_y: usize,
        /// Start index in X (fastest axis).
        start_x: usize,
        /// Number of voxels to extract in Z.
        size_z: usize,
        /// Number of voxels to extract in Y.
        size_y: usize,
        /// Number of voxels to extract in X.
        size_x: usize,
    },

    /// Permute axes: `order[i]` = input axis for output axis i
    /// (ITK `PermuteAxesImageFilter`).
    PermuteAxes {
        /// Source axis for output axis 0 (Z).
        order_0: usize,
        /// Source axis for output axis 1 (Y).
        order_1: usize,
        /// Source axis for output axis 2 (X).
        order_2: usize,
    },

    /// Arithmetic mean of (2·radius+1)³ neighbourhood (ITK `MeanImageFilter`).
    Mean {
        /// Neighbourhood half-width in voxels. Default: 1.
        radius: usize,
    },

    /// Extract border voxels of binary objects (ITK `BinaryContourImageFilter`).
    BinaryContour {
        /// Use 26-connectivity; false = 6-connectivity.
        fully_connected: bool,
        /// Foreground voxel value. Default: 1.0.
        foreground_value: f32,
    },

    /// Extract boundaries between label regions (ITK `LabelContourImageFilter`).
    LabelContour {
        /// Use 26-connectivity; false = 6-connectivity.
        fully_connected: bool,
        /// Background voxel value. Default: 0.0.
        background_value: f32,
    },

    /// Cellular automata voting step (ITK `VotingBinaryImageFilter`).
    VotingBinary {
        /// Neighbourhood half-width in voxels.
        radius: usize,
        /// Min foreground neighbours needed for birth (background→foreground).
        birth_threshold: usize,
        /// Min foreground neighbours needed for survival (foreground→foreground).
        survival_threshold: usize,
        /// Foreground voxel value. Default: 1.0.
        foreground_value: f32,
        /// Background voxel value. Default: 0.0.
        background_value: f32,
    },

    /// Integer downsampling by tile averaging (ITK `ShrinkImageFilter`).
    Shrink {
        /// Downsampling factor along Z.
        factor_z: usize,
        /// Downsampling factor along Y.
        factor_y: usize,
        /// Downsampling factor along X.
        factor_x: usize,
    },

    /// Constant-value padding (ITK `ConstantPadImageFilter`).
    ConstantPad {
        pad_lower_z: usize,
        pad_lower_y: usize,
        pad_lower_x: usize,
        pad_upper_z: usize,
        pad_upper_y: usize,
        pad_upper_x: usize,
        /// Fill value. Default: 0.0.
        constant: f32,
    },

    /// Mirror reflection padding (ITK `MirrorPadImageFilter`).
    MirrorPad {
        pad_lower_z: usize,
        pad_lower_y: usize,
        pad_lower_x: usize,
        pad_upper_z: usize,
        pad_upper_y: usize,
        pad_upper_x: usize,
    },

    /// Periodic (wrap) padding (ITK `WrapPadImageFilter`).
    WrapPad {
        pad_lower_z: usize,
        pad_lower_y: usize,
        pad_lower_x: usize,
        pad_upper_z: usize,
        pad_upper_y: usize,
        pad_upper_x: usize,
    },

    #[doc = include_str!("variant_docs/grayscale_erode.md")]
    GrayscaleErode {
        /// Structuring element half-width in voxels. Default: 1.
        radius: usize,
    },

    #[doc = include_str!("variant_docs/grayscale_dilate.md")]
    GrayscaleDilate {
        /// Structuring element half-width in voxels. Default: 1.
        radius: usize,
    },

    #[doc = include_str!("variant_docs/binary_threshold.md")]
    BinaryThreshold {
        /// Inclusive lower bound.
        lower: f32,
        /// Inclusive upper bound.
        upper: f32,
        /// Output value for voxels inside the interval. Default: 1.0.
        foreground: f32,
        /// Output value for voxels outside the interval. Default: 0.0.
        background: f32,
    },

    #[doc = include_str!("variant_docs/rescale_intensity.md")]
    RescaleIntensity {
        /// Minimum output intensity. Default: 0.0.
        out_min: f32,
        /// Maximum output intensity. Default: 1.0.
        out_max: f32,
    },

    #[doc = include_str!("variant_docs/clamp.md")]
    Clamp {
        /// Inclusive lower bound for output. Default: 0.0.
        lower: f32,
        /// Inclusive upper bound for output. Default: 255.0.
        upper: f32,
    },

    #[doc = include_str!("variant_docs/connected_threshold.md")]
    ConnectedThreshold {
        /// Seed voxel depth index (z).
        seed_z: usize,
        /// Seed voxel row index (y).
        seed_y: usize,
        /// Seed voxel column index (x).
        seed_x: usize,
        /// Inclusive lower intensity bound.
        lower: f32,
        /// Inclusive upper intensity bound.
        upper: f32,
    },

    #[doc = include_str!("variant_docs/confidence_connected.md")]
    ConfidenceConnected {
        /// Seed voxel depth index (z).
        seed_z: usize,
        /// Seed voxel row index (y).
        seed_y: usize,
        /// Seed voxel column index (x).
        seed_x: usize,
        /// Initial lower bound (first iteration when σ=0). Default: 0.0.
        initial_lower: f32,
        /// Initial upper bound (first iteration when σ=0). Default: 100.0.
        initial_upper: f32,
        /// k multiplier for k·σ interval. Default: 2.5.
        multiplier: f32,
        /// Maximum iterations. Default: 15.
        max_iterations: u32,
    },

    #[doc = include_str!("variant_docs/neighborhood_connected.md")]
    NeighborhoodConnected {
        /// Seed voxel depth index (z).
        seed_z: usize,
        /// Seed voxel row index (y).
        seed_y: usize,
        /// Seed voxel column index (x).
        seed_x: usize,
        /// Inclusive lower intensity bound.
        lower: f32,
        /// Inclusive upper intensity bound.
        upper: f32,
        /// Neighborhood half-radius z. Default: 1.
        radius_z: usize,
        /// Neighborhood half-radius y. Default: 1.
        radius_y: usize,
        /// Neighborhood half-radius x. Default: 1.
        radius_x: usize,
    },

    /// Pixelwise arctangent (ITK `AtanImageFilter`). `out(x) = atan(in(x))`.
    Atan,
    /// Pixelwise sine (ITK `SinImageFilter`). `out(x) = sin(in(x))`.
    Sin,
    /// Pixelwise cosine (ITK `CosImageFilter`). `out(x) = cos(in(x))`.
    Cos,
    /// Pixelwise tangent (ITK `TanImageFilter`). `out(x) = tan(in(x))`.
    Tan,
    /// Pixelwise arcsine (ITK `AsinImageFilter`). `out(x) = asin(in(x))`.
    Asin,
    /// Pixelwise arccosine (ITK `AcosImageFilter`). `out(x) = acos(in(x))`.
    Acos,
    /// Pixelwise bounded reciprocal (ITK `BoundedReciprocalImageFilter`). `out(x) = 1/(1+|x|)`.
    BoundedReciprocal,

    /// Pure mean curvature flow (ITK `CurvatureFlowImageFilter`). `∂I/∂t = κ`.
    CurvatureFlow {
        /// Number of explicit-Euler iterations.
        iterations: u32,
        /// Time step Δt ≤ 1/6.
        time_step: f32,
    },
    #[doc = include_str!("variant_docs/cpr.md")]
    Cpr {
        /// Control points in physical coordinates `[z, y, x]`.
        /// At least 2 control points are required.
        control_points: Vec<[f64; 3]>,
        /// Number of samples along the path (output columns).
        num_path_samples: u32,
        /// Cross-section half-width in physical units (mm).
        cross_section_half_width: f32,
        /// Number of cross-section samples (output rows).
        num_cross_samples: u32,
    },
}
