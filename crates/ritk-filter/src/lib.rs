// ── Internal ────────────────────────────────────────────────────────────────

// ── Re-export gaussian_kernel from ritk-core::filter ────────────────────────────
pub use ritk_tensor_ops::gaussian_kernel;

// ── Bias correction ──────────────────────────────────────────────────────────
pub mod bias;
pub use bias::N4BiasFieldCorrectionFilter;
pub mod color;
pub use color::map_color_components;

// ── Denoising & smoothing ────────────────────────────────────────────────────
pub mod bilateral;
pub mod diffusion;
pub mod discrete_gaussian;
pub mod gaussian;
pub mod median;
pub mod noise;
pub mod recursive_gaussian;
pub mod smoothing;

pub use bilateral::{BilateralFilter, RangeSigma, SpatialSigma};
pub use diffusion::{
    AnisotropicDiffusionFilter, CoherenceConfig, CoherenceEnhancingDiffusionFilter,
    ConductanceFunction, ConductanceKernel, CurvatureAnisotropicDiffusionFilter, CurvatureConfig,
    CurvatureFlowConfig, CurvatureFlowImageFilter, DiffusionConfig, ExponentialConductance,
    GradientAnisotropicDiffusionFilter, GradientDiffusionConfig, QuadraticConductance,
};
pub use discrete_gaussian::{DiscreteGaussianFilter, SpacingMode};
pub use gaussian::GaussianFilter;
pub use median::MedianFilter;
pub use noise::{
    AdditiveGaussianNoiseFilter, SaltAndPepperNoiseFilter, ShotNoiseFilter, SpeckleNoiseFilter,
};
pub use recursive_gaussian::{DerivativeOrder, RecursiveGaussianFilter, ScaleNormalization};
pub use smoothing::MeanImageFilter;

// ── Intensity & histogram ────────────────────────────────────────────────────
pub mod intensity;

pub use intensity::binary_ops::{AddOp, BinaryOp, BinaryOpFilter, MaxOp};
pub use intensity::{
    AbsImageFilter, AbsoluteValueDifferenceImageFilter, AcosImageFilter, AddImageFilter,
    AsinImageFilter, Atan2ImageFilter, AtanImageFilter, BedSeparationConfig, BedSeparationFilter,
    BinaryMagnitudeImageFilter, BinaryThresholdImageFilter, BlendImageFilter,
    BoundedReciprocalImageFilter, ClaheFilter, ClaheScratch, ClampImageFilter, ClampPolicy,
    ComponentPolicy, CosImageFilter, DivideFloorImageFilter, DivideImageFilter,
    DivideRealImageFilter, DoubleThresholdImageFilter, EqualImageFilter, ExpImageFilter, ExpNegativeImageFilter,
    GreaterEqualImageFilter, GreaterImageFilter, HistogramEqualizationFilter, ImageMaxFilter,
    ImageMinFilter, IntensityWindowingFilter, InvertIntensityFilter, LessEqualImageFilter,
    LessImageFilter, Log10ImageFilter, LogImageFilter, MaskImageFilter, MaskNegatedImageFilter,
    MultiplyImageFilter, NormalizeImageFilter, NotEqualImageFilter, PowImageFilter,
    RescaleIntensityFilter, RoundImageFilter, ShiftScaleImageFilter, SigmoidImageFilter,
    SinImageFilter, SqrtImageFilter, SquareImageFilter, SquaredDifferenceImageFilter,
    SubtractImageFilter, SuvBodyWeightImageFilter, TanImageFilter, TernaryAddImageFilter,
    TernaryMagnitudeImageFilter, TernaryMagnitudeSquaredImageFilter, ThresholdImageFilter,
    ThresholdMode, UnaryMinusImageFilter, UnsharpMaskFilter, ZeroCrossingImageFilter,
};

// ── Morphology ────────────────────────────────────────────────────────────────
pub mod morphology;

pub use morphology::{
    BinaryContourImageFilter, BinaryDilateFilter, BinaryErodeFilter, BinaryFillholeFilter,
    BinaryMorphologicalClosing, BinaryMorphologicalOpening, BlackTopHatFilter,
    ClosingByReconstructionFilter, Connectivity, ForegroundValue, GrayscaleClosingFilter,
    GrayscaleDilation, GrayscaleErosion, GrayscaleFillholeFilter, GrayscaleGeodesicDilationFilter,
    GrayscaleGeodesicErosionFilter, GrayscaleGrindPeakFilter, GrayscaleMorphologicalGradientFilter,
    GrayscaleOpeningFilter, HConcaveFilter, HConvexFilter, HMaximaFilter, HMinimaFilter,
    HitOrMissTransform, LabelClosing, LabelContourImageFilter, LabelDilation, LabelErosion,
    LabelOpening, MorphologicalReconstruction, OpeningByReconstructionFilter, ReconstructionMode,
    RegionalMaximaFilter, RegionalMinimaFilter, ValuedRegionalMaximaFilter,
    ValuedRegionalMinimaFilter, VotingBinaryImageFilter, WhiteTopHatFilter,
};

// ── Edge detection & vesselness ──────────────────────────────────────────────
pub mod edge;
pub mod vesselness;

pub use edge::{
    CannyEdgeDetector, GaussianSigma, GradientMagnitudeFilter, LaplacianFilter,
    LaplacianOfGaussianFilter, SobelFilter,
};
pub use vesselness::{
    FrangiConfig, FrangiVesselnessFilter, SatoConfig, SatoLineFilter, VesselPolarity,
};

// ── Frequency domain & deconvolution ─────────────────────────────────────────
pub mod deconvolution;
pub mod fft;

pub use deconvolution::{
    LandweberDeconvolution, RichardsonLucyDeconvolution, TikhonovDeconvolution, WienerDeconvolution,
};
pub use fft::{
    FftConvolution3DFilter, FftConvolutionFilter, FftFilterKind, FftNormalizedCorrelation3DFilter,
    FftNormalizedCorrelationFilter, FftShiftFilter, ForwardFftFilter, FrequencyDomainFilter,
    InverseFftFilter,
};

// ── Spatial transforms & grid operations ─────────────────────────────────────
pub mod bin_shrink;
pub mod cpr;
pub mod downsample;
pub mod projection;
pub mod pyramid;
pub mod resample;
pub mod transform;

pub use bin_shrink::BinShrinkImageFilter;
pub use cpr::{generate_path, generate_path_batch};
pub use cpr::{CprConfig, CprImageFilter};
pub use downsample::DownsampleFilter;
pub use projection::{
    BinaryProjectionFilter, BinaryThresholdProjectionFilter, MaxIntensityProjectionFilter,
    MeanIntensityProjectionFilter, MedianIntensityProjectionFilter, MinIntensityProjectionFilter,
    ProjectionAxis, StdDevIntensityProjectionFilter, SumIntensityProjectionFilter,
};
pub use pyramid::MultiResolutionPyramid;
pub use resample::ResampleImageFilter;
pub use transform::{
    ConstantPadImageFilter, CyclicShiftImageFilter, ExpandImageFilter, FlipImageFilter, FlipPolicy,
    MirrorPadImageFilter, Padding, PasteImageFilter, PermuteAxesImageFilter,
    RegionOfInterestImageFilter, ShrinkImageFilter, WrapPadImageFilter,
    ZeroFluxNeumannPadImageFilter,
};

// ── Surface & distance ───────────────────────────────────────────────────────
pub mod distance;
pub mod rank;
pub mod surface;

pub use distance::{
    BinarizationThreshold, DistanceTransformImageFilter, SignedDistanceTransformImageFilter,
};
pub use rank::{PercentileFilter, RankFilter};
pub use surface::{MarchingCubesFilter, Mesh, MeshBuilder};
