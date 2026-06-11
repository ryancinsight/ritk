// ── Internal ────────────────────────────────────────────────────────────────

// ── Re-export gaussian_kernel_1d from ritk-core::filter::ops ──────────────────
pub use ritk_core::filter::gaussian_kernel_1d;

// ── Bias correction ──────────────────────────────────────────────────────────
pub mod bias;
pub use bias::N4BiasFieldCorrectionFilter;

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

pub use intensity::{
    AbsImageFilter, AcosImageFilter, AddImageFilter, AsinImageFilter, AtanImageFilter,
    BedSeparationConfig, BedSeparationFilter, BinaryThresholdImageFilter, BlendImageFilter,
    BoundedReciprocalImageFilter, ClaheFilter, ClaheScratch, ClampImageFilter, ClampPolicy,
    ComponentPolicy, CosImageFilter, DivideImageFilter, ExpImageFilter,
    HistogramEqualizationFilter, ImageMaxFilter, ImageMinFilter, IntensityWindowingFilter,
    InvertIntensityFilter, LogImageFilter, MaskImageFilter, MaskNegatedImageFilter,
    MultiplyImageFilter, NormalizeImageFilter, RescaleIntensityFilter, ShiftScaleImageFilter,
    SigmoidImageFilter, SinImageFilter, SqrtImageFilter, SquareImageFilter, SubtractImageFilter,
    SuvBodyWeightImageFilter, TanImageFilter, ThresholdImageFilter, ThresholdMode,
    UnsharpMaskFilter, ZeroCrossingImageFilter,
};

// ── Morphology ────────────────────────────────────────────────────────────────
pub mod morphology;

pub use morphology::{
    BinaryContourImageFilter, BinaryDilateFilter, BinaryErodeFilter, BinaryFillholeFilter,
    BinaryMorphologicalClosing, BinaryMorphologicalOpening, BlackTopHatFilter, Connectivity,
    ForegroundValue, GrayscaleClosingFilter, GrayscaleDilation, GrayscaleErosion,
    GrayscaleFillholeFilter, GrayscaleGeodesicDilationFilter, GrayscaleGeodesicErosionFilter,
    GrayscaleMorphologicalGradientFilter, GrayscaleOpeningFilter, HitOrMissTransform, LabelClosing,
    LabelContourImageFilter, LabelDilation, LabelErosion, LabelOpening,
    MorphologicalReconstruction, ReconstructionMode, VotingBinaryImageFilter, WhiteTopHatFilter,
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
    MaxIntensityProjectionFilter, MeanIntensityProjectionFilter, MinIntensityProjectionFilter,
    ProjectionAxis, StdDevIntensityProjectionFilter, SumIntensityProjectionFilter,
};
pub use pyramid::MultiResolutionPyramid;
pub use resample::ResampleImageFilter;
pub use transform::{
    ConstantPadImageFilter, FlipImageFilter, FlipPolicy, MirrorPadImageFilter, Padding,
    PasteImageFilter, PermuteAxesImageFilter, RegionOfInterestImageFilter, ShrinkImageFilter,
    WrapPadImageFilter,
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
