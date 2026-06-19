"""Type stubs for the ``_ritk.filter`` submodule (compiled by PyO3/maturin)."""

from __future__ import annotations

from ritk._ritk.image import ColorImage, Image

def gaussian_filter(image: Image, sigma: float) -> Image: ...
def discrete_gaussian(
    image: Image,
    variance: float,
    maximum_error: float = 0.01,
    use_image_spacing: bool = True,
) -> Image: ...
def median_filter(image: Image, radius: int = 1) -> Image: ...
def discrete_gaussian_derivative(
    image: Image,
    order_x: int,
    order_y: int,
    order_z: int,
    variance: float,
    maximum_error: float = 0.01,
    use_image_spacing: bool = False,
) -> Image:
    """Discrete Gaussian derivative (per-axis order, sitk x/y/z). ITK Parity: DiscreteGaussianDerivativeImageFilter."""
    ...

def normalize_image(image: Image) -> Image: ...
def normalize_to_constant(image: Image, constant: float = 1.0) -> Image:
    """Scale so the sum of voxels equals `constant`. ITK Parity: NormalizeToConstantImageFilter."""
    ...

def derivative(
    image: Image, direction: int = 0, order: int = 1, use_image_spacing: bool = True
) -> Image:
    """Directional central-difference derivative (direction sitk x/y/z; order 1 or 2). ITK Parity: DerivativeImageFilter."""
    ...

def bilateral_filter(
    image: Image, spatial_sigma: float, range_sigma: float
) -> Image: ...
def n4_bias_correction(
    image: Image,
    num_fitting_levels: int = 4,
    num_iterations: int = 50,
    noise_estimate: float = 0.01,
    shrink_factor: int = 4,
) -> Image: ...
def anisotropic_diffusion(
    image: Image,
    iterations: int = 20,
    conductance: float = 3.0,
    time_step: float = 0.0625,
    exponential: bool = True,
) -> Image: ...
def gradient_magnitude(image: Image) -> Image: ...
def laplacian(image: Image) -> Image: ...
def frangi_vesselness(
    image: Image,
    scales: list[float] | None = None,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 15.0,
    bright_vessels: bool = True,
) -> Image: ...
def canny_edge_detect(
    image: Image,
    sigma: float = 1.0,
    low_threshold: float = 0.1,
    high_threshold: float = 0.2,
) -> Image: ...
def laplacian_of_gaussian(image: Image, sigma: float = 1.0) -> Image: ...
def recursive_gaussian(image: Image, sigma: float = 1.0, order: int = 0) -> Image: ...
def recursive_gaussian_directional(
    image: Image, sigma: float = 1.0, order: int = 0, direction: int = 2
) -> Image: ...
def sobel_gradient(image: Image) -> Image: ...
def unsharp_mask(
    image: Image,
    sigma: float = 1.0,
    amount: float = 0.5,
    threshold: float = 0.0,
    clamp: bool = False,
) -> Image: ...
def grayscale_erosion(image: Image, radius: int = 1) -> Image: ...
def grayscale_dilation(image: Image, radius: int = 1) -> Image: ...
def curvature_anisotropic_diffusion(
    image: Image,
    iterations: int = 20,
    time_step: float = 0.0625,
) -> Image: ...
def curvature_flow(
    image: Image, time_step: float = 0.0625, iterations: int = 5
) -> Image:
    """Pure mean-curvature flow (dI/dt = kappa). ITK Parity: CurvatureFlowImageFilter."""
    ...

def min_max_curvature_flow(
    image: Image, time_step: float = 0.05, iterations: int = 5, stencil_radius: int = 2
) -> Image:
    """Min/max curvature flow (curvature flow gated by a stencil min/max threshold). ITK Parity: MinMaxCurvatureFlowImageFilter (sitk.MinMaxCurvatureFlow)."""
    ...

def binary_min_max_curvature_flow(
    image: Image,
    time_step: float = 0.05,
    iterations: int = 5,
    stencil_radius: int = 2,
    threshold: float = 0.0,
) -> Image:
    """Binary min/max curvature flow (curvature flow gated by sphere-average vs scalar threshold). ITK Parity: BinaryMinMaxCurvatureFlowImageFilter (sitk.BinaryMinMaxCurvatureFlow)."""
    ...

def sato_line_filter(
    image: Image,
    scales: list[float] | None = None,
    alpha: float = 0.5,
    bright_tubes: bool = True,
) -> Image: ...
def rescale_intensity(
    image: Image,
    out_min: float = 0.0,
    out_max: float = 1.0,
) -> Image: ...
def adaptive_histogram_equalization(
    image: Image,
    radius: tuple[int, int, int] = (5, 5, 5),
    alpha: float = 0.3,
    beta: float = 0.3,
) -> Image:
    """Adaptive histogram equalization (Stark alpha/beta). ITK Parity: AdaptiveHistogramEqualizationImageFilter."""
    ...

def intensity_windowing(
    image: Image,
    window_min: float,
    window_max: float,
    out_min: float = 0.0,
    out_max: float = 1.0,
) -> Image: ...
def threshold_below(
    image: Image,
    threshold: float,
    outside_value: float = 0.0,
) -> Image: ...
def threshold_above(
    image: Image,
    threshold: float,
    outside_value: float = 0.0,
) -> Image: ...
def threshold_outside(
    image: Image,
    lower: float,
    upper: float,
    outside_value: float = 0.0,
) -> Image: ...
def sigmoid_filter(
    image: Image,
    alpha: float,
    beta: float,
    min_output: float = 0.0,
    max_output: float = 1.0,
) -> Image: ...
def binary_threshold(
    image: Image,
    lower_threshold: float,
    upper_threshold: float,
    foreground: float = 1.0,
    background: float = 0.0,
) -> Image: ...
def double_threshold(
    image: Image,
    threshold1: float = 0.0,
    threshold2: float = 1.0,
    threshold3: float = 254.0,
    threshold4: float = 255.0,
    inside_value: float = 1.0,
    outside_value: float = 0.0,
) -> Image:
    """Hysteresis double-threshold (inner band reconstructed under outer band). ITK Parity: DoubleThresholdImageFilter."""
    ...

def zero_crossing_image(
    image: Image,
    foreground_value: float = 1.0,
    background_value: float = 0.0,
) -> Image: ...
def blend_images(a: Image, b: Image, alpha: float = 0.5) -> Image:
    """Linearly blend two co-registered images.

    out(x) = (1 - alpha) * a(x) + alpha * b(x)

    alpha=0 returns a; alpha=1 returns b. Both images must have identical shapes.
    Spatial metadata is preserved from a. ITK Parity: BlendImageFilter.
    """
    ...

def add_images(a: Image, b: Image) -> Image:
    """Pixelwise addition: out(x) = a(x) + b(x). ITK Parity: AddImageFilter."""
    ...

def subtract_images(a: Image, b: Image) -> Image:
    """Pixelwise subtraction: out(x) = a(x) - b(x). ITK Parity: SubtractImageFilter."""
    ...

def multiply_images(a: Image, b: Image) -> Image:
    """Pixelwise multiplication: out(x) = a(x) * b(x). ITK Parity: MultiplyImageFilter."""
    ...

def divide_images(a: Image, b: Image) -> Image:
    """Pixelwise division: out(x) = a(x) / b(x); division by zero yields 0. ITK Parity: DivideImageFilter."""
    ...

def minimum_images(a: Image, b: Image) -> Image:
    """Pixelwise minimum: out(x) = min(a(x), b(x)). ITK Parity: MinimumImageFilter."""
    ...

def abs_image(image: Image) -> Image:
    """Pixelwise absolute value: out(x) = |in(x)|. ITK Parity: AbsImageFilter."""
    ...

def sqrt_image(image: Image) -> Image:
    """Pixelwise square root: out(x) = sqrt(in(x)). ITK Parity: SqrtImageFilter."""
    ...

def square_image(image: Image) -> Image:
    """Pixelwise square: out(x) = in(x)^2. ITK Parity: SquareImageFilter."""
    ...

def exp_image(image: Image) -> Image:
    """Pixelwise exponential: out(x) = exp(in(x)). ITK Parity: ExpImageFilter."""
    ...

def log_image(image: Image) -> Image:
    """Pixelwise natural log: out(x) = ln(in(x)). ITK Parity: LogImageFilter."""
    ...

def log10_image(image: Image) -> Image:
    """Pixelwise base-10 log: out(x) = log10(in(x)). ITK Parity: Log10ImageFilter."""
    ...

def unary_minus_image(image: Image) -> Image:
    """Pixelwise negation: out(x) = -in(x). ITK Parity: UnaryMinusImageFilter."""
    ...

def round_image(image: Image) -> Image:
    """Pixelwise round to nearest integer (half-up). ITK Parity: RoundImageFilter."""
    ...

def not_image(image: Image) -> Image:
    """Pixelwise logical NOT of a mask (1 where in==0, else 0). ITK Parity: NotImageFilter."""
    ...

def binary_not(image: Image, foreground: float = 1.0, background: float = 0.0) -> Image:
    """Binary logical NOT: background where in==foreground, else foreground. ITK Parity: BinaryNotImageFilter."""
    ...

def modulus(image: Image, dividend: int) -> Image:
    """Pixelwise in % dividend (integer images, C truncated remainder). ITK Parity: ModulusImageFilter."""
    ...

def binary_magnitude_images(a: Image, b: Image) -> Image:
    """Pixelwise magnitude: out(x) = sqrt(a(x)^2 + b(x)^2). ITK Parity: BinaryMagnitudeImageFilter."""
    ...

def equal_images(a: Image, b: Image) -> Image:
    """Pixelwise equality mask (1 where a==b). ITK Parity: EqualImageFilter."""
    ...

def not_equal_images(a: Image, b: Image) -> Image:
    """Pixelwise inequality mask (1 where a!=b). ITK Parity: NotEqualImageFilter."""
    ...

def greater_images(a: Image, b: Image) -> Image:
    """Pixelwise greater-than mask (1 where a>b). ITK Parity: GreaterImageFilter."""
    ...

def greater_equal_images(a: Image, b: Image) -> Image:
    """Pixelwise greater-or-equal mask (1 where a>=b). ITK Parity: GreaterEqualImageFilter."""
    ...

def less_images(a: Image, b: Image) -> Image:
    """Pixelwise less-than mask (1 where a<b). ITK Parity: LessImageFilter."""
    ...

def less_equal_images(a: Image, b: Image) -> Image:
    """Pixelwise less-or-equal mask (1 where a<=b). ITK Parity: LessEqualImageFilter."""
    ...

def and_images(a: Image, b: Image) -> Image:
    """Pixelwise logical AND of binary masks (1 where a>0 and b>0). ITK Parity: AndImageFilter."""
    ...

def or_images(a: Image, b: Image) -> Image:
    """Pixelwise logical OR of binary masks (1 where a>0 or b>0). ITK Parity: OrImageFilter."""
    ...

def xor_images(a: Image, b: Image) -> Image:
    """Pixelwise logical XOR of binary masks (1 where exactly one of a,b > 0). ITK Parity: XorImageFilter."""
    ...

def divide_real_images(a: Image, b: Image) -> Image:
    """Pixelwise real division a/b (FLT_MAX where b==0). ITK Parity: DivideRealImageFilter."""
    ...

def nary_add(images: list[Image]) -> Image:
    """Pixelwise sum of any number of images. ITK Parity: NaryAddImageFilter."""
    ...

def nary_maximum(images: list[Image]) -> Image:
    """Pixelwise maximum of any number of images. ITK Parity: NaryMaximumImageFilter."""
    ...

def divide_floor_images(a: Image, b: Image) -> Image:
    """Pixelwise floored division floor(a/b) (FLT_MAX where b==0). ITK Parity: DivideFloorImageFilter."""
    ...

def ternary_add_images(a: Image, b: Image, c: Image) -> Image:
    """Pixelwise sum of three images: a+b+c. ITK Parity: TernaryAddImageFilter."""
    ...

def ternary_magnitude_images(a: Image, b: Image, c: Image) -> Image:
    """Pixelwise magnitude: sqrt(a^2+b^2+c^2). ITK Parity: TernaryMagnitudeImageFilter."""
    ...

def ternary_magnitude_squared_images(a: Image, b: Image, c: Image) -> Image:
    """Pixelwise squared magnitude: a^2+b^2+c^2. ITK Parity: TernaryMagnitudeSquaredImageFilter."""
    ...

def exp_negative_image(image: Image) -> Image:
    """Pixelwise negative exponential: out(x) = exp(-in(x)). ITK Parity: ExpNegativeImageFilter."""
    ...

def sin_image(image: Image) -> Image:
    """Pixelwise sine: out(x) = sin(in(x)). ITK Parity: SinImageFilter."""
    ...

def cos_image(image: Image) -> Image:
    """Pixelwise cosine: out(x) = cos(in(x)). ITK Parity: CosImageFilter."""
    ...

def tan_image(image: Image) -> Image:
    """Pixelwise tangent: out(x) = tan(in(x)). ITK Parity: TanImageFilter."""
    ...

def asin_image(image: Image) -> Image:
    """Pixelwise arcsine: out(x) = asin(in(x)). ITK Parity: AsinImageFilter."""
    ...

def acos_image(image: Image) -> Image:
    """Pixelwise arccosine: out(x) = acos(in(x)). ITK Parity: AcosImageFilter."""
    ...

def atan_image(image: Image) -> Image:
    """Pixelwise arctangent: out(x) = atan(in(x)). ITK Parity: AtanImageFilter."""
    ...

def bounded_reciprocal_image(image: Image) -> Image:
    """Pixelwise bounded reciprocal: out(x) = 1 / (1 + in(x)). ITK Parity: BoundedReciprocalImageFilter."""
    ...

def mean_filter(image: Image, radius: int = 1) -> Image:
    """Mean (box) filter: average over a (2r+1)^3 cube. ITK Parity: MeanImageFilter."""
    ...

def binomial_blur(image: Image, repetitions: int = 1) -> Image:
    """Separable [1/4,1/2,1/4] blur applied `repetitions` times (reflect boundary). ITK Parity: BinomialBlurImageFilter."""
    ...

def box_mean(
    image: Image, radius_z: int = 1, radius_y: int = 1, radius_x: int = 1
) -> Image:
    """Box mean over a clipped (2r+1) window (shrink boundary). ITK Parity: BoxMeanImageFilter."""
    ...

def box_sigma(
    image: Image, radius_z: int = 1, radius_y: int = 1, radius_x: int = 1
) -> Image:
    """Box sample std-dev over a clipped (2r+1) window (divisor n-1). ITK Parity: BoxSigmaImageFilter."""
    ...

def rank(
    image: Image,
    rank: float = 0.5,
    radius_z: int = 1,
    radius_y: int = 1,
    radius_x: int = 1,
) -> Image:
    """Order statistic at floor(rank*(n-1)) of the sorted clipped window. ITK Parity: RankImageFilter."""
    ...

def color_median(image: ColorImage, radius: int = 1) -> ColorImage:
    """Per-component median on an RGB/vector image (box radius). ITK Parity: MedianImageFilter (vector)."""
    ...

def color_mean(image: ColorImage, radius: int = 1) -> ColorImage:
    """Per-component mean (box) filter on an RGB/vector image. ITK Parity: MeanImageFilter (vector)."""
    ...

def color_smoothing_recursive_gaussian(image: ColorImage, sigma: float) -> ColorImage:
    """Per-component smoothing recursive Gaussian on an RGB/vector image. ITK Parity: SmoothingRecursiveGaussianImageFilter (vector)."""
    ...

def compose(c0: Image, c1: Image, c2: Image) -> ColorImage:
    """Compose three scalar images into a 3-component vector image. ITK Parity: ComposeImageFilter."""
    ...

def gradient(image: Image, use_image_spacing: bool = True) -> ColorImage:
    """Central-difference gradient -> 3-component vector (dx,dy,dz). ITK Parity: GradientImageFilter."""
    ...

def gradient_recursive_gaussian(image: Image, sigma: float = 1.0) -> ColorImage:
    """Gaussian-smoothed gradient -> 3-component vector (dx,dy,dz). ITK Parity: GradientRecursiveGaussianImageFilter."""
    ...

def scalar_to_rgb_colormap(image: Image, colormap: str = "grey") -> ColorImage:
    """Map scalar -> RGB via a colormap (grey/red/green/blue). ITK Parity: ScalarToRGBColormapImageFilter."""
    ...

def label_to_rgb(image: Image, background: int = 0) -> ColorImage:
    """Map a label image to RGB via ITK's 30-colour table (background->black). ITK Parity: LabelToRGBImageFilter."""
    ...

def label_overlay(
    image: Image, label: Image, opacity: float = 0.5, background: int = 0
) -> ColorImage:
    """Overlay a label image on grayscale as RGB (alpha-blend with colour table). ITK Parity: LabelOverlayImageFilter."""
    ...

def label_map_contour_overlay(
    image: Image, label: Image, opacity: float = 0.5, background: int = 0
) -> ColorImage:
    """Overlay label contours on grayscale as RGB. ITK Parity: LabelMapContourOverlayImageFilter (sitk.LabelMapContourOverlay, default geometry)."""
    ...

def physical_point_image_source(
    size: tuple[int, int, int],
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> ColorImage:
    """Generate a vector image holding each voxel's physical coordinate. ITK Parity: PhysicalPointImageSource."""
    ...

def vector_index_selection_cast(image: ColorImage, index: int) -> Image:
    """Extract one component of a vector image as a scalar image. ITK Parity: VectorIndexSelectionCastImageFilter."""
    ...

def edge_potential(image: ColorImage) -> Image:
    """Edge potential exp(-|vector|) of a vector (gradient) image. ITK Parity: EdgePotentialImageFilter."""
    ...

def vector_magnitude(image: ColorImage) -> Image:
    """Per-voxel Euclidean magnitude of a vector image. ITK Parity: VectorMagnitudeImageFilter."""
    ...

def clamp_image(image: Image, lower: float, upper: float) -> Image:
    """Pixelwise clamp to [lower, upper]. ITK Parity: ClampImageFilter."""
    ...

def invert_intensity(image: Image, maximum: float = 255.0) -> Image:
    """Invert intensities about maximum: out(x) = maximum - in(x). ITK Parity: InvertIntensityImageFilter."""
    ...

def mask_image(image: Image, mask: Image, outside_value: float = 0.0) -> Image:
    """Keep image where mask > 0, else outside_value. ITK Parity: MaskImageFilter."""
    ...

def mask_negated_image(image: Image, mask: Image, outside_value: float = 0.0) -> Image:
    """Keep image where mask <= 0, else outside_value. ITK Parity: MaskNegatedImageFilter."""
    ...

def masked_assign(image: Image, mask: Image, assign_value: float = 0.0) -> Image:
    """Assign value where mask>0, keep image elsewhere. ITK Parity: MaskedAssignImageFilter."""
    ...

def maximum_images(a: Image, b: Image) -> Image:
    """Pixelwise maximum: out(x) = max(a(x), b(x)). ITK Parity: MaximumImageFilter."""
    ...

def squared_difference_images(a: Image, b: Image) -> Image:
    """Pixelwise squared difference: out(x) = (a(x) - b(x))^2. ITK Parity: SquaredDifferenceImageFilter."""
    ...

def absolute_value_difference_images(a: Image, b: Image) -> Image:
    """Pixelwise absolute difference: out(x) = |a(x) - b(x)|. ITK Parity: AbsoluteValueDifferenceImageFilter."""
    ...

def atan2_images(a: Image, b: Image) -> Image:
    """Pixelwise four-quadrant arctangent: out(x) = atan2(a(x), b(x)). ITK Parity: Atan2ImageFilter."""
    ...

def pow_images(a: Image, b: Image) -> Image:
    """Pixelwise power: out(x) = a(x) ^ b(x). ITK Parity: PowImageFilter."""
    ...

def white_top_hat(image: Image, radius: int) -> Image: ...
def black_top_hat(image: Image, radius: int) -> Image: ...
def hit_or_miss(image: Image, fg_radius: int, bg_radius: int) -> Image: ...
def label_dilation(image: Image, radius: int) -> Image: ...
def label_erosion(image: Image, radius: int = 1) -> Image: ...
def label_opening(image: Image, radius: int = 1) -> Image: ...
def label_closing(image: Image, radius: int = 1) -> Image: ...
def morphological_reconstruction(
    marker: Image,
    mask: Image,
    mode: str = "dilation",
    fully_connected: bool = False,
) -> Image: ...
def h_maxima(image: Image, height: float, fully_connected: bool = False) -> Image:
    """H-maxima transform (suppress bright maxima below `height`). ITK Parity: HMaximaImageFilter."""
    ...

def h_minima(image: Image, height: float, fully_connected: bool = False) -> Image:
    """H-minima transform (suppress dark minima below `height`). ITK Parity: HMinimaImageFilter."""
    ...

def h_convex(image: Image, height: float, fully_connected: bool = False) -> Image:
    """H-convex transform: f - HMaxima_h(f). ITK Parity: HConvexImageFilter."""
    ...

def h_concave(image: Image, height: float, fully_connected: bool = False) -> Image:
    """H-concave transform: HMinima_h(f) - f. ITK Parity: HConcaveImageFilter."""
    ...

def regional_maxima(
    image: Image,
    foreground: float = 1.0,
    background: float = 0.0,
    fully_connected: bool = False,
) -> Image:
    """Binary regional maxima. ITK Parity: RegionalMaximaImageFilter."""
    ...

def regional_minima(
    image: Image,
    foreground: float = 1.0,
    background: float = 0.0,
    fully_connected: bool = False,
) -> Image:
    """Binary regional minima. ITK Parity: RegionalMinimaImageFilter."""
    ...

def valued_regional_maxima(image: Image, fully_connected: bool = False) -> Image:
    """Valued regional maxima (non-maxima → -FLT_MAX). ITK Parity: ValuedRegionalMaximaImageFilter."""
    ...

def valued_regional_minima(image: Image, fully_connected: bool = False) -> Image:
    """Valued regional minima (non-minima → +FLT_MAX). ITK Parity: ValuedRegionalMinimaImageFilter."""
    ...

def opening_by_reconstruction(
    image: Image, radius: int, fully_connected: bool = False
) -> Image:
    """Opening by reconstruction (box SE). ITK Parity: OpeningByReconstructionImageFilter."""
    ...

def closing_by_reconstruction(
    image: Image, radius: int, fully_connected: bool = False
) -> Image:
    """Closing by reconstruction (box SE). ITK Parity: ClosingByReconstructionImageFilter."""
    ...

def grayscale_closing(image: Image, radius: int) -> Image:
    """Grayscale morphological closing (box SE). ITK Parity: GrayscaleMorphologicalClosingImageFilter."""
    ...

def grayscale_opening(image: Image, radius: int) -> Image:
    """Grayscale morphological opening (box SE). ITK Parity: GrayscaleMorphologicalOpeningImageFilter."""
    ...

def grayscale_fillhole(image: Image) -> Image:
    """Grayscale fill-hole. ITK Parity: GrayscaleFillholeImageFilter."""
    ...

def grayscale_grind_peak(image: Image, fully_connected: bool = False) -> Image:
    """Grayscale grind-peak (dual of fill-hole). ITK Parity: GrayscaleGrindPeakImageFilter."""
    ...

def binary_contour(
    image: Image, fully_connected: bool = False, foreground_value: float = 1.0
) -> Image:
    """Binary object boundary. ITK Parity: BinaryContourImageFilter."""
    ...

def label_contour(
    image: Image, fully_connected: bool = False, background_value: float = 0.0
) -> Image:
    """Per-label boundary voxels. ITK Parity: LabelContourImageFilter."""
    ...

def voting_binary(
    image: Image,
    radius: int = 1,
    birth_threshold: int = 1,
    survival_threshold: int = 1,
    foreground_value: float = 1.0,
    background_value: float = 0.0,
) -> Image:
    """One binary voting (cellular-automaton) step. ITK Parity: VotingBinaryImageFilter."""
    ...

def voting_binary_hole_filling(
    image: Image,
    radius: int = 1,
    majority_threshold: int = 1,
    foreground_value: float = 1.0,
    background_value: float = 0.0,
) -> Image:
    """Fill background holes by majority vote (fg always survives, clamp boundary). ITK Parity: VotingBinaryHoleFillingImageFilter."""
    ...

def voting_binary_iterative_hole_filling(
    image: Image,
    radius: int = 1,
    max_iterations: int = 10,
    majority_threshold: int = 1,
    foreground_value: float = 1.0,
    background_value: float = 0.0,
) -> Image:
    """Iteratively fill background holes by majority vote. ITK Parity: VotingBinaryIterativeHoleFillingImageFilter."""
    ...

def flip(
    image: Image, flip_z: bool = False, flip_y: bool = False, flip_x: bool = False
) -> Image:
    """Flip along Z/Y/X axes. ITK Parity: FlipImageFilter (sitk.Flip, axes reversed to [x,y,z])."""
    ...

def constant_pad(
    image: Image,
    lower: tuple[int, int, int],
    upper: tuple[int, int, int],
    constant: float = 0.0,
) -> Image:
    """Constant-value pad; lower/upper are (z,y,x). ITK Parity: ConstantPadImageFilter."""
    ...

def mirror_pad(
    image: Image, lower: tuple[int, int, int], upper: tuple[int, int, int]
) -> Image:
    """Mirror (symmetric) pad; lower/upper are (z,y,x). ITK Parity: MirrorPadImageFilter."""
    ...

def wrap_pad(
    image: Image, lower: tuple[int, int, int], upper: tuple[int, int, int]
) -> Image:
    """Wrap (periodic) pad; lower/upper are (z,y,x). ITK Parity: WrapPadImageFilter."""
    ...

def zero_flux_neumann_pad(
    image: Image, lower: tuple[int, int, int], upper: tuple[int, int, int]
) -> Image:
    """Edge-replicate (zero-flux Neumann) pad; lower/upper are (z,y,x). ITK Parity: ZeroFluxNeumannPadImageFilter."""
    ...

def fft_pad(image: Image, max_prime: int = 5, boundary: int = 1) -> Image:
    """Pad each axis to the next size whose greatest prime factor is <= max_prime
    (default 5). boundary: 0 zero, 1 zero-flux Neumann (default), 2 periodic.
    ITK Parity: FFTPadImageFilter."""
    ...

def region_of_interest(
    image: Image, start: tuple[int, int, int], size: tuple[int, int, int]
) -> Image:
    """Crop to a sub-region; start/size are (z,y,x). ITK Parity: RegionOfInterestImageFilter."""
    ...

def crop(
    image: Image, lower: tuple[int, int, int], upper: tuple[int, int, int]
) -> Image:
    """Crop lower/upper voxels off each axis face (z,y,x). ITK Parity: CropImageFilter."""
    ...

def cyclic_shift(image: Image, shift: tuple[int, int, int]) -> Image:
    """Cyclically roll the image by (z,y,x) voxels (wrap-around). ITK Parity: CyclicShiftImageFilter."""
    ...

def join_series(images: list[Image]) -> Image:
    """Stack a list of [zi,Y,X] images along Z into [sum zi, Y, X]. ITK Parity: JoinSeriesImageFilter."""
    ...

def tile(
    images: list[Image], layout: tuple[int, int, int], default_value: float = 0.0
) -> Image:
    """Tile same-sized images into a grid (layout = nx,ny,nz tiles per axis). ITK Parity: TileImageFilter."""
    ...

def checker_board(
    image1: Image, image2: Image, pattern: tuple[int, int, int] = (4, 4, 4)
) -> Image:
    """Checkerboard-combine two images (pattern = nx,ny,nz cells). ITK Parity: CheckerBoardImageFilter."""
    ...

def slice_image(
    image: Image,
    start: tuple[int, int, int],
    stop: tuple[int, int, int],
    step: tuple[int, int, int],
) -> Image:
    """Strided sub-region (start:stop:step per axis, z,y,x order). ITK Parity: SliceImageFilter."""
    ...

def expand(image: Image, factors: tuple[int, int, int]) -> Image:
    """Integer-factor upsample (z,y,x) via linear interpolation. ITK Parity: ExpandImageFilter."""
    ...

def permute_axes(image: Image, order: tuple[int, int, int]) -> Image:
    """Permute tensor axes (order is a [z,y,x] permutation). ITK Parity: PermuteAxesImageFilter."""
    ...

def paste(dest: Image, source: Image, dest_start: tuple[int, int, int]) -> Image:
    """Paste source into dest at dest_start (z,y,x). ITK Parity: PasteImageFilter."""
    ...

def resample_image(
    image: Image,
    spacing_z: float = 1.0,
    spacing_y: float = 1.0,
    spacing_x: float = 1.0,
    mode: str = "linear",
) -> Image: ...
def warp(moving: Image, disp_z: Image, disp_y: Image, disp_x: Image) -> Image:
    """Warp `moving` through a dense displacement field (trilinear). out(p)=moving(p+D(p)). ITK Parity: WarpImageFilter."""
    ...

def bspline_decomposition(image: Image) -> Image:
    """Cubic B-spline interpolation coefficients (mirror boundary). ITK Parity: BSplineDecompositionImageFilter (order 3)."""
    ...

def rotate_image(
    image: Image,
    angle_x: float = 0.0,
    angle_y: float = 0.0,
    angle_z: float = 0.0,
    mode: str = "linear",
    default_pixel_value: float = 0.0,
) -> Image: ...
def shift_image(
    image: Image,
    shift_z: float = 0.0,
    shift_y: float = 0.0,
    shift_x: float = 0.0,
    mode: str = "linear",
    default_pixel_value: float = 0.0,
) -> Image: ...
def zoom_image(
    image: Image,
    zoom_z: float = 1.0,
    zoom_y: float = 1.0,
    zoom_x: float = 1.0,
    mode: str = "linear",
) -> Image: ...
def distance_transform(
    image: Image,
    foreground_threshold: float = 0.5,
    squared: bool = False,
) -> Image:
    """Euclidean distance transform (Meijster et al. 2000).

    For each background voxel, computes distance to nearest foreground voxel
    in physical units (respecting image spacing). Foreground voxels get 0.0.

    Args:
        image:                Binary input image.
        foreground_threshold: Voxels above this value are foreground (default 0.5).
        squared:              If True, return squared distances (default False).

    Returns:
        Distance image with identical shape and spatial metadata.
    """
    ...

def signed_distance_map(image: Image, foreground_threshold: float = 0.5) -> Image:
    """Signed Euclidean distance map (voxel-centre convention).

    Foreground voxels get -dist to nearest background; background voxels get
    +dist to nearest foreground (physical units). Float-exact to scipy signed
    distance_transform_edt. NOT sitk.SignedMaurerDistanceMap (which measures to
    the object boundary, differing by up to sqrt(2) voxel).
    """
    ...

# -- Noise generators ---------------------------------------------------------

def additive_gaussian_noise(
    image: Image, std: float, mean: float = 0.0, seed: int = 42
) -> Image: ...
def salt_and_pepper_noise(
    image: Image, probability: float, seed: int = 42
) -> Image: ...
def shot_noise(image: Image, scale: float, seed: int = 42) -> Image: ...
def speckle_noise(image: Image, std: float, seed: int = 42) -> Image: ...

# -- Resampling / diffusion ---------------------------------------------------

def bin_shrink(
    image: Image, factor_z: int = 2, factor_y: int = 2, factor_x: int = 2
) -> Image: ...
def shrink(
    image: Image, factor_z: int = 2, factor_y: int = 2, factor_x: int = 2
) -> Image:
    """Subsample (no averaging), ITK Shrink. factors (z,y,x); sitk.Shrink uses [x,y,z]. ITK Parity: ShrinkImageFilter."""
    ...

def gaussian_image_source(
    size: tuple[int, int, int],
    sigma: tuple[float, float, float],
    mean: tuple[float, float, float],
    scale: float = 255.0,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Image:
    """Generate a Gaussian blob image (params in sitk x,y,z order). ITK Parity: GaussianImageSource."""
    ...

def grid_image_source(
    size: tuple[int, int, int],
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    sigma: tuple[float, float, float] = (0.5, 0.5, 0.5),
    grid_spacing: tuple[float, float, float] = (4.0, 4.0, 4.0),
    grid_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale: float = 255.0,
    which_dimensions: tuple[bool, bool, bool] = (True, True, True),
) -> Image:
    """Generate a grid-pattern image (dark Gaussian lines). ITK Parity: GridImageSource."""
    ...

def gabor_image_source(
    size: tuple[int, int, int],
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    sigma: tuple[float, float, float] = (16.0, 16.0, 16.0),
    mean: tuple[float, float, float] = (32.0, 32.0, 32.0),
    frequency: float = 0.4,
) -> Image:
    """Generate a Gabor-wavelet image (Gaussian envelope * cosine along x). ITK Parity: GaborImageSource."""
    ...

def coherence_enhancing_diffusion(
    image: Image,
    sigma: float = 3.0,
    contrast: float = 1e-10,
    alpha: float = 0.001,
    time_step: float = 0.0625,
    iterations: int = 10,
) -> Image: ...
def anti_alias_binary(
    image: Image,
    max_rms_error: float = 0.01,
    number_of_iterations: int = 50,
) -> Image:
    """Narrow-band level-set smoothing of binary boundaries via mean curvature flow.
    Returns the level-set function \u03c6 (negative inside). ITK Parity: AntiAliasBinaryImageFilter."""
    ...

def scalar_chan_and_vese_dense_level_set(
    initial_level_set: Image,
    feature_image: Image,
    number_of_iterations: int = 20,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
) -> Image:
    """Dense Chan-Vese level set with user-supplied \u03c6\u2080 and feature image.
    ITK Parity: ScalarChanAndVeseDenseLevelSetImageFilter."""
    ...

def canny_segmentation_level_set(
    initial_level_set: Image,
    feature_image: Image,
    canny_threshold: float = 0.1,
    canny_variance: float = 1.0,
    number_of_iterations: int = 20,
    max_rms_error: float = 0.01,
    propagation_scaling: float = 1.0,
) -> Image:
    """Canny-edge-guided level set segmentation.
    ITK Parity: CannySegmentationLevelSetImageFilter."""
    ...

def patch_based_denoising(
    image: Image,
    number_of_iterations: int = 1,
    number_of_sample_patches: int = 200,
    patch_radius: int = 4,
    kernel_bandwidth_estimation: bool = False,
) -> Image:
    """Patch-based non-local-means denoising.
    ITK Parity: PatchBasedDenoisingImageFilter."""
    ...

# -- FFT spectral filters -----------------------------------------------------

def forward_fft(image: Image) -> Image: ...
def inverse_fft(image: Image) -> Image: ...
def fft_shift(image: Image) -> Image: ...
def complex_to_real(image: Image) -> Image:
    """Real part of an interleaved [D,H,2W] complex image. ITK Parity: ComplexToRealImageFilter."""
    ...

def complex_to_imaginary(image: Image) -> Image:
    """Imaginary part of a complex image. ITK Parity: ComplexToImaginaryImageFilter."""
    ...

def complex_to_modulus(image: Image) -> Image:
    """Modulus sqrt(re^2+im^2) of a complex image. ITK Parity: ComplexToModulusImageFilter."""
    ...

def complex_to_phase(image: Image) -> Image:
    """Phase atan2(im,re) of a complex image. ITK Parity: ComplexToPhaseImageFilter."""
    ...

def real_and_imaginary_to_complex(real: Image, imaginary: Image) -> Image:
    """Build interleaved [D,H,2W] complex from real+imag. ITK Parity: RealAndImaginaryToComplexImageFilter."""
    ...

def magnitude_and_phase_to_complex(magnitude: Image, phase: Image) -> Image:
    """Build complex from magnitude+phase (re=m*cos p, im=m*sin p). ITK Parity: MagnitudeAndPhaseToComplexImageFilter."""
    ...

def fft_convolve(image: Image, kernel: Image) -> Image: ...
def fft_convolve_3d(volume: Image, kernel: Image) -> Image: ...
def fft_normalized_correlate(image: Image, template: Image) -> Image: ...
def fft_normalized_correlate_3d(volume: Image, template: Image) -> Image: ...
def masked_fft_normalized_correlation(
    fixed: Image,
    moving: Image,
    fixed_mask: Image,
    moving_mask: Image,
    greatest_prime_factor: int = 0,
    required_fraction: float = 0.0,
) -> Image:
    """Masked FFT normalized cross-correlation. ITK Parity: MaskedFFTNormalizedCorrelationImageFilter."""
    ...

def normalized_correlation(
    image: Image,
    mask: Image,
    template: Image,
) -> Image:
    """Local normalized correlation of image with a template (mask-gated). ITK Parity: NormalizedCorrelationImageFilter."""
    ...

def fft_ideal_low_pass(image: Image, cutoff: float = 0.3) -> Image: ...
def fft_ideal_high_pass(image: Image, cutoff: float = 0.3) -> Image: ...
def fft_butterworth_low_pass(
    image: Image, cutoff: float = 0.3, order: int = 2
) -> Image: ...
def fft_butterworth_high_pass(
    image: Image, cutoff: float = 0.3, order: int = 2
) -> Image: ...

# -- Deconvolution ------------------------------------------------------------

def richardson_lucy_deconvolution(
    image: Image, kernel: Image, max_iterations: int = 30, tolerance: float = 1e-06
) -> Image: ...
def landweber_deconvolution(
    image: Image,
    kernel: Image,
    step_size: float = 0.1,
    max_iterations: int = 100,
    tolerance: float = 1e-06,
) -> Image: ...
def projected_landweber_deconvolution(
    image: Image,
    kernel: Image,
    step_size: float = 0.1,
    max_iterations: int = 100,
    tolerance: float = 1e-06,
) -> Image:
    """Landweber with a non-negativity projection each iteration. ITK Parity: ProjectedLandweberDeconvolution."""
    ...

def inverse_deconvolution(
    image: Image, kernel: Image, kernel_zero_magnitude_threshold: float = 1e-4
) -> Image:
    """Direct inverse filter U=G/H, zeroing frequencies where |H|<threshold. ITK Parity: InverseDeconvolution."""
    ...

def wiener_deconvolution(
    image: Image, kernel: Image, noise_to_signal: float = 0.01
) -> Image: ...
def tikhonov_deconvolution(
    image: Image, kernel: Image, lambda_: float = 0.01
) -> Image: ...

# -- Intensity projections ----------------------------------------------------

def max_intensity_projection(image: Image, axis: int = 0) -> Image: ...
def min_intensity_projection(image: Image, axis: int = 0) -> Image: ...
def mean_intensity_projection(image: Image, axis: int = 0) -> Image: ...
def median_intensity_projection(image: Image, axis: int = 0) -> Image: ...
def binary_projection(
    image: Image, axis: int = 0, foreground: float = 1.0, background: float = 0.0
) -> Image:
    """Binary projection (foreground if any voxel == foreground). ITK Parity: BinaryProjectionImageFilter."""
    ...

def binary_threshold_projection(
    image: Image,
    axis: int = 0,
    threshold: float = 0.0,
    foreground: float = 1.0,
    background: float = 0.0,
) -> Image:
    """Binary-threshold projection (foreground if any voxel >= threshold). ITK Parity: BinaryThresholdProjectionImageFilter."""
    ...

def sum_intensity_projection(image: Image, axis: int = 0) -> Image: ...
def stddev_intensity_projection(image: Image, axis: int = 0) -> Image: ...

# -- Parity resolution additions ----------------------------------------------

def binary_thinning(image: Image) -> Image:
    """Thin a binary image to its 1-pixel-wide skeleton. ITK Parity: BinaryThinningImageFilter."""
    ...

def binary_pruning(image: Image, iteration: int = 3) -> Image:
    """Prune short spurs from a binary skeleton. ITK Parity: BinaryPruningImageFilter."""
    ...

def erode_object_morphology(
    image: Image,
    radius: int = 1,
    object_value: float = 1.0,
    background_value: float = 0.0,
) -> Image:
    """Erode an object's surface with a box structuring element. ITK Parity: ErodeObjectMorphologyImageFilter."""
    ...

def real_to_half_hermitian_forward_fft(image: Image) -> Image:
    """Real-to-half-Hermitian forward FFT. ITK Parity: RealToHalfHermitianForwardFFTImageFilter."""
    ...

def half_hermitian_to_real_inverse_fft(
    image: Image, actual_x_is_odd: bool = False
) -> Image:
    """Half-Hermitian-to-real inverse FFT. ITK Parity: HalfHermitianToRealInverseFFTImageFilter."""
    ...

def laplacian_sharpening(image: Image, use_image_spacing: bool = True) -> Image:
    """Sharpen an image by subtracting its (range-rescaled) Laplacian. ITK Parity: LaplacianSharpeningImageFilter."""
    ...

def zero_crossing_based_edge_detection(
    image: Image,
    variance: float = 1.0,
    maximum_error: float = 0.01,
    foreground_value: float = 1.0,
    background_value: float = 0.0,
) -> Image:
    """Zero-crossing-based edge detection. ITK Parity: ZeroCrossingBasedEdgeDetectionImageFilter."""
    ...

def iso_contour_distance(
    image: Image, level_set_value: float = 0.0, far_value: float = 10.0
) -> Image:
    """Narrow-band signed distance to the iso-contour. ITK Parity: IsoContourDistanceImageFilter."""
    ...

def approximate_signed_distance_map(
    image: Image,
    inside_value: float = 1.0,
    outside_value: float = 0.0,
) -> Image:
    """Approximate signed distance map via IsoContourDistance + FastChamferDistance.
    ITK Parity: ApproximateSignedDistanceMapImageFilter."""
    ...

def local_noise(
    image: Image, radius_z: int = 1, radius_y: int = 1, radius_x: int = 1
) -> Image:
    """Estimate local image noise. ITK Parity: NoiseImageFilter."""
    ...

def stochastic_fractal_dimension(image: Image, radius: int = 2) -> Image:
    """Per-voxel stochastic fractal dimension. ITK Parity: StochasticFractalDimensionImageFilter."""
    ...

def transform_to_displacement_field(
    reference: Image,
    matrix: list[list[float]],
    translation: list[float],
    center: list[float] = [0.0, 0.0, 0.0],
) -> tuple[Image, Image, Image]:
    """Sample an affine transform onto a reference grid as a dense displacement field. ITK Parity: TransformToDisplacementField."""
    ...

def fast_marching(
    image: Image,
    trial_points: list[list[int]],
    normalization_factor: float = 1.0,
    stopping_value: float | None = None,
    initial_trial_values: list[float] = [],
) -> Image:
    """Solve the Eikonal arrival-time field by fast marching. ITK Parity: FastMarchingImageFilter."""
    ...

def colliding_fronts(
    image: Image,
    seeds1: list[list[int]],
    seeds2: list[list[int]],
    apply_connectivity: bool = True,
    negative_epsilon: float = -1e-6,
) -> Image:
    """Colliding-fronts segmentation potential. ITK Parity: CollidingFrontsImageFilter."""
    ...

def transform_geometry(
    image: Image,
    matrix: list[list[float]],
    translation: list[float],
    center: list[float] = [0.0, 0.0, 0.0],
) -> Image:
    """Apply an affine transform to the image geometry (origin + direction), leaving
    voxel data and spacing unchanged. ITK Parity: TransformGeometryImageFilter."""
    ...

def invert_displacement_field(
    disp_z: Image,
    disp_y: Image,
    disp_x: Image,
    max_iterations: int = 10,
    max_error_tolerance: float = 0.1,
    mean_error_tolerance: float = 0.001,
    enforce_boundary: bool = True,
) -> tuple[Image, Image, Image]:
    """Iteratively invert a dense displacement field (Chen et al. fixed-point scheme).
    Returns (disp_z, disp_y, disp_x) inverse field components.
    ITK Parity: InvertDisplacementFieldImageFilter."""
    ...

def inverse_displacement_field(
    disp_z: Image,
    disp_y: Image,
    disp_x: Image,
    subsampling_factor: int = 16,
) -> tuple[Image, Image, Image]:
    """Invert a dense displacement field by thin-plate-spline landmark fitting.
    Returns (disp_z, disp_y, disp_x) inverse field components; float-exact (z==1 ⇒ 2-D).
    ITK Parity: InverseDisplacementFieldImageFilter (sitk.InverseDisplacementField)."""
    ...

def iterative_inverse_displacement_field(
    disp_z: Image,
    disp_y: Image,
    disp_x: Image,
    number_of_iterations: int = 5,
    stop_value: float = 0.0,
) -> tuple[Image, Image, Image]:
    """Coordinate-descent line-search inversion of a dense displacement field.
    Returns (disp_z, disp_y, disp_x) inverse field components.
    ITK Parity: IterativeInverseDisplacementFieldImageFilter."""
    ...

def dicom_orient(image: Image, orientation: str) -> Image:
    """Reorient image to the specified DICOM orientation code (e.g. 'LPS', 'RAS').
    Performs a signed axis permutation — no resampling. ITK Parity: DICOMOrientImageFilter."""
    ...

def reinitialize_level_set(image: Image, level_set_value: float = 0.0) -> Image:
    """Reinitialize a level-set image to a signed distance function.
    Deterministic: zero-crossing extractor + fast marching (NOT an iterative PDE).
    ITK Parity: ReinitializeLevelSetImageFilter."""
    ...

def bitwise_not(
    image: Image,
    bits: int = 8,
    signed: bool = False,
) -> Image:
    """Bitwise complement of an integer image represented as f32.
    For unsigned width `bits`: result = (2^bits - 1) - round(x).
    For signed two's complement: result = -round(x) - 1.
    Bit-exact to sitk.BitwiseNot for uint8/uint16/int16.
    ITK Parity: BitwiseNotImageFilter."""
    ...
