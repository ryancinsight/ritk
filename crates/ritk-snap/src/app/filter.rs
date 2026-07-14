use super::state::{LoadBackend, SnapApp};
use ritk_image::tensor::{Shape, TensorData};
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};

mod native;

impl SnapApp {
    pub(crate) fn apply_filter_to_loaded_volume(&mut self) {
        use ritk_filter::{
            AbsImageFilter, BedSeparationFilter, BinaryDilateFilter, BinaryErodeFilter,
            BinaryFillholeFilter, BinaryMorphologicalClosing, BinaryMorphologicalOpening,
            ClaheFilter, Connectivity, ExpImageFilter, GaussianFilter, GaussianSigma,
            GradientAnisotropicDiffusionFilter, GradientDiffusionConfig, GrayscaleClosingFilter,
            GrayscaleFillholeFilter, GrayscaleMorphologicalGradientFilter, GrayscaleOpeningFilter,
            HistogramEqualizationFilter, InvertIntensityFilter, LogImageFilter, MedianFilter,
            NormalizeImageFilter, SqrtImageFilter, SquareImageFilter, UnsharpMaskFilter,
        };
        use ritk_segmentation::{
            ConnectedComponentsFilter, MultiOtsuThreshold, RelabelComponentFilter,
        };

        let Some(vol) = self.loaded.as_ref() else {
            self.status_message = "No volume loaded.".to_owned();
            return;
        };

        let filter_kind = self.active_filter.clone();
        if let Some(native_result) = native::apply_if_supported(vol, &filter_kind) {
            match native_result {
                Err(error) => {
                    self.status_message = format!("Filter failed: {error:#}");
                }
                Ok(output) => self.replace_loaded_volume_native(output),
            }
            return;
        }

        let [depth, rows, cols] = vol.shape;
        let device = burn_ndarray::NdArrayDevice::Cpu;

        // Build Image<LoadBackend, 3> from the flat volume data.
        let td = TensorData::new((*vol.data).clone(), Shape::new([depth, rows, cols]));
        let tensor = ritk_image::tensor::Tensor::<LoadBackend, 3>::from_data(td, &device);

        let origin = Point::new(vol.origin);
        let spacing = Spacing::new(vol.spacing);
        let direction = Direction::from_rows([
            [vol.direction[0], vol.direction[1], vol.direction[2]],
            [vol.direction[3], vol.direction[4], vol.direction[5]],
            [vol.direction[6], vol.direction[7], vol.direction[8]],
        ]);
        let image: Image<LoadBackend, 3> = Image::new(tensor, origin, spacing, direction);

        // Apply the selected filter.
        let filter_result = {
            match &filter_kind {
                crate::FilterKind::BedSeparation(config) => {
                    BedSeparationFilter::new(*config).apply(&image)
                }
                crate::FilterKind::Gaussian { sigma } => {
                    let s_val = f64::from(*sigma);
                    let g_sigma = GaussianSigma::new(s_val)
                        .unwrap_or_else(|| GaussianSigma::new_unchecked(1e-9));
                    Ok(GaussianFilter::<LoadBackend>::new(vec![g_sigma; 3]).apply(&image))
                }
                crate::FilterKind::Median { radius } => MedianFilter::new(*radius).apply(&image),
                crate::FilterKind::Clahe {
                    tile_grid_size,
                    clip_limit,
                } => ClaheFilter::new(*tile_grid_size, *clip_limit, 256).apply(&image),
                crate::FilterKind::HistEq { bins } => {
                    HistogramEqualizationFilter::new(*bins).apply(&image)
                }
                crate::FilterKind::UnsharpMask {
                    sigma,
                    amount,
                    threshold,
                    clamp,
                } => UnsharpMaskFilter::new(
                    vec![GaussianSigma::new_unchecked(f64::from(*sigma))],
                    f64::from(*amount),
                    f64::from(*threshold),
                    *clamp,
                )
                .apply(&image),
                crate::FilterKind::GradientAnisotropicDiffusion {
                    iterations,
                    time_step,
                    conductance,
                } => GradientAnisotropicDiffusionFilter::new(GradientDiffusionConfig {
                    num_iterations: *iterations as usize,
                    time_step: *time_step,
                    conductance: *conductance,
                })
                .apply(&image),
                crate::FilterKind::ConnectedComponents {
                    connectivity,
                    background_value,
                } => {
                    let seg_connectivity = match connectivity {
                        Connectivity::Face6 => ritk_segmentation::labeling::Connectivity::Six,
                        Connectivity::Vertex26 => {
                            ritk_segmentation::labeling::Connectivity::TwentySix
                        }
                    };
                    ConnectedComponentsFilter::with_connectivity(seg_connectivity)
                        .with_background(*background_value)
                        .map(|filter| filter.apply(&image).0)
                }
                crate::FilterKind::RelabelComponents {
                    minimum_object_size,
                } => {
                    RelabelComponentFilter::with_minimum_object_size(*minimum_object_size as usize)
                        .apply(&image)
                        .map(|(relabeled, _stats)| relabeled)
                }
                crate::FilterKind::MultiOtsuThreshold { num_classes } => {
                    Ok(MultiOtsuThreshold::new(*num_classes as usize).apply(&image))
                }
                crate::FilterKind::BinaryErode {
                    radius,
                    foreground_value,
                } => BinaryErodeFilter::new(*radius)
                    .with_foreground(*foreground_value)
                    .apply(&image),
                crate::FilterKind::BinaryDilate {
                    radius,
                    foreground_value,
                } => BinaryDilateFilter::new(*radius)
                    .with_foreground(*foreground_value)
                    .apply(&image),
                crate::FilterKind::BinaryClosing {
                    radius,
                    foreground_value,
                } => BinaryMorphologicalClosing::new(*radius)
                    .with_foreground(*foreground_value)
                    .apply(&image),
                crate::FilterKind::BinaryOpening {
                    radius,
                    foreground_value,
                } => BinaryMorphologicalOpening::new(*radius)
                    .with_foreground(*foreground_value)
                    .apply(&image),
                crate::FilterKind::BinaryFillhole { foreground_value } => {
                    BinaryFillholeFilter::new()
                        .with_foreground(*foreground_value)
                        .apply(&image)
                }
                crate::FilterKind::GrayscaleClosing { radius } => {
                    GrayscaleClosingFilter::new(*radius).apply(&image)
                }
                crate::FilterKind::GrayscaleOpening { radius } => {
                    GrayscaleOpeningFilter::new(*radius).apply(&image)
                }
                crate::FilterKind::GrayscaleFillhole => {
                    GrayscaleFillholeFilter::new().apply(&image)
                }
                crate::FilterKind::Abs => Ok(AbsImageFilter::new().apply(&image)),
                crate::FilterKind::InvertIntensity { maximum } => Ok(match maximum {
                    Some(m) => InvertIntensityFilter::with_maximum(*m).apply(&image),
                    None => InvertIntensityFilter::new().apply(&image),
                }),
                crate::FilterKind::NormalizeIntensity => {
                    Ok(NormalizeImageFilter::new().apply(&image))
                }
                crate::FilterKind::Square => Ok(SquareImageFilter::new().apply(&image)),
                crate::FilterKind::Sqrt => Ok(SqrtImageFilter::new().apply(&image)),
                crate::FilterKind::Log => Ok(LogImageFilter::new().apply(&image)),
                crate::FilterKind::Exp => Ok(ExpImageFilter::new().apply(&image)),
                crate::FilterKind::MorphologicalGradient { radius } => {
                    GrayscaleMorphologicalGradientFilter::new(*radius).apply(&image)
                }
                crate::FilterKind::DistanceTransform { threshold } => {
                    ritk_filter::DistanceTransformImageFilter::new()
                        .with_threshold(*threshold)
                        .apply(&image)
                }
                crate::FilterKind::SignedDistanceTransform { threshold } => {
                    ritk_filter::SignedDistanceTransformImageFilter::new()
                        .with_threshold(*threshold)
                        .apply(&image)
                }
                crate::FilterKind::FlipZ => ritk_filter::FlipImageFilter::flip_z().apply(&image),
                crate::FilterKind::FlipY => ritk_filter::FlipImageFilter::flip_y().apply(&image),
                crate::FilterKind::FlipX => ritk_filter::FlipImageFilter::flip_x().apply(&image),
                crate::FilterKind::MaskThreshold { threshold } => {
                    let dims = image.shape();
                    match image.try_data_vec() {
                        Ok(vals) => {
                            let mask_vals: Vec<f32> = vals
                                .iter()
                                .map(|&v| {
                                    if v > f32::from(*threshold) {
                                        1.0_f32
                                    } else {
                                        0.0_f32
                                    }
                                })
                                .collect();
                            let device = image.data().device();
                            let mask_td = ritk_image::tensor::TensorData::new(
                                mask_vals,
                                ritk_image::tensor::Shape::new(dims),
                            );
                            let mask_tensor =
                                ritk_image::tensor::Tensor::<LoadBackend, 3>::from_data(
                                    mask_td, &device,
                                );
                            let mask_image = ritk_image::Image::new(
                                mask_tensor,
                                *image.origin(),
                                *image.spacing(),
                                *image.direction(),
                            );
                            ritk_filter::MaskImageFilter::new().apply(&image, &mask_image)
                        }
                        Err(error) => Err(anyhow::anyhow!(
                            "cannot construct MaskThreshold input from the loaded image: {error:?}"
                        )),
                    }
                }
                crate::FilterKind::GeodesicDilationSelf => {
                    ritk_filter::GrayscaleGeodesicDilationFilter::new().apply(&image, &image)
                }
                crate::FilterKind::GeodesicErosionSelf => {
                    ritk_filter::GrayscaleGeodesicErosionFilter::new().apply(&image, &image)
                }
                crate::FilterKind::ShiftScale { shift, scale } => {
                    ritk_filter::ShiftScaleImageFilter::new(*shift, *scale).apply(&image)
                }
                crate::FilterKind::ZeroCrossing {
                    foreground_value,
                    background_value,
                } => ritk_filter::ZeroCrossingImageFilter::new()
                    .with_foreground(*foreground_value)
                    .with_background(*background_value)
                    .apply(&image),
                crate::FilterKind::RegionOfInterest {
                    start_z,
                    start_y,
                    start_x,
                    size_z,
                    size_y,
                    size_x,
                } => ritk_filter::RegionOfInterestImageFilter::new(
                    [*start_z, *start_y, *start_x],
                    [*size_z, *size_y, *size_x],
                )
                .apply(&image),
                crate::FilterKind::PermuteAxes {
                    order_0,
                    order_1,
                    order_2,
                } => ritk_filter::PermuteAxesImageFilter::new([*order_0, *order_1, *order_2])
                    .apply(&image),
                crate::FilterKind::Mean { radius } => {
                    ritk_filter::MeanImageFilter::new(*radius).apply(&image)
                }
                crate::FilterKind::BinaryContour {
                    connectivity,
                    foreground_value,
                } => ritk_filter::BinaryContourImageFilter::new(*connectivity, *foreground_value)
                    .apply(&image),
                crate::FilterKind::LabelContour {
                    connectivity,
                    background_value,
                } => ritk_filter::LabelContourImageFilter::new(*connectivity, *background_value)
                    .apply(&image),
                crate::FilterKind::VotingBinary {
                    radius,
                    birth_threshold,
                    survival_threshold,
                    foreground_value,
                    background_value,
                } => ritk_filter::VotingBinaryImageFilter::new(
                    *radius,
                    *birth_threshold,
                    *survival_threshold,
                    *foreground_value,
                    *background_value,
                )
                .apply(&image),
                crate::FilterKind::Shrink {
                    factor_z,
                    factor_y,
                    factor_x,
                } => ritk_filter::TileMeanShrinkFilter::new([*factor_z, *factor_y, *factor_x])
                    .apply(&image),
                crate::FilterKind::ConstantPad {
                    pad_lower_z,
                    pad_lower_y,
                    pad_lower_x,
                    pad_upper_z,
                    pad_upper_y,
                    pad_upper_x,
                    constant,
                } => ritk_filter::ConstantPadImageFilter::new(
                    ritk_filter::Padding::new([*pad_lower_z, *pad_lower_y, *pad_lower_x]),
                    ritk_filter::Padding::new([*pad_upper_z, *pad_upper_y, *pad_upper_x]),
                    *constant,
                )
                .apply(&image),
                crate::FilterKind::MirrorPad {
                    pad_lower_z,
                    pad_lower_y,
                    pad_lower_x,
                    pad_upper_z,
                    pad_upper_y,
                    pad_upper_x,
                } => ritk_filter::MirrorPadImageFilter::new(
                    ritk_filter::Padding::new([*pad_lower_z, *pad_lower_y, *pad_lower_x]),
                    ritk_filter::Padding::new([*pad_upper_z, *pad_upper_y, *pad_upper_x]),
                )
                .apply(&image),
                crate::FilterKind::WrapPad {
                    pad_lower_z,
                    pad_lower_y,
                    pad_lower_x,
                    pad_upper_z,
                    pad_upper_y,
                    pad_upper_x,
                } => ritk_filter::WrapPadImageFilter::new(
                    ritk_filter::Padding::new([*pad_lower_z, *pad_lower_y, *pad_lower_x]),
                    ritk_filter::Padding::new([*pad_upper_z, *pad_upper_y, *pad_upper_x]),
                )
                .apply(&image),
                crate::FilterKind::GrayscaleErode { radius } => {
                    ritk_filter::GrayscaleErosion::new(*radius).apply(&image)
                }
                crate::FilterKind::GrayscaleDilate { radius } => {
                    ritk_filter::GrayscaleDilation::new(*radius).apply(&image)
                }
                crate::FilterKind::BinaryThreshold {
                    lower,
                    upper,
                    foreground,
                    background,
                } => Ok(ritk_segmentation::binary_threshold(
                    &image,
                    *lower,
                    *upper,
                    (*foreground).into(),
                    *background,
                )),
                crate::FilterKind::RescaleIntensity { out_min, out_max } => {
                    ritk_filter::RescaleIntensityFilter::new(*out_min, *out_max).apply(&image)
                }
                crate::FilterKind::Clamp { lower, upper } => {
                    ritk_filter::ClampImageFilter::new(*lower, *upper).apply(&image)
                }
                crate::FilterKind::ConnectedThreshold {
                    seed_z,
                    seed_y,
                    seed_x,
                    lower,
                    upper,
                } => Ok(
                    ritk_segmentation::region_growing::ConnectedThresholdFilter::new(
                        [*seed_z, *seed_y, *seed_x],
                        *lower,
                        *upper,
                    )
                    .apply(&image),
                ),
                crate::FilterKind::ConfidenceConnected {
                    seed_z,
                    seed_y,
                    seed_x,
                    initial_lower,
                    initial_upper,
                    multiplier,
                    max_iterations,
                } => ritk_segmentation::region_growing::ConfidenceConnectedFilter::new(
                    [*seed_z, *seed_y, *seed_x],
                    *initial_lower,
                    *initial_upper,
                )
                .with_multiplier(*multiplier)
                .map(|filter| {
                    filter
                        .with_max_iterations(*max_iterations as usize)
                        .apply(&image)
                }),
                crate::FilterKind::NeighborhoodConnected {
                    seed_z,
                    seed_y,
                    seed_x,
                    lower,
                    upper,
                    radius_z,
                    radius_y,
                    radius_x,
                } => Ok(
                    ritk_segmentation::region_growing::NeighborhoodConnectedFilter::new(
                        [*seed_z, *seed_y, *seed_x],
                        *lower,
                        *upper,
                    )
                    .with_radius([*radius_z, *radius_y, *radius_x])
                    .apply(&image),
                ),
                crate::FilterKind::Atan => Ok(ritk_filter::AtanImageFilter::new().apply(&image)),
                crate::FilterKind::Sin => Ok(ritk_filter::SinImageFilter::new().apply(&image)),
                crate::FilterKind::Cos => Ok(ritk_filter::CosImageFilter::new().apply(&image)),
                crate::FilterKind::Tan => Ok(ritk_filter::TanImageFilter::new().apply(&image)),
                crate::FilterKind::Asin => Ok(ritk_filter::AsinImageFilter::new().apply(&image)),
                crate::FilterKind::Acos => Ok(ritk_filter::AcosImageFilter::new().apply(&image)),
                crate::FilterKind::BoundedReciprocal => {
                    Ok(ritk_filter::BoundedReciprocalImageFilter::new().apply(&image))
                }
                crate::FilterKind::CurvatureFlow {
                    iterations,
                    time_step,
                } => ritk_filter::CurvatureFlowImageFilter::new(ritk_filter::CurvatureFlowConfig {
                    num_iterations: *iterations as usize,
                    time_step: *time_step,
                })
                .apply(&image),
                crate::FilterKind::Cpr { .. } => {
                    unreachable!("invariant: CPR returns from the native dispatcher")
                }
            }
        };

        match filter_result {
            Err(e) => {
                self.status_message = format!("Filter failed: {e:#}");
            }
            Ok(out_img) => {
                let out_td = out_img.into_tensor().into_data();
                let out_vec: Vec<f32> = out_td.as_slice::<f32>().unwrap_or(&[]).to_vec();
                self.replace_loaded_volume_data(out_vec);
            }
        }
    }

    fn replace_loaded_volume_data(&mut self, data: Vec<f32>) {
        let volume = self
            .loaded
            .as_mut()
            .expect("invariant: a filter result exists only when a volume is loaded");
        volume.data = std::sync::Arc::new(data);
        self.mark_filter_applied();
    }

    fn replace_loaded_volume_native(&mut self, output: native::NativeFilterOutput) {
        let volume = self
            .loaded
            .as_mut()
            .expect("invariant: a filter result exists only when a volume is loaded");
        volume.data = std::sync::Arc::new(output.data);
        volume.shape = output.shape;
        volume.origin = output.origin;
        volume.spacing = output.spacing;
        volume.direction = output.direction;
        self.mark_filter_applied();
    }

    fn mark_filter_applied(&mut self) {
        self.texture_dirty = true;
        self.coronal_dirty = true;
        self.sagittal_dirty = true;
        self.mip_dirty = true;
        self.status_message = "Filter applied.".to_owned();
    }
}
