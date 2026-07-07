use crate::filter::promote::elevate_to_volume;
use crate::filter::FilterKind;
use crate::viewer::{Study, ViewerCore, ViewerEvent};
use ritk_filter::{
    AbsImageFilter, AcosImageFilter, AsinImageFilter, AtanImageFilter, BedSeparationFilter,
    BinaryContourImageFilter, BinaryDilateFilter, BinaryErodeFilter, BinaryFillholeFilter,
    BinaryMorphologicalClosing, BinaryMorphologicalOpening, BinaryThresholdImageFilter,
    BoundedReciprocalImageFilter, ClaheFilter, ClampImageFilter, Connectivity,
    ConstantPadImageFilter, CosImageFilter, CprConfig, CprImageFilter, CurvatureFlowConfig,
    CurvatureFlowImageFilter, DistanceTransformImageFilter, ExpImageFilter, FlipImageFilter,
    GaussianFilter, GaussianSigma, GradientAnisotropicDiffusionFilter, GradientDiffusionConfig,
    GrayscaleClosingFilter, GrayscaleDilation, GrayscaleErosion, GrayscaleFillholeFilter,
    GrayscaleGeodesicDilationFilter, GrayscaleGeodesicErosionFilter,
    GrayscaleMorphologicalGradientFilter, GrayscaleOpeningFilter, HistogramEqualizationFilter,
    InvertIntensityFilter, LabelContourImageFilter, LogImageFilter, MaskImageFilter,
    MeanImageFilter, MedianFilter, MirrorPadImageFilter, NormalizeImageFilter, Padding,
    PermuteAxesImageFilter, RegionOfInterestImageFilter, RescaleIntensityFilter,
    ShiftScaleImageFilter, SignedDistanceTransformImageFilter, SinImageFilter, SqrtImageFilter,
    SquareImageFilter, TanImageFilter, TileMeanShrinkFilter, UnsharpMaskFilter,
    VotingBinaryImageFilter, WrapPadImageFilter, ZeroCrossingImageFilter,
};
use ritk_image::Image;
use ritk_segmentation::region_growing::{
    ConfidenceConnectedFilter, ConnectedThresholdFilter, NeighborhoodConnectedFilter,
};
use ritk_segmentation::{ConnectedComponentsFilter, MultiOtsuThreshold, RelabelComponentFilter};

impl<B: ritk_image::tensor::Backend> ViewerCore<B, 3> {
    /// Apply a filter to the currently loaded study's image.
    ///
    /// Returns `Ok(Status)` if no study is loaded (no error raised).
    /// On success, `self.study` contains the filtered image with spatial metadata
    /// preserved; the filter name and new shape are in the status message.
    /// On filter failure, the original study is restored and the error is propagated.
    pub fn apply_filter(&mut self, kind: &FilterKind) -> anyhow::Result<ViewerEvent> {
        let study = match self.study.take() {
            None => {
                return Ok(ViewerEvent::Status {
                    message: "no study loaded".to_string(),
                });
            }
            Some(s) => s,
        };

        // Apply the selected filter. The borrow of study.image is released before
        // `filter_result` is consumed below, enabling safe move of study fields.
        let filter_result: anyhow::Result<Image<B, 3>> = match kind {
            FilterKind::BedSeparation(config) => {
                BedSeparationFilter::new(*config).apply(&study.image)
            }
            FilterKind::Gaussian { sigma } => {
                // GaussianFilter takes physical-unit sigmas per dimension.
                // Broadcasting a single sigma across all three axes.
                let s_val = f64::from(*sigma);
                let g_sigma =
                    GaussianSigma::new(s_val).unwrap_or_else(|| GaussianSigma::new_unchecked(1e-9));
                Ok(GaussianFilter::<B>::new(vec![g_sigma; 3]).apply(&study.image))
            }
            FilterKind::Median { radius } => MedianFilter::new(*radius).apply(&study.image),
            FilterKind::Clahe {
                tile_grid_size,
                clip_limit,
            } => ClaheFilter::new(*tile_grid_size, *clip_limit, 256).apply(&study.image),
            FilterKind::HistEq { bins } => {
                HistogramEqualizationFilter::new(*bins).apply(&study.image)
            }
            FilterKind::UnsharpMask {
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
            .apply(&study.image),
            FilterKind::GradientAnisotropicDiffusion {
                iterations,
                time_step,
                conductance,
            } => GradientAnisotropicDiffusionFilter::new(GradientDiffusionConfig {
                num_iterations: *iterations as usize,
                time_step: *time_step,
                conductance: *conductance,
            })
            .apply(&study.image),
            FilterKind::ConnectedComponents {
                connectivity,
                background_value,
            } => {
                let seg_connectivity = match connectivity {
                    Connectivity::Face6 => ritk_segmentation::labeling::Connectivity::Six,
                    Connectivity::Vertex26 => ritk_segmentation::labeling::Connectivity::TwentySix,
                };
                let filter = ConnectedComponentsFilter::with_connectivity(seg_connectivity)
                    .with_background(*background_value);
                let (label_image, _stats) = filter.apply(&study.image);
                Ok(label_image)
            }
            FilterKind::RelabelComponents {
                minimum_object_size,
            } => {
                let (relabeled, _stats) =
                    RelabelComponentFilter::with_minimum_object_size(*minimum_object_size as usize)
                        .apply(&study.image);
                Ok(relabeled)
            }
            FilterKind::MultiOtsuThreshold { num_classes } => {
                Ok(MultiOtsuThreshold::new(*num_classes as usize).apply(&study.image))
            }
            FilterKind::BinaryErode {
                radius,
                foreground_value,
            } => BinaryErodeFilter::new(*radius)
                .with_foreground(*foreground_value)
                .apply(&study.image),
            FilterKind::BinaryDilate {
                radius,
                foreground_value,
            } => BinaryDilateFilter::new(*radius)
                .with_foreground(*foreground_value)
                .apply(&study.image),
            FilterKind::BinaryClosing {
                radius,
                foreground_value,
            } => BinaryMorphologicalClosing::new(*radius)
                .with_foreground(*foreground_value)
                .apply(&study.image),
            FilterKind::BinaryOpening {
                radius,
                foreground_value,
            } => BinaryMorphologicalOpening::new(*radius)
                .with_foreground(*foreground_value)
                .apply(&study.image),
            FilterKind::BinaryFillhole { foreground_value } => BinaryFillholeFilter::new()
                .with_foreground(*foreground_value)
                .apply(&study.image),
            FilterKind::GrayscaleClosing { radius } => {
                GrayscaleClosingFilter::new(*radius).apply(&study.image)
            }
            FilterKind::GrayscaleOpening { radius } => {
                GrayscaleOpeningFilter::new(*radius).apply(&study.image)
            }
            FilterKind::GrayscaleFillhole => GrayscaleFillholeFilter::new().apply(&study.image),
            FilterKind::Abs => Ok(AbsImageFilter::new().apply(&study.image)),
            FilterKind::InvertIntensity { maximum } => Ok(match maximum {
                Some(m) => InvertIntensityFilter::with_maximum(*m).apply(&study.image),
                None => InvertIntensityFilter::new().apply(&study.image),
            }),
            FilterKind::NormalizeIntensity => Ok(NormalizeImageFilter::new().apply(&study.image)),
            FilterKind::Square => Ok(SquareImageFilter::new().apply(&study.image)),
            FilterKind::Sqrt => Ok(SqrtImageFilter::new().apply(&study.image)),
            FilterKind::Log => Ok(LogImageFilter::new().apply(&study.image)),
            FilterKind::Exp => Ok(ExpImageFilter::new().apply(&study.image)),
            FilterKind::MorphologicalGradient { radius } => {
                GrayscaleMorphologicalGradientFilter::new(*radius).apply(&study.image)
            }
            FilterKind::DistanceTransform { threshold } => DistanceTransformImageFilter::new()
                .with_threshold(*threshold)
                .apply(&study.image),
            FilterKind::SignedDistanceTransform { threshold } => {
                SignedDistanceTransformImageFilter::new()
                    .with_threshold(*threshold)
                    .apply(&study.image)
            }
            FilterKind::FlipZ => FlipImageFilter::flip_z().apply(&study.image),
            FilterKind::FlipY => FlipImageFilter::flip_y().apply(&study.image),
            FilterKind::FlipX => FlipImageFilter::flip_x().apply(&study.image),
            FilterKind::MaskThreshold { threshold } => {
                // Build a binary mask from the current image: voxels > threshold → 1, else 0.
                // Then apply MaskImageFilter to zero out subthreshold voxels.
                let threshold_f32 = f32::from(*threshold);
                let dims = study.image.shape();
                let vals: Vec<f32> = study
                    .image
                    .try_data_vec()
                    .map_err(|e| anyhow::anyhow!("MaskThreshold: f32 required: {:?}", e))?;
                let mask_vals: Vec<f32> = vals
                    .iter()
                    .map(|&v| if v > threshold_f32 { 1.0_f32 } else { 0.0_f32 })
                    .collect();
                let device = study.image.data().device();
                let mask_td = ritk_image::tensor::TensorData::new(
                    mask_vals,
                    ritk_image::tensor::Shape::new(dims),
                );
                let mask_tensor = ritk_image::tensor::Tensor::<B, 3>::from_data(mask_td, &device);
                let mask_image = Image::new(
                    mask_tensor,
                    *study.image.origin(),
                    *study.image.spacing(),
                    *study.image.direction(),
                );
                MaskImageFilter::new().apply(&study.image, &mask_image)
            }
            FilterKind::GeodesicDilationSelf => {
                // Use current image as both marker and mask (no-op reconstruction)
                GrayscaleGeodesicDilationFilter::new().apply(&study.image, &study.image)
            }
            FilterKind::GeodesicErosionSelf => {
                GrayscaleGeodesicErosionFilter::new().apply(&study.image, &study.image)
            }
            FilterKind::ShiftScale { shift, scale } => {
                ShiftScaleImageFilter::new(*shift, *scale).apply(&study.image)
            }
            FilterKind::ZeroCrossing {
                foreground_value,
                background_value,
            } => ZeroCrossingImageFilter::new()
                .with_foreground(*foreground_value)
                .with_background(*background_value)
                .apply(&study.image),
            FilterKind::RegionOfInterest {
                start_z,
                start_y,
                start_x,
                size_z,
                size_y,
                size_x,
            } => RegionOfInterestImageFilter::new(
                [*start_z, *start_y, *start_x],
                [*size_z, *size_y, *size_x],
            )
            .apply(&study.image),
            FilterKind::PermuteAxes {
                order_0,
                order_1,
                order_2,
            } => PermuteAxesImageFilter::new([*order_0, *order_1, *order_2]).apply(&study.image),
            FilterKind::Mean { radius } => MeanImageFilter::new(*radius).apply(&study.image),
            FilterKind::BinaryContour {
                connectivity,
                foreground_value,
            } => {
                BinaryContourImageFilter::new(*connectivity, *foreground_value).apply(&study.image)
            }
            FilterKind::LabelContour {
                connectivity,
                background_value,
            } => LabelContourImageFilter::new(*connectivity, *background_value).apply(&study.image),
            FilterKind::VotingBinary {
                radius,
                birth_threshold,
                survival_threshold,
                foreground_value,
                background_value,
            } => VotingBinaryImageFilter::new(
                *radius,
                *birth_threshold,
                *survival_threshold,
                *foreground_value,
                *background_value,
            )
            .apply(&study.image),
            FilterKind::Shrink {
                factor_z,
                factor_y,
                factor_x,
            } => TileMeanShrinkFilter::new([*factor_z, *factor_y, *factor_x]).apply(&study.image),
            FilterKind::ConstantPad {
                pad_lower_z,
                pad_lower_y,
                pad_lower_x,
                pad_upper_z,
                pad_upper_y,
                pad_upper_x,
                constant,
            } => ConstantPadImageFilter::new(
                Padding::new([*pad_lower_z, *pad_lower_y, *pad_lower_x]),
                Padding::new([*pad_upper_z, *pad_upper_y, *pad_upper_x]),
                *constant,
            )
            .apply(&study.image),
            FilterKind::MirrorPad {
                pad_lower_z,
                pad_lower_y,
                pad_lower_x,
                pad_upper_z,
                pad_upper_y,
                pad_upper_x,
            } => MirrorPadImageFilter::new(
                Padding::new([*pad_lower_z, *pad_lower_y, *pad_lower_x]),
                Padding::new([*pad_upper_z, *pad_upper_y, *pad_upper_x]),
            )
            .apply(&study.image),
            FilterKind::WrapPad {
                pad_lower_z,
                pad_lower_y,
                pad_lower_x,
                pad_upper_z,
                pad_upper_y,
                pad_upper_x,
            } => WrapPadImageFilter::new(
                Padding::new([*pad_lower_z, *pad_lower_y, *pad_lower_x]),
                Padding::new([*pad_upper_z, *pad_upper_y, *pad_upper_x]),
            )
            .apply(&study.image),
            FilterKind::GrayscaleErode { radius } => {
                GrayscaleErosion::new(*radius).apply(&study.image)
            }
            FilterKind::GrayscaleDilate { radius } => {
                GrayscaleDilation::new(*radius).apply(&study.image)
            }
            FilterKind::BinaryThreshold {
                lower,
                upper,
                foreground,
                background,
            } => BinaryThresholdImageFilter::new(*lower, *upper, *foreground, *background)
                .apply(&study.image),
            FilterKind::RescaleIntensity { out_min, out_max } => {
                RescaleIntensityFilter::new(*out_min, *out_max).apply(&study.image)
            }
            FilterKind::Clamp { lower, upper } => {
                ClampImageFilter::new(*lower, *upper).apply(&study.image)
            }
            FilterKind::ConnectedThreshold {
                seed_z,
                seed_y,
                seed_x,
                lower,
                upper,
            } => Ok(
                ConnectedThresholdFilter::new([*seed_z, *seed_y, *seed_x], *lower, *upper)
                    .apply(&study.image),
            ),
            FilterKind::ConfidenceConnected {
                seed_z,
                seed_y,
                seed_x,
                initial_lower,
                initial_upper,
                multiplier,
                max_iterations,
            } => Ok(ConfidenceConnectedFilter::new(
                [*seed_z, *seed_y, *seed_x],
                *initial_lower,
                *initial_upper,
            )
            .with_multiplier(*multiplier)
            .with_max_iterations(*max_iterations as usize)
            .apply(&study.image)),
            FilterKind::NeighborhoodConnected {
                seed_z,
                seed_y,
                seed_x,
                lower,
                upper,
                radius_z,
                radius_y,
                radius_x,
            } => Ok(
                NeighborhoodConnectedFilter::new([*seed_z, *seed_y, *seed_x], *lower, *upper)
                    .with_radius([*radius_z, *radius_y, *radius_x])
                    .apply(&study.image),
            ),
            FilterKind::Atan => Ok(AtanImageFilter::new().apply(&study.image)),
            FilterKind::Sin => Ok(SinImageFilter::new().apply(&study.image)),
            FilterKind::Cos => Ok(CosImageFilter::new().apply(&study.image)),
            FilterKind::Tan => Ok(TanImageFilter::new().apply(&study.image)),
            FilterKind::Asin => Ok(AsinImageFilter::new().apply(&study.image)),
            FilterKind::Acos => Ok(AcosImageFilter::new().apply(&study.image)),
            FilterKind::BoundedReciprocal => {
                Ok(BoundedReciprocalImageFilter::new().apply(&study.image))
            }
            FilterKind::CurvatureFlow {
                iterations,
                time_step,
            } => CurvatureFlowImageFilter::new(CurvatureFlowConfig {
                num_iterations: *iterations as usize,
                time_step: *time_step,
            })
            .apply(&study.image),
            FilterKind::Cpr {
                control_points,
                num_path_samples,
                cross_section_half_width,
                num_cross_samples,
            } => {
                let cpr_filter = CprImageFilter::new(
                    control_points.clone(),
                    CprConfig {
                        num_path_samples: *num_path_samples as usize,
                        cross_section_half_width: f64::from(*cross_section_half_width),
                        num_cross_samples: *num_cross_samples as usize,
                    },
                );
                let image_2d = cpr_filter.apply(&study.image)?;
                elevate_to_volume(image_2d)
            }
        };

        let filter_name = match kind {
            FilterKind::BedSeparation(_) => "BedSeparation",
            FilterKind::Gaussian { .. } => "Gaussian",
            FilterKind::Median { .. } => "Median",
            FilterKind::Clahe { .. } => "CLAHE",
            FilterKind::HistEq { .. } => "HistEq",
            FilterKind::UnsharpMask { .. } => "UnsharpMask",
            FilterKind::GradientAnisotropicDiffusion { .. } => "GradientAnisotropicDiffusion",
            FilterKind::ConnectedComponents { .. } => "ConnectedComponents",
            FilterKind::RelabelComponents { .. } => "RelabelComponents",
            FilterKind::MultiOtsuThreshold { .. } => "MultiOtsuThreshold",
            FilterKind::BinaryErode { .. } => "BinaryErode",
            FilterKind::BinaryDilate { .. } => "BinaryDilate",
            FilterKind::BinaryClosing { .. } => "BinaryClosing",
            FilterKind::BinaryOpening { .. } => "BinaryOpening",
            FilterKind::BinaryFillhole { .. } => "BinaryFillhole",
            FilterKind::GrayscaleClosing { .. } => "GrayscaleClosing",
            FilterKind::GrayscaleOpening { .. } => "GrayscaleOpening",
            FilterKind::GrayscaleFillhole => "GrayscaleFillhole",
            FilterKind::Abs => "Abs",
            FilterKind::InvertIntensity { .. } => "InvertIntensity",
            FilterKind::NormalizeIntensity => "NormalizeIntensity",
            FilterKind::Square => "Square",
            FilterKind::Sqrt => "Sqrt",
            FilterKind::Log => "Log",
            FilterKind::Exp => "Exp",
            FilterKind::MorphologicalGradient { .. } => "MorphologicalGradient",
            FilterKind::DistanceTransform { .. } => "DistanceTransform",
            FilterKind::SignedDistanceTransform { .. } => "SignedDistanceTransform",
            FilterKind::FlipZ => "FlipZ",
            FilterKind::FlipY => "FlipY",
            FilterKind::FlipX => "FlipX",
            FilterKind::MaskThreshold { .. } => "MaskThreshold",
            FilterKind::GeodesicDilationSelf => "GeodesicDilationSelf",
            FilterKind::GeodesicErosionSelf => "GeodesicErosionSelf",
            FilterKind::ShiftScale { .. } => "ShiftScale",
            FilterKind::ZeroCrossing { .. } => "ZeroCrossing",
            FilterKind::RegionOfInterest { .. } => "RegionOfInterest",
            FilterKind::PermuteAxes { .. } => "PermuteAxes",
            FilterKind::Mean { .. } => "Mean",
            FilterKind::BinaryContour { .. } => "BinaryContour",
            FilterKind::LabelContour { .. } => "LabelContour",
            FilterKind::VotingBinary { .. } => "VotingBinary",
            FilterKind::Shrink { .. } => "Shrink",
            FilterKind::ConstantPad { .. } => "ConstantPad",
            FilterKind::MirrorPad { .. } => "MirrorPad",
            FilterKind::WrapPad { .. } => "WrapPad",
            FilterKind::GrayscaleErode { .. } => "GrayscaleErode",
            FilterKind::GrayscaleDilate { .. } => "GrayscaleDilate",
            FilterKind::BinaryThreshold { .. } => "BinaryThreshold",
            FilterKind::RescaleIntensity { .. } => "RescaleIntensity",
            FilterKind::Clamp { .. } => "Clamp",
            FilterKind::ConnectedThreshold { .. } => "ConnectedThreshold",
            FilterKind::ConfidenceConnected { .. } => "ConfidenceConnected",
            FilterKind::NeighborhoodConnected { .. } => "NeighborhoodConnected",
            FilterKind::Atan => "Atan",
            FilterKind::Sin => "Sin",
            FilterKind::Cos => "Cos",
            FilterKind::Tan => "Tan",
            FilterKind::Asin => "Asin",
            FilterKind::Acos => "Acos",
            FilterKind::BoundedReciprocal => "BoundedReciprocal",
            FilterKind::CurvatureFlow { .. } => "CurvatureFlow",
            FilterKind::Cpr { .. } => "CPR",
        };

        match filter_result {
            Err(e) => {
                // Restore the original study so the core remains usable after a
                // filter error.
                self.study = Some(study);
                Err(e)
            }
            Ok(new_image) => {
                let shape = new_image.shape();
                self.study = Some(Study {
                    image: new_image,
                    dicom: study.dicom,
                    source: study.source,
                });
                Ok(ViewerEvent::Status {
                    message: format!("{filter_name} applied; shape {:?}", shape),
                })
            }
        }
    }
}
