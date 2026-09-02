# Summary

[Introduction](README.md)

# Part I — Image I/O and Format Boundaries

- [DICOM Format Boundary](dicom_format.md)
  - [Example: DICOM to NIfTI Conversion](examples/dicom_to_nifti.md)
  - [Example: DICOM Dump Utility](examples/dump_dicom.md)
- [NIfTI Format Boundary](nifti_format.md)
- [NRRD Format Boundary](nrrd_format.md)
- [MetaImage Format Boundary](metaimage_format.md)
- [Analyze 7.5 Format Boundary](analyze_format.md)
  - [Example: Analyze 7.5 Round Trip](examples/analyze_roundtrip.md)
- [MGH and MGZ Format Boundary](mgh_format.md)
  - [Example: MGH and MGZ Round Trip](examples/mgh_roundtrip.md)
- [MINC2 Format Boundary](minc_format.md)
  - [Example: MINC2 Round Trip](examples/minc_roundtrip.md)
- [TIFF and BigTIFF Format Boundary](tiff_format.md)
  - [Example: Multi-page TIFF Round Trip](examples/tiff_roundtrip.md)
- [PNG Format Boundary](png_format.md)
- [JPEG Format Boundary](jpeg_format.md)
- [JPEG-LS Native Codec](jpeg_ls_codec.md)
  - [Example: Lossless and Near-Lossless Coding](examples/jpeg_ls_codec.md)
- [JPEG 2000 Native Codec](jpeg_2000_codec.md)
  - [Example: Quality and Size](examples/jpeg_2000_codec.md)
- [VTK Format Boundary](vtk_format.md)

# Part II — Image Processing Pipeline

- [Descriptive Statistics and Histograms](descriptive_statistics.md)
  - [Example: Full-image and Masked Distributions](examples/descriptive_statistics.md)
- [Intensity Transformations](intensity_transforms.md)
  - [Example: Windowing and Rescaling](examples/windowing_rescale.md)
  - [Example: Complete Processing Pipeline](examples/processing_pipeline.md)
  - [Example: Thresholding](examples/thresholding.md)
  - [Example: Sigmoid and Arithmetic](examples/sigmoid_arithmetic.md)
  - [Example: N4 Bias-Field Correction](examples/n4_bias_correction.md)
- [Spatial Filtering](spatial_filters.md)
  - [Example: Gaussian Smoothing](examples/gaussian_smoothing.md)
  - [Example: Gradient Magnitude](examples/gradient_magnitude.md)
  - [Example: Canny Edge Detection](examples/canny_edges.md)
- [Morphological Operations](morphology.md)
  - [Example: Binary Opening and Closing](examples/binary_morphology.md)
  - [Example: Grayscale Opening/Closing](examples/grayscale_morphology.md)
- [Diffusion Filtering](diffusion_filters.md)
  - [Example: Perona-Malik Diffusion](examples/perona_malik.md)
  - [Example: Curvature Flow](examples/curvature_flow.md)
- [Seeded Segmentation](segmentation.md)
  - [Example: GrowCut from Sparse Seeds](examples/growcut.md)
- [Registration Metrics](registration_metrics.md)
- [Optimization and Registration](optimization_registration.md)

# Part III — Diffusion MRI and Tractography

- [Diffusion Gradient Schemes](diffusion_scheme.md)
- [Diffusion Models](ritk_diffusion.md)
- [Diffusion MRI Acquisition and Q-ball ODFs](diffusion_mri.md)
- [Creating and Validating Deterministic Tractography](tractography.md)
  - [Example: Signal to Streamlines](examples/diffusion_tractography.md)
  - [Example: Reusable DTI-Volume Tracking](examples/dti_volume_tractography.md)
  - [Human Tractography and Connectomics](examples/brain_tractography.md)
- [Anatomical Parcellation](parcellation.md)
  - [Example: Atlas Parcellation](examples/atlas_parcellation.md)
- [Connectome Construction and Graph Measures](connectome.md)

## Tractogram Interchange Formats

- [.trk — DSI Studio / TrackVis](trk_format.md)
- [.tck — MRtrix3](tck_format.md)
- [.trx — Tractography Reference eXchange](trx_format.md)

# Part IV — Registration Algorithms

- [Classical Registration](classical_registration.md)
  - [Example: Geometry Validation](examples/geometry_check.md)
  - [Example: DL Registration](examples/dl_registration.md)
  - [Example: DL Training](examples/dl_train.md)
- [Temporal Signal Synchronization](temporal_synchronization.md)
  - [Example: Before and After Temporal Alignment](examples/temporal_synchronization.md)
- [Multi-modal Registration](multi_modal_registration.md)
  - [Example: CT/MR Mutual-Information Registration](examples/registration_compare_figure.md)
- [Deformable Registration](demons_registration.md)
- [Validation and Benchmarking](validation_benchmarking.md)
  - [Example: Validation Suite](examples/validation_suite.md)

# Part V — Performance and Low-level Optimizations

- [Benchmarking](benchmarking.md)
  - [Example: Gradient Recursive Gaussian Benchmark](examples/bench_gradient_rg.md)
- [Backend Dispatch](backend_dispatch.md)
- [Zero-copy I/O](zero_copy_io.md)

# Part VI — Integration with atlas Foundation

- [Coeus Nonlinear Least-Squares Solver](coeus_optim.md)
- [Leto Linear Algebra Operations](leto_linalg.md)
- [Apollo Spherical Harmonic Basis](apollo_sht.md)
- [Gaia Polyline Geometry](gaia_polyline.md)
- [Moirai Parallel Execution Backend](moirai_execution.md)
- [Coeus Tensor Integration](coeus_integration.md)
- [Leto Operations Integration](leto_integration.md)
