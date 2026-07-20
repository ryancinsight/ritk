# Summary

[Introduction](README.md)

# Part I — Image I/O and Format Boundaries

- [DICOM Format Boundary](dicom_format.md)
  - [Example: DICOM to NIfTI Conversion](examples/dicom_to_nifti.md)
  - [Example: DICOM Dump Utility](examples/dump_dicom.md)
- [NIfTI Format Boundary](nifti_format.md)
- [NRRD Format Boundary](nrrd_format.md)
- [MetaImage Format Boundary](metaimage_format.md)
- [PNG Format Boundary](png_format.md)
- [JPEG Format Boundary](jpeg_format.md)
- [VTK Format Boundary](vtk_format.md)

# Part II — Image Processing Pipeline

- [Intensity Transformations](intensity_transforms.md)
  - [Example: Windowing and Rescaling](examples/windowing_rescale.md)
  - [Example: Thresholding](examples/thresholding.md)
  - [Example: Sigmoid and Arithmetic](examples/sigmoid_arithmetic.md)
- [Spatial Filtering](spatial_filters.md)
  - [Example: Gaussian Smoothing](examples/gaussian_smoothing.md)
  - [Example: Gradient Magnitude](examples/gradient_magnitude.md)
  - [Example: Canny Edge Detection](examples/canny_edges.md)
- [Morphological Operations](morphology.md)
  - [Example: Binary Erosion/Dilation](examples/binary_morphology.md)
  - [Example: Grayscale Opening/Closing](examples/grayscale_morphology.md)
- [Diffusion Filtering](diffusion_filters.md)
  - [Example: Perona-Malik Diffusion](examples/perona_malik.md)
  - [Example: Curvature Flow](examples/curvature_flow.md)
- [Registration Metrics](registration_metrics.md)
- [Optimization and Registration](optimization_registration.md)

# Part III — Registration Algorithms

- [Classical Registration](classical_registration.md)
  - [Example: Geometry Validation](examples/geometry_check.md)
  - [Example: DL Registration](examples/dl_registration.md)
  - [Example: DL Training](examples/dl_train.md)
- [Multi-modal Registration](multi_modal_registration.md)
  - [Example: Registration Comparison Figure](examples/registration_compare_figure.md)
- [Validation and Benchmarking](validation_benchmarking.md)
  - [Example: Validation Suite](examples/validation_suite.md)

# Part IV — Performance and Low-level Optimizations

- [Benchmarking](benchmarking.md)
  - [Example: Gradient Recursive Gaussian Benchmark](examples/bench_gradient_rg.md)
- [Backend Dispatch](backend_dispatch.md)
- [Zero-copy I/O](zero_copy_io.md)

# Part V — Integration with atlas Foundation

- [Coeus Tensor Integration](coeus_integration.md)
- [Leto Operations Integration](leto_integration.md)
- [Moirai Execution Integration](moirai_integration.md)
