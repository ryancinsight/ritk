use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use ritk_core::image::Image;
// use ritk_core::spatial::{Point, Spacing, Direction};
use ritk_core::transform::displacement_field::DisplacementField;
use anyhow::{Result, ensure};

/// Adapter for converting between ritk Images and Burn tensors
pub struct ImageToTensorAdapter<B: Backend> {
    device: B::Device,
}

impl<B: Backend> ImageToTensorAdapter<B> {
    /// Create new adapter
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }

    /// Get the device
    pub fn device(&self) -> B::Device {
        self.device.clone()
    }

    /// Convert 3D image to tensor [1, 1, D, H, W]
    pub fn image_to_tensor_3d(&self, image: &Image<B, 3>) -> Result<Tensor<B, 5>> {
        let dims = image.shape();
        let (d, h, w) = (dims[0], dims[1], dims[2]);
        // Ensure data is on the correct device
        let data = image.data().clone().to_device(&self.device);
        Ok(data.reshape([1, 1, d, h, w]))
    }

    /// Convert tensor [1, 3, D, H, W] to DisplacementField3D
    pub fn tensor_to_displacement_field_3d(
        &self,
        tensor: &Tensor<B, 5>,
        reference: &Image<B, 3>,
    ) -> Result<DisplacementField<B, 3>> {
        let [batch, channels, d, h, w] = tensor.dims();
        ensure!(batch == 1, "Batch size must be 1");
        ensure!(channels == 3, "Displacement field must have 3 channels");
        
        // Extract components
        // Tensor is [1, 3, D, H, W]
        // We want 3 tensors of [D, H, W]
        let x = tensor.clone().slice([0..1, 0..1, 0..d, 0..h, 0..w]).reshape([d, h, w]);
        let y = tensor.clone().slice([0..1, 1..2, 0..d, 0..h, 0..w]).reshape([d, h, w]);
        let z = tensor.clone().slice([0..1, 2..3, 0..d, 0..h, 0..w]).reshape([d, h, w]);
        
        let components = vec![x, y, z];
        
        Ok(DisplacementField::new(
            components,
            reference.origin().clone(),
            reference.spacing().clone(),
            reference.direction().clone(),
        ))
    }
}

/// Converts a batch of Images into a 5D Tensor [Batch, Channels, D, H, W].
///
/// Assumes all images have the same shape and spatial metadata.
/// Channel dimension is set to 1.
pub fn images_to_batch<B: Backend>(images: Vec<Image<B, 3>>) -> Result<Tensor<B, 5>> {
    ensure!(!images.is_empty(), "Cannot batch empty list of images");
    
    let ref_shape = images[0].shape();
    // Validate consistency
    for (i, img) in images.iter().enumerate().skip(1) {
        ensure!(img.shape() == ref_shape, "Image {} shape mismatch: {:?} vs {:?}", i, img.shape(), ref_shape);
    }

    let d = ref_shape[0];
    let h = ref_shape[1];
    let w = ref_shape[2];

    let tensors: Vec<Tensor<B, 5>> = images.into_iter()
        .map(|img| {
            // Image data is [D, H, W]
            // Add Batch and Channel dims: [1, 1, D, H, W]
            img.data().clone().reshape([1, 1, d, h, w])
        })
        .collect();

    // Concatenate along batch dimension (0)
    Ok(Tensor::cat(tensors, 0))
}

/// Helper to convert a single image to a model input tensor [1, 1, D, H, W]
pub fn image_to_tensor<B: Backend>(image: &Image<B, 3>) -> Tensor<B, 5> {
    let dims = image.shape();
    let (d, h, w) = (dims[0], dims[1], dims[2]);
    image.data().clone().reshape([1, 1, d, h, w])
}
