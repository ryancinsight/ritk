use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use ritk_core::spatial::{Point, Spacing, Direction};
use anyhow::{Result, ensure};

/// A dense vector field representing displacement in physical space.
/// 
/// Unlike a scalar Image, this holds a vector at each voxel.
/// The tensor shape is [Channels, D, H, W] where Channels = 3 for 3D.
/// The vector components are in physical space units (mm), not indices.
#[derive(Debug, Clone)]
pub struct DisplacementField<B: Backend> {
    /// Data tensor of shape [3, D, H, W] (Z, Y, X spatial ordering)
    pub data: Tensor<B, 4>,
    pub origin: Point<3>,
    pub spacing: Spacing<3>,
    pub direction: Direction<3>,
}

impl<B: Backend> DisplacementField<B> {
    pub fn new(data: Tensor<B, 4>, origin: Point<3>, spacing: Spacing<3>, direction: Direction<3>) -> Self {
        // Validate shape
        let dims = data.shape().dims;
        assert_eq!(dims[0], 3, "DisplacementField must have 3 channels");
        Self { data, origin, spacing, direction }
    }
    
    /// Create from a model output tensor [Batch, 3, D, H, W].
    /// Returns a vector of DisplacementFields, one for each item in the batch.
    /// 
    /// Requires a reference image to establish physical space.
    pub fn from_batch(batch: Tensor<B, 5>, reference: &Image<B, 3>) -> Vec<Self> {
        let dims = batch.shape().dims;
        let (b, c, d, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        
        assert_eq!(c, 3, "Batch must have 3 channels");
        let ref_shape = reference.shape();
        assert_eq!([d, h, w], [ref_shape[0], ref_shape[1], ref_shape[2]], "Batch spatial dims must match reference image");

        let mut fields = Vec::with_capacity(b);
        // Iterate over batch dimension
        // Note: Burn doesn't have a simple "split to vec" so we slice
        for i in 0..b {
            let slice = batch.clone().slice([i..i+1]); // [1, 3, D, H, W]
            let squeezed = slice.squeeze::<4>(0); // [3, D, H, W]
            
            fields.push(Self::new(
                squeezed,
                *reference.origin(),
                *reference.spacing(),
                *reference.direction()
            ));
        }
        fields
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
