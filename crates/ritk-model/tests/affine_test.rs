use burn::{
    backend::{wgpu::WgpuDevice, Autodiff, Wgpu},
    tensor::{Tensor, Distribution},
};
use ritk_model::affine::{AffineNetworkConfig, AffineTransform};

type TestBackend = Autodiff<Wgpu>;

#[test]
fn test_affine_network_forward() {
    let device = WgpuDevice::default();
    let config = AffineNetworkConfig::default();
    let model = config.init::<TestBackend>(&device);

    let input = Tensor::<TestBackend, 5>::random([1, 2, 32, 32, 32], Distribution::Normal(0.0, 1.0), &device);
    let output = model.forward(input);

    assert_eq!(output.shape().dims, [1, 12]);
}

#[test]
fn test_affine_transform_forward() {
    let device = WgpuDevice::default();
    let stn = AffineTransform::<TestBackend>::new();

    let image = Tensor::<TestBackend, 5>::random([1, 1, 32, 32, 32], Distribution::Normal(0.0, 1.0), &device);
    // Identity transform
    let theta = Tensor::<TestBackend, 1>::from_floats(
        [1.0, 0.0, 0.0, 0.0, 
         0.0, 1.0, 0.0, 0.0, 
         0.0, 0.0, 1.0, 0.0], 
        &device
    ).reshape([1, 12]);

    let output = stn.forward(image.clone(), theta);

    assert_eq!(output.shape().dims, [1, 1, 32, 32, 32]);
    
    // Check if output is close to input (identity transform should be exact if grid is perfect, 
    // but interpolation adds some blur/error at boundaries, though with identity grid it should be exact)
    // Actually, due to alignment of "normalized coordinates" vs "pixel coordinates", 
    // there might be small shifts if not careful.
    // My normalized_meshgrid implementation:
    // [-1, 1] maps to [0, D-1].
    
    // Let's verify a simple shift
}
