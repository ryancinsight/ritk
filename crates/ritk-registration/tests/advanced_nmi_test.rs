use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn_ndarray::NdArray;
use ritk_core::image::Image;
use ritk_core::spatial::{Point, Spacing, Direction};
use ritk_core::transform::TranslationTransform;
use ritk_registration::metric::{Metric, AdvancedNormalizedMutualInformation, NormalizationMethod};
use burn::tensor::{Shape, TensorData};

type B = NdArray<f32>;

fn create_test_image(data: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
    let device = Default::default();
    let tensor = Tensor::from_data(TensorData::new(data, Shape::new(shape)), &device);
    let spacing = Spacing::new([1.0, 1.0, 1.0]);
    let origin = Point::new([0.0, 0.0, 0.0]);
    let direction = Direction::identity();
    Image::new(tensor, origin, spacing, direction)
}

#[test]
fn test_anmi_identical() {
    let shape = [10, 10, 10];
    let size = shape.iter().product();
    // Create random data
    let data: Vec<f32> = (0..size).map(|i| (i % 255) as f32).collect();
    
    let fixed = create_test_image(data.clone(), shape);
    let moving = create_test_image(data, shape);
    
    // Identity transform
    let transform = TranslationTransform::<B, 3>::new(Tensor::zeros([3], &Default::default()));
    
    let metric = AdvancedNormalizedMutualInformation::<B>::new(
        32, 0.0, 255.0, 1.0, NormalizationMethod::JointEntropy
    );
    
    let loss = metric.forward(&fixed, &moving, &transform);
    let loss_val = loss.into_scalar();
    
    println!("ANMI (Identical) Loss: {}", loss_val);
    
    // NMI should be high (close to 1.0 or higher depending on normalization), so loss (negative NMI) should be low (negative).
    // For JointEntropy normalization: NMI = (H(X)+H(Y))/H(X,Y). For identical X=Y, H(X,Y)=H(X).
    // So NMI = 2*H(X)/H(X) = 2.0.
    // Loss should be around -2.0.
    
    assert!(loss_val < -1.0, "ANMI loss for identical images should be significantly negative (NMI > 1.0)");
}

#[test]
fn test_anmi_shifted() {
    let shape = [10, 10, 10];
    let size = shape.iter().product();
    let data: Vec<f32> = (0..size).map(|i| (i % 255) as f32).collect();
    
    let fixed = create_test_image(data.clone(), shape);
    let moving = create_test_image(data, shape);
    
    // Shifted transform
    let device = Default::default();
    let params = Tensor::from_floats([2.0, 2.0, 2.0], &device);
    let transform = TranslationTransform::<B, 3>::new(params);
    
    let metric = AdvancedNormalizedMutualInformation::<B>::new(
        32, 0.0, 255.0, 1.0, NormalizationMethod::JointEntropy
    );
    
    let loss = metric.forward(&fixed, &moving, &transform);
    let loss_val = loss.into_scalar();
    
    println!("ANMI (Shifted) Loss: {}", loss_val);
    
    // Compare with identical
    let identity = TranslationTransform::<B, 3>::new(Tensor::zeros([3], &Default::default()));
    let loss_identity = metric.forward(&fixed, &moving, &identity).into_scalar();
    
    assert!(loss_identity < loss_val, "Identity loss should be lower (more negative) than shifted loss");
}
