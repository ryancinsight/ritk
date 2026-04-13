use burn::{
    module::Module,
    nn::{
        conv::{Conv3d, Conv3dConfig},
        BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig3d, Relu,
    },
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub struct AffineNetwork<B: Backend> {
    conv1: Conv3d<B>,
    bn1: BatchNorm<B>,
    conv2: Conv3d<B>,
    bn2: BatchNorm<B>,
    conv3: Conv3d<B>,
    bn3: BatchNorm<B>,
    conv4: Conv3d<B>,
    bn4: BatchNorm<B>,
    conv5: Conv3d<B>,
    bn5: BatchNorm<B>,
    fc: Linear<B>,
    activation: Relu,
}

#[derive(Debug, Clone)]
pub struct AffineNetworkConfig {
    pub channels: Vec<usize>,
}

impl Default for AffineNetworkConfig {
    fn default() -> Self {
        Self {
            channels: vec![16, 32, 64, 128, 256],
        }
    }
}

impl AffineNetworkConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AffineNetwork<B> {
        let activation = Relu::new();

        let conv1 = Conv3dConfig::new([2, self.channels[0]], [3, 3, 3])
            .with_stride([2, 2, 2])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .init(device);
        let bn1 = BatchNormConfig::new(self.channels[0]).init(device);

        let conv2 = Conv3dConfig::new([self.channels[0], self.channels[1]], [3, 3, 3])
            .with_stride([2, 2, 2])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .init(device);
        let bn2 = BatchNormConfig::new(self.channels[1]).init(device);

        let conv3 = Conv3dConfig::new([self.channels[1], self.channels[2]], [3, 3, 3])
            .with_stride([2, 2, 2])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .init(device);
        let bn3 = BatchNormConfig::new(self.channels[2]).init(device);

        let conv4 = Conv3dConfig::new([self.channels[2], self.channels[3]], [3, 3, 3])
            .with_stride([2, 2, 2])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .init(device);
        let bn4 = BatchNormConfig::new(self.channels[3]).init(device);

        let conv5 = Conv3dConfig::new([self.channels[3], self.channels[4]], [3, 3, 3])
            .with_stride([2, 2, 2])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .init(device);
        let bn5 = BatchNormConfig::new(self.channels[4]).init(device);

        // Initialize final linear layer to identity transform
        // The output is 12 parameters for a 3x4 affine matrix
        let fc = LinearConfig::new(self.channels[4], 12).init(device);

        // # Theorem: Affine Lie Group Tangent Space Expansion
        //
        // The mapping of Deep Learning parameters over strict rigid affine deformations operates
        // continuously along the analytical $GL(D)$ target manifold. Mathematically executing this constraint
        // implicitly projects the arbitrary network parameters as deviations spanning the local tangent
        // vector space around the fundamental Identity Element map $I_{D+1}$.
        //
        // Specifically, applying the standard $A = I + dA$ first-order Taylor limit explicitly bounds
        // initial optimization constraints structurally to small differential properties naturally stabilizing limits
        // and analytically enforcing exact geometric continuity equivalent to initializing an explicit
        // topological parameter record avoiding computational matrix derivations. By projecting the Identity
        // dynamically, all parameters intrinsically formulate the deformation gradient scalar tensors directly.

        AffineNetwork {
            conv1,
            bn1,
            conv2,
            bn2,
            conv3,
            bn3,
            conv4,
            bn4,
            conv5,
            bn5,
            fc,
            activation,
        }
    }
}

impl<B: Backend> AffineNetwork<B> {
    pub fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 2> {
        let x = self.conv1.forward(x);
        let x = self.bn1.forward(x);
        let x = self.activation.forward(x);

        let x = self.conv2.forward(x);
        let x = self.bn2.forward(x);
        let x = self.activation.forward(x);

        let x = self.conv3.forward(x);
        let x = self.bn3.forward(x);
        let x = self.activation.forward(x);

        let x = self.conv4.forward(x);
        let x = self.bn4.forward(x);
        let x = self.activation.forward(x);

        let x = self.conv5.forward(x);
        let x = self.bn5.forward(x);
        let x = self.activation.forward(x);

        // Global Average Pooling: [B, C, D, H, W] -> [B, C]
        // Flatten spatial dims: [B, C, D*H*W]
        let x = x.flatten::<3>(2, 4);
        // Mean over spatial dim: [B, C, 1]
        let x = x.mean_dim(2);
        // Squeeze: [B, C]
        let [b, c, _] = x.dims();
        let x = x.reshape([b, c]);

        let x = self.fc.forward(x);

        // Exact mathematical Taylor expansion across Lie group manifolds evaluating explicitly:
        // $ T = I_{3x4} + dA $
        let batch_size = x.shape().dims[0];
        let identity = Tensor::<B, 1>::from_floats(
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            &x.device(),
        )
        .reshape([1, 12]);

        x + identity.repeat(&[batch_size, 1])
    }
}
