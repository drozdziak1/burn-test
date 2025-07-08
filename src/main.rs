mod data_helpers;
mod model;
mod train;

use anyhow::Result;
use burn::{
    backend::{cuda::CudaDevice, libtorch::LibTorchDevice, Autodiff, Cuda, LibTorch},
    grad_clipping::GradientClippingConfig,
    optim::AdamWConfig,
};
use model::TrafoConfig;
use train::TrainingConfig;

fn main() -> Result<()> {
    type MyBackend = LibTorch<half::bf16, i8>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    env_logger::init();

    let devices = vec![LibTorchDevice::Cuda(0), LibTorchDevice::Cuda(1)];

    let artifact_dir = "/tmp/burn-test";

    crate::train::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(
            TrafoConfig::new(50257),
            AdamWConfig::new()
                .with_beta_1(0.9)
                .with_beta_2(0.95)
                .with_weight_decay(0.1)
                .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0))),
        ),
        devices,
    )?;

    Ok(())
}
