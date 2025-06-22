mod data_helpers;
mod model;
mod train;

use anyhow::Result;
use burn::{
    backend::{cuda::CudaDevice, Autodiff, Cuda},
    grad_clipping::GradientClippingConfig,
    optim::AdamWConfig,
};
use model::TrafoConfig;
use train::TrainingConfig;

fn main() -> Result<()> {
    type MyBackend = Cuda<half::f16, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    env_logger::init();

    let devices = vec![CudaDevice::new(0)]; // , CudaDevice::new(1)];

    let artifact_dir = "/tmp/burn-test";

    crate::train::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(
            TrafoConfig::new(50257),
            AdamWConfig::new().with_grad_clipping(Some(GradientClippingConfig::Norm(1.0))),
        ),
        devices,
    )?;

    Ok(())
}
