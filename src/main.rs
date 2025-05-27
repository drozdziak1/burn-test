mod data_helpers;
mod model;
mod train;

use anyhow::Result;
use burn::{backend::{Autodiff, Cuda}, optim::AdamWConfig};
use model::TrafoConfig;
use train::TrainingConfig;

use crate::data_helpers::FinewebBatcher;

fn main() -> Result<()> {
    type MyBackend = Cuda<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::cuda::CudaDevice::default();

    let artifact_dir = "/tmp/burn-test";

    crate::train::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(
            TrafoConfig::new(1024, 50257, 768, 12, 12, 4 * 768),
            AdamWConfig::new(),
        ),
        device.clone(),
    )?;

    Ok(())
}
