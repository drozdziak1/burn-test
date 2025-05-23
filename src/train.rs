use std::{
    fs::{create_dir_all, remove_dir_all},
    sync::Arc,
};

use anyhow::Result;
use burn::{
    data::dataloader::{DataLoaderBuilder, MultiThreadDataLoader},
    optim::AdamWConfig,
    prelude::*,
    tensor::backend::AutodiffBackend,
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::{
    data_helpers::{FinewebBatch, FinewebBatcher, PackedBatchStrategy},
    model::{TrafoConfig, Transformer},
};

impl<B: AutodiffBackend> TrainStep<FinewebBatch<B>, ClassificationOutput<B>> for Transformer<B> {
    fn step(&self, batch: FinewebBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.x, batch.y_gt);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<FinewebBatch<B>, ClassificationOutput<B>> for Transformer<B> {
    fn step(&self, batch: FinewebBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.x, batch.y_gt)
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub model: TrafoConfig,
    pub optimizer: AdamWConfig,
    #[config(default = 600_000)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 32)]
    pub num_workers: usize,
    #[config(default = 0xdeadbeef)]
    pub rng_seed: u64,
    #[config(default = 3e-4)]
    pub lr: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    remove_dir_all(artifact_dir).expect("Could not clean artifact dir");
    create_dir_all(artifact_dir).expect("Could not create artifact dir");
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) -> Result<()> {
    create_artifact_dir(artifact_dir);

    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Could not save config");

    B::seed(config.rng_seed);

    let batcher = Arc::new(FinewebBatcher::default());

    let dataset = unimplemented!();
    let delim = unimplemented!();

    let dataloader_train = MultiThreadDataLoader::new(
        Box::new(PackedBatchStrategy::new(
            config.batch_size.try_into()?,
            config.model.ctx_size + 1,
            Some(delim),
        )),
        dataset,
        batcher.clone(),
        config.num_workers,
        device.clone(),
        None,
    );
}
