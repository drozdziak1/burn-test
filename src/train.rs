use std::{
    fs::{create_dir_all, remove_dir_all},
    sync::Arc,
};

use anyhow::{Result, anyhow};
use burn::{
    data::{
        dataloader::MultiThreadDataLoader,
        dataset::{
            transform::{MapperDataset, PartialDataset, SamplerDataset}, HuggingfaceDatasetLoader
        },
    },
    lr_scheduler::noam::NoamLrSchedulerConfig,
    optim::AdamWConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        metric::{AccuracyMetric, CudaMetric, LearningRateMetric, LossMetric}, ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep
    },
};
use tokenizers::Tokenizer;

use crate::{
    data_helpers::{FinewebBatch, FinewebBatcher, FinewebMapper, PackedBatchStrategy},
    model::{TrafoConfig, Transformer},
};

impl<B: AutodiffBackend> TrainStep<FinewebBatch<B>, ClassificationOutput<B>> for Transformer<B> {
    fn step(&self, batch: FinewebBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.x, batch.y_gt);

        let grads = item.loss.backward();

        let out = TrainOutput::new(self, grads, item);

        out
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
    pub num_train_steps: usize,
    #[config(default = 10_000)]
    pub num_steps_per_epoch: usize,
    #[config(default = 200)]
    pub num_val_steps: usize,
    #[config(default = 9)]
    pub batch_size: usize,
    #[config(default = 8)]
    pub num_workers: usize,
    #[config(default = 0xdeadbeef)]
    pub rng_seed: u64,
    #[config(default = 3e-4)]
    pub max_lr: f64,
    #[config(default = 2_000)]
    pub warmup_steps: usize,
    #[config(default = 50)]
    pub grad_accum_steps: usize,
}

fn create_artifact_dir(artifact_dir: &str) {
    remove_dir_all(artifact_dir).ok();
    create_dir_all(artifact_dir).expect("Could not create artifact dir");
}

pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
    config: TrainingConfig,
    devices: Vec<B::Device>,
) -> Result<()> {
    create_artifact_dir(artifact_dir);

    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Could not save config");

    B::seed(config.rng_seed);

    let batcher = Arc::new(FinewebBatcher::default());

    let full_dataset = HuggingfaceDatasetLoader::new("HuggingFaceFW/fineweb")
        .with_subset("sample-10BT")
        .with_use_python_venv(false)
        .dataset("train")
        .expect("Could not load dataset");

    let tokenizer = Tokenizer::from_pretrained("gpt2", None).expect("Could not load tokenizer");
    let delim = tokenizer
        .encode("<|endoftext|>", true)
        .expect("Could not encode delim")
        .get_ids()[0];

    println!("Using delim {}", delim);

    let tokenized_dataset = Arc::new(MapperDataset::new(
        full_dataset,
        FinewebMapper::new(tokenizer),
    ));

    let dataset_train = PartialDataset::new(
        tokenized_dataset.clone(),
        config.num_val_steps + 1,
        config.num_train_steps + config.num_val_steps + 1,
    );
    let dataset_test = PartialDataset::new(
        tokenized_dataset,
        0,
        config.num_val_steps,
    );

    let dataloader_train = MultiThreadDataLoader::new(
        Box::new(PackedBatchStrategy::new(
            config.batch_size.try_into()?,
            config.model.ctx_size + 1,
            Some(delim),
        )),
        Arc::new(SamplerDataset::with_replacement(dataset_train, config.num_steps_per_epoch)),
        batcher.clone(),
        config.num_workers,
        devices[0].clone(),
        None,
    );

    let dataloader_test = MultiThreadDataLoader::new(
        Box::new(PackedBatchStrategy::new(
            config.batch_size.try_into()?,
            config.model.ctx_size + 1,
            Some(delim),
        )),
        Arc::new(dataset_test),
        batcher.clone(),
        config.num_workers,
        devices[0].clone(),
        None,
    );

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train(CudaMetric::new())
        .metric_valid(CudaMetric::new())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(devices.clone())
        .num_epochs(config.num_train_steps / config.num_steps_per_epoch)
        .summary()
        .grads_accumulation(config.grad_accum_steps / devices.len())
        .build(
            config.model.init_transformer::<B>(&devices[0]),
            config.optimizer.init(),
            NoamLrSchedulerConfig::new(config.max_lr * config.grad_accum_steps as f64)
                .with_model_size(config.model.embed_dim)
                .with_warmup_steps(config.warmup_steps)
                .init()
                .map_err(|e| anyhow!(e))?,
        );

    let model_trained = learner.fit(Arc::new(dataloader_train), Arc::new(dataloader_test));

    model_trained.save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())?;

    Ok(())
}
