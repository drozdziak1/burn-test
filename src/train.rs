use std::{
    fs::{create_dir_all, remove_dir_all},
    sync::Arc,
};

use anyhow::Result;
use burn::{
    data::{
        dataloader::{DataLoaderBuilder, MultiThreadDataLoader},
        dataset::{
            HuggingfaceDatasetLoader,
            transform::{MapperDataset, PartialDataset},
        },
    },
    optim::AdamWConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep,
        metric::{AccuracyMetric, LossMetric},
    },
};
use tokenizers::Tokenizer;

use crate::{
    data_helpers::{FinewebBatch, FinewebBatcher, FinewebMapper, PackedBatchStrategy},
    model::{TrafoConfig, Transformer},
};

impl<B: AutodiffBackend> TrainStep<FinewebBatch<B>, ClassificationOutput<B>> for Transformer<B> {
    fn step(&self, batch: FinewebBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
	println!("train: before forward_classification");
        let item = self.forward_classification(batch.x, batch.y_gt);

	println!("train: before gradients");
	let grads = item.loss.backward();
	println!("train: after gradients");

        println!("train: before output constructor");
	let out = TrainOutput::new(self, grads, item);
        println!("train: after output constructor");

	out
    }
}

impl<B: Backend> ValidStep<FinewebBatch<B>, ClassificationOutput<B>> for Transformer<B> {
    fn step(&self, batch: FinewebBatch<B>) -> ClassificationOutput<B> {
	println!("valid: before forward_classification");
        self.forward_classification(batch.x, batch.y_gt)
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub model: TrafoConfig,
    pub optimizer: AdamWConfig,
    #[config(default = 600_000)]
    pub num_epochs: usize,
    #[config(default = 100)]
    pub num_test_epochs: usize,
    #[config(default = 2)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 0xdeadbeef)]
    pub rng_seed: u64,
    #[config(default = 3e-4)]
    pub lr: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    remove_dir_all(artifact_dir).ok();
    create_dir_all(artifact_dir).expect("Could not create artifact dir");
}

pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
    config: TrainingConfig,
    device: B::Device,
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

    let tokenized_dataset = Arc::new(MapperDataset::new(full_dataset, FinewebMapper::new(tokenizer)));

    let dataset_train = PartialDataset::new(
        tokenized_dataset.clone(),
        config.num_test_epochs * config.batch_size + 1,
        config.num_epochs + config.num_test_epochs * config.batch_size,
    );
    let dataset_test = PartialDataset::new(tokenized_dataset, 0, config.num_test_epochs * config.batch_size);

    let dataloader_train = MultiThreadDataLoader::new(
        Box::new(PackedBatchStrategy::new(
            config.batch_size.try_into()?,
            config.model.ctx_size + 1,
            Some(delim),
        )),
        Arc::new(dataset_train),
        batcher.clone(),
        config.num_workers,
        device.clone(),
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
        device.clone(),
        None,
    );

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init_transformer::<B>(&device),
            config.optimizer.init(),
            config.lr,
        );

    let model_trained = learner.fit(Arc::new(dataloader_train), Arc::new(dataloader_test));

    model_trained.save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())?;

    Ok(())
}
