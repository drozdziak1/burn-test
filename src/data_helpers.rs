use anyhow::{Result, anyhow, bail};
use burn::{
    data::{dataloader::{batcher::Batcher, BatchStrategy}, dataset::transform::Mapper},
    prelude::Backend,
    tensor::{s, Int, Tensor},
};
use hf_hub::api::sync::{Api, ApiRepo};
use log::{debug, warn};
use polars::prelude::*;
use tokenizers::{InputSequence, Tokenizer};

use std::{fs::File, io::Read, num::NonZeroUsize, path::PathBuf};

use burn::data::dataset::HuggingfaceDatasetLoader;

pub struct PackedBatchStrategy<T> {
    batch_size: NonZeroUsize,
    batch_item_size: usize,
    delim: Option<T>,
    buffers: Vec<Vec<T>>,
}

impl<T: Clone> PackedBatchStrategy<T> {
    pub fn new(batch_size: NonZeroUsize, batch_item_size: usize, delim: Option<T>) -> Self {
        Self {
            batch_size,
            batch_item_size,
            delim,
            buffers: vec![Vec::new(); batch_size.into()],
        }
    }
}

impl<T: Send + Clone + 'static> BatchStrategy<Vec<T>> for PackedBatchStrategy<T> {
    fn add(&mut self, mut item: Vec<T>) {
        let shortest_buf = self
            .buffers
            .iter_mut()
            .min_by(|x, y| x.len().cmp(&y.len()))
            .expect(
                "INTERNAL: PackedBatchStrategy: a shortest vector in self.buffers does not exist, despite batch_size: NonZeroUsize",
            );

        if !shortest_buf.is_empty() {
            if let Some(delim) = self.delim.as_ref() {
                shortest_buf.push(delim.clone());
            }
        }

        shortest_buf.append(&mut item);
    }

    fn batch(&mut self, force: bool) -> Option<Vec<Vec<T>>> {
        // Cheap: move on if force
        if !force {
            // Less cheap: check if we have enough data for a batch
            let any_buf_too_short = self
                .buffers
                .iter()
                .any(|buf| buf.len() < self.batch_item_size);

            if any_buf_too_short {
                return None;
            }
        }

        let mut batch: Vec<Vec<T>> = Vec::new();

        for buf in self.buffers.iter_mut() {
            // Skip incomplete buffers to avoid split_at panic (only possible if force)
            if force && buf.len() < self.batch_item_size {
                continue;
            }

            let (new_batch_item, new_buf) = buf.split_at(self.batch_item_size);

            batch.push(new_batch_item.to_vec());
            *buf = new_buf.to_vec();
        }

        Some(batch)
    }

    fn clone_dyn(&self) -> Box<dyn BatchStrategy<Vec<T>>> {
        Box::new(Self::new(
            self.batch_size,
            self.batch_item_size,
            self.delim.clone(),
        ))
    }
}

#[derive(Clone, Debug)]
pub struct FinewebBatch<B: Backend> {
    pub x: Tensor<B, 2, Int>,
    pub y_gt: Tensor<B, 2, Int>,
}

#[derive(Clone, Default)]
pub struct FinewebBatcher;

impl<B: Backend> Batcher<B, Vec<u32>, FinewebBatch<B>> for FinewebBatcher {
    fn batch(&self, items: Vec<Vec<u32>>, device: &B::Device) -> FinewebBatch<B> {
        let tensors_vec = items
            .iter()
            .map(|item| {
                let t: Tensor<B, 1, Int> = Tensor::from_ints(item.as_slice(), device);

                t.unsqueeze()
            })
            .collect();

        let t: Tensor<B, 2, _> = Tensor::cat(tensors_vec, 0);

        let x = t.clone().slice(s![.., ..-1]);
        let y_gt = t.slice(s![.., 1..]);

        FinewebBatch { x, y_gt }
    }
}

#[derive(Clone)]
pub struct FinewebMapper {
    pub tokenizer: Tokenizer,
}

impl FinewebMapper {
    pub fn new(tokenizer: Tokenizer) -> Self {
	Self {
	    tokenizer
	}
    }
}

impl Mapper<String, Vec<u32>> for FinewebMapper {
    fn map(&self, item: &String) -> Vec<u32> {

	self.tokenizer.encode(item.as_str(), true).expect("Could not map using tokenizer").get_ids().to_vec()
    }
}
