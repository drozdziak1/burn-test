use burn::{
    nn::{
        Embedding, EmbeddingConfig, Gelu, LayerNorm, Linear, LinearConfig, PositionalEncoding,
        attention::MultiHeadAttention,
    },
    prelude::*,
    train::ClassificationOutput,
};
use nn::{
    LayerNormConfig, PositionalEncodingConfig,
    attention::{MhaInput, MultiHeadAttentionConfig, generate_autoregressive_mask},
    loss::CrossEntropyLossConfig,
};

#[derive(Module, Debug)]
pub struct TrafoBlock<B: Backend> {
    ln_atn: LayerNorm<B>,
    atn: MultiHeadAttention<B>,
    ln_ff: LayerNorm<B>,
    ff1: Linear<B>,
    act: Gelu,
    ff2: Linear<B>,
}

impl<B: Backend> TrafoBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_length, _embed_dim] = x.dims();
        let dev = x.device();

        let x_ln_atn = self.ln_atn.forward(x.clone());
        let x_atn_input = MhaInput::self_attn(x_ln_atn)
            .mask_attn(generate_autoregressive_mask(batch_size, seq_length, &dev));

        let x_atn = self.atn.forward(x_atn_input).context;
        let x_with_atn = x + x_atn;

        let x_ln_ff = self.ln_ff.forward(x_with_atn.clone());
        let x_ff1_act = self.act.forward(self.ff1.forward(x_ln_ff));
        let x_ff2 = self.ff2.forward(x_ff1_act);

        return x_with_atn + x_ff2;
    }
}

#[derive(Module, Debug)]
pub struct Transformer<B: Backend> {
    tok_emb: Embedding<B>,
    pos_enc: PositionalEncoding<B>,
    blocks: Vec<TrafoBlock<B>>,
    unembed: Linear<B>,
}

impl<B: Backend> Transformer<B> {
    pub fn forward(&self, x: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let x_tok_emb = self.tok_emb.forward(x);
        let x_pos_enc = self.pos_enc.forward(x_tok_emb);

        let x_blocks = self
            .blocks
            .iter()
            .fold(x_pos_enc, |h, block| block.forward(h));

        let logits = self.unembed.forward(x_blocks);

        logits
    }
    pub fn forward_classification(
        &self,
        x: Tensor<B, 2, Int>,
        y_gt: Tensor<B, 2, Int>,
    ) -> ClassificationOutput<B> {
        let y_hat = self.forward(x);

        let y_hat_flattened = y_hat.flatten(0, 1);

        let y_gt_reshaped = y_gt.reshape([-1]);

        let loss = CrossEntropyLossConfig::new()
            .init(&y_hat_flattened.device())
            .forward(y_hat_flattened.clone(), y_gt_reshaped.clone());

        ClassificationOutput::new(loss, y_hat_flattened, y_gt_reshaped)
    }
}

#[derive(Config, Debug)]
pub struct TrafoConfig {
    #[config(default = 1024)]
    pub ctx_size: usize,
    pub vocab_size: usize,
    #[config(default = 768)]
    pub embed_dim: usize,
    #[config(default = 12)]
    pub num_blocks: usize,
    #[config(default = 12)]
    pub num_heads: usize,
    #[config(default = 3072)]
    pub ff_dim: usize,
}

impl TrafoConfig {
    pub fn init_trafo_block<B: Backend>(&self, device: &B::Device) -> TrafoBlock<B> {
        TrafoBlock {
            ln_atn: LayerNormConfig::new(self.embed_dim).init(device),
            atn: MultiHeadAttentionConfig::new(self.embed_dim, self.num_heads).init(device),
            ln_ff: LayerNormConfig::new(self.embed_dim).init(device),
            ff1: LinearConfig::new(self.embed_dim, self.ff_dim).init(device),
            act: Gelu::new(),
            ff2: LinearConfig::new(self.ff_dim, self.embed_dim).init(device),
        }
    }

    pub fn init_transformer<B: Backend>(&self, device: &B::Device) -> Transformer<B> {
        let blocks = (0..self.num_blocks)
            .map(|_| self.init_trafo_block(device))
            .collect();

        Transformer {
            tok_emb: EmbeddingConfig::new(self.vocab_size, self.embed_dim).init(device),
            pos_enc: PositionalEncodingConfig::new(self.embed_dim).init(device),
            blocks,
            unembed: LinearConfig::new(self.embed_dim, self.vocab_size)
                .with_bias(false)
                .init(device),
        }
    }
}
