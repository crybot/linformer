import torch
from torch import nn, Tensor
from models.transformers import Transformer, MultiHeadAttention
from models.transformers import TransformerEncoder, TransformerEncoderLayer
from models.transformers import TransformerDecoder, TransformerDecoderLayer
from models.transformers import NLPTransformer 
from models.transformers import LanguageModelingHead
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import CSVDataset
from training import TrainingLoop, make_optimizer
import pkbar
from training.callbacks import ProgressbarCallback
from evaluation.metrics import perplexity
from evaluation.utils import extract_probs
from utils import to_device

def multiple_eight(n: int) -> int:
    return (n + 7) // 8 * 8

def max_non_padded_length(sequence: Tensor, pad_token = 1) -> int:
    non_padded_mask = sequence != pad_token
    # Sum along the sequence dimension (N) to count non-padding tokens for each sequence
    lengths = non_padded_mask.sum(dim=1)

    # Find the maximum length in the batch
    return lengths.max().item()

def trim_batch_pad_tokens(inputs: tuple[Tensor]) -> Tensor:
    src, tgt, src_mask, tgt_mask = inputs
    max_src_length = max_non_padded_length(src)
    max_tgt_length = max_non_padded_length(tgt)

    src, src_mask = src[:, :max_src_length], src_mask[:, :max_src_length]
    tgt, tgt_mask = tgt[:, :max_tgt_length], tgt_mask[:, :max_tgt_length]

    return src, tgt, src_mask, tgt_mask

def seq2seq_perplexity(pred, inputs):
    _, tgt, _, tgt_mask = inputs
    tgt_mask = tgt_mask[..., 1:].bool()
    masked_pred = pred[tgt_mask]
    masked_tgt = tgt[..., 1:][tgt_mask]

    return perplexity(extract_probs(masked_pred, masked_tgt))


class CustomTrainingLoop(TrainingLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess_batch(
            self,
            inputs: tuple[Tensor, Tensor, Tensor, Tensor],
            *args,
            **kwargs
            ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return to_device(*trim_batch_pad_tokens(inputs), device=self.device)

    def forward(
            self,
            inputs: tuple[Tensor, Tensor, Tensor, Tensor],
            *args,
            **kwargs
            ) -> tuple[Tensor, Tensor]:

        src, tgt, src_mask, tgt_mask = inputs
        pred = self.model(src, tgt, src_mask, tgt_mask, log=True)

        # TODO: refactor masking
        tgt_mask = tgt_mask[..., 1:].bool()
        masked_pred = pred[tgt_mask].view(-1, self.model.classes)
        masked_tgt = tgt[..., 1:][tgt_mask].reshape(-1)
        loss = self.loss_fn(masked_pred, masked_tgt)

        return pred, loss

def main():
    dim = 512
    mlp_dim = 1024
    n_heads = 8
    n_layers = 3
    batch_size = 170
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 5

    attn = MultiHeadAttention(dim, n_heads)
    encoder = TransformerEncoder(TransformerEncoderLayer(dim, mlp_dim, attn), n_layers=n_layers)
    decoder = TransformerDecoder(TransformerDecoderLayer(dim, mlp_dim, attn), n_layers=n_layers)

    tokenizer = AutoTokenizer.from_pretrained('./HLT/models/facebook/bart-base', padding_side='right', clean_up_tokenization_spaces=True, use_fast=False)
    vocab_size = multiple_eight(tokenizer.vocab_size)
    transformer = NLPTransformer(encoder = encoder, decoder = decoder, padding = True, vocab_size = vocab_size)
    model = LanguageModelingHead(transformer).to(device)
    loss_fn = nn.NLLLoss(reduction='mean').to(device)

    dataset = CSVDataset('./HLT/datasets/wmt14_translate_de-en_test.csv', src_key = 'en', tgt_key='de', tokenizer=tokenizer, device='cpu')
    opt = make_optimizer(torch.optim.Adam, lr=1e-4)

    training_loop = CustomTrainingLoop(
            dataset,
            loss_fn,
            opt,
            train_p = 0.8,
            val_p = 0.1,
            test_p = 0.1,
            random_split = False,
            mixed_precision = True,
            batch_size = batch_size,
            shuffle = True,
            device = device,
            num_workers = 4,
            val_metrics = {'perplexity': seq2seq_perplexity},
            callbacks = [
                ProgressbarCallback(epochs=epochs, width=20)
                ]
            )

    training_loop.run(model, epochs=epochs)

if __name__ == '__main__':
    main()
