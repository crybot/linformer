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

def max_non_padded_length(sequence: Tensor, pad_token: int = 1) -> int:
    non_padded_mask = sequence != pad_token
    # Sum along the sequence dimension (N) to count non-padding tokens for each sequence
    lengths = non_padded_mask.sum(dim=1)

    # Find the maximum length in the batch
    return lengths.max().item()

def trim_batch_pad_tokens(inputs: tuple[Tensor], pad_token: int = 1) -> Tensor:
    src, tgt, src_mask, tgt_mask = inputs
    max_src_length = max_non_padded_length(src, pad_token = pad_token)
    max_tgt_length = max_non_padded_length(tgt, pad_token = pad_token)

    src, src_mask = src[:, :max_src_length], src_mask[:, :max_src_length]
    tgt, tgt_mask = tgt[:, :max_tgt_length], tgt_mask[:, :max_tgt_length]

    return src, tgt, src_mask, tgt_mask

def seq2seq_perplexity(pred, inputs):
    _, _, tgt, _, _, tgt_mask = encoder_decoder_inputs(*inputs)

    tgt = tgt[tgt_mask]
    return perplexity(extract_probs(pred, tgt))

def encoder_decoder_inputs(src, tgt, src_mask, tgt_mask):
    """ Return the appropriate inputs for an encoder-decoder model:

        Expected arguments:
        - src:        source sequence enclosed in <s> </s> possibly padded
        - tgt:        target sequence enclosed in <s> </s> and possibly padded
        - src_mask:   source mask
        - tgt_mask:   target mask

        Returns tuple containing in order:
        - encoder input
        - decoder input
        - target sequence
        - encoder input mask
        - decoder input mask
        - target mask
    """
    return src, tgt[..., :-1], tgt[..., 1:], src_mask.bool(), tgt_mask[..., :-1].bool(), tgt_mask[..., 1:].bool()


class CustomTrainingLoop(TrainingLoop):
    def __init__(self, *args, pad_token_id: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_token_id = pad_token_id

    def preprocess_batch(
            self,
            inputs: tuple[Tensor, Tensor, Tensor, Tensor],
            *args,
            **kwargs
            ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return to_device(*trim_batch_pad_tokens(inputs, self.pad_token_id), device=self.device)

    def forward(
            self,
            inputs: tuple[Tensor, Tensor, Tensor, Tensor],
            *args,
            **kwargs
            ) -> tuple[Tensor, Tensor]:

        enc_in, dec_in, tgt, enc_mask, dec_mask, tgt_mask = encoder_decoder_inputs(*inputs)
        pred = self.model(enc_in, dec_in, enc_mask, dec_mask, log=True)

        pred = pred[tgt_mask]
        tgt = tgt[tgt_mask].reshape(-1)
        loss = self.loss_fn(pred.view(-1, self.model.classes), tgt)

        return pred, loss

def warmup_model(model: nn.Module, inputs, loss_fn, device='cpu'):
    inputs = to_device(*inputs, device=device)
    enc_in, dec_in, tgt, enc_mask, dec_mask, tgt_mask = encoder_decoder_inputs(*inputs)

    print(enc_in.shape)
    print(dec_in.shape)

    with torch.autocast(device_type=device):
        pred = model(enc_in, dec_in, enc_mask, dec_mask, log=True)
        pred = pred[tgt_mask]
        tgt = tgt[tgt_mask].reshape(-1)
        loss = loss_fn(pred.view(-1, model.classes), tgt)

    loss.backward()


def main():
    dim = 512
    mlp_dim = 2048
    n_heads = 8
    n_layers = 3
    batch_size = 128
    max_length = 200
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 5

    attn = MultiHeadAttention(dim, n_heads)
    encoder = TransformerEncoder(TransformerEncoderLayer(dim, mlp_dim, attn), n_layers=n_layers)
    decoder = TransformerDecoder(TransformerDecoderLayer(dim, mlp_dim, attn), n_layers=n_layers)

    tokenizer = AutoTokenizer.from_pretrained(
            './HLT/models/facebook/bart-base',
            padding_side='right',
            clean_up_tokenization_spaces=True,
            use_fast=False
            )
    vocab_size = multiple_eight(tokenizer.vocab_size)

    transformer = NLPTransformer(encoder = encoder, decoder = decoder, padding = True, vocab_size = vocab_size)
    model = LanguageModelingHead(transformer).to(device)
    loss_fn = nn.NLLLoss(reduction='mean').to(device)

    # TODO: save npz of tokenized dataset to save time
    dataset = CSVDataset(
            './HLT/datasets/wmt14-tokenized.npz',
            from_dump=True
            )
    # dataset = CSVDataset(
    #         './HLT/datasets/wmt14_translate_de-en_train-0.csv',
    #         src_key = 'en',
    #         tgt_key='de',
    #         tokenizer=tokenizer,
    #         padding='max_length',
    #         max_length=max_length,
    #         truncation=True,
    #         device='cpu'
    #         )
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
            pad_token_id = tokenizer.pad_token_id,
            val_metrics = {'perplexity': seq2seq_perplexity},
            callbacks = [
                ProgressbarCallback(epochs=epochs, width=20)
                ]
            )

    warmup_model(model, dataset[:batch_size], loss_fn, device=device)

    training_loop.run(model, epochs=epochs)

if __name__ == '__main__':
    main()
