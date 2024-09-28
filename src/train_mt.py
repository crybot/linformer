import torch
from torch import nn, Tensor
from models.transformers import Transformer, MultiHeadAttention, LinformerAttention
from models.transformers import TransformerEncoder, TransformerEncoderLayer
from models.transformers import TransformerDecoder, TransformerDecoderLayer
from models.transformers import NLPTransformer 
from models.transformers import LanguageModelingHead
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import CSVDataset
from training import TrainingLoop, make_optimizer
import pkbar
from training.callbacks import ProgressbarCallback, CheckpointCallback
from training.callbacks import WandbCallback, LRSchedulerCallback
from evaluation.metrics import perplexity
from evaluation.utils import extract_probs
from utils import to_device, print_summary, set_random_state, download_wandb_checkpoint
import yaml
import argparse

# TODO: Dropout as in the paper

# TODO: Label smoothing

# TODO: Implement BLEU metric

# NOTE: BLEU needs to be computed from a candidate translation against a (set
# of) reference ones. The problem is that the candidate has to be
# autoregressively computed and this might take a while. Moreover this cannot
# be computed in batches (or can it?) because of the variable length of each
# production. We can defer this metric to a last evaluation step, not to be
# computed as a validation step.

# TODO: Move utility functions to an appropriate module

# TODO: generalize
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

    with torch.autocast(device_type=device):
        pred = model(enc_in, dec_in, enc_mask, dec_mask, log=True)
        pred = pred[tgt_mask]
        tgt = tgt[tgt_mask].reshape(-1)
        loss = loss_fn(pred.view(-1, model.classes), tgt)

    loss.backward()

def make_model(config: dict, device='cpu') -> nn.Module:
    n = config['dataset']['max_length']
    config = config['model']
    dim = config['dim']
    mlp_dim = config['mlp_dim']
    n_heads = config['n_heads']
    n_layers = config['n_layers']
    vocab_size = config['vocab_size']

    if config.get('type', None) == 'Linformer':
        attn = LinformerAttention(dim, n_heads, k = config['k'], sequence_length = n)
    else:
        attn = MultiHeadAttention(dim, n_heads)

    encoder = TransformerEncoder(TransformerEncoderLayer(dim, mlp_dim, attn), n_layers=n_layers)
    decoder = TransformerDecoder(TransformerDecoderLayer(dim, mlp_dim, attn), n_layers=n_layers)

    transformer = NLPTransformer(encoder = encoder, decoder = decoder, vocab_size = vocab_size)
    return LanguageModelingHead(transformer).to(device)

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def parse_args():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Script with --checkpoint flag")
    parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint file')
    return parser.parse_args()

def main():
    set_random_state(42)
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ARTIFACTS_PATH = './HLT/artifacts'
    config = load_config('./HLT/configs/experiment_config.yaml')

    tokenizer = AutoTokenizer.from_pretrained(
            './HLT/models/facebook/bart-base',
            padding_side='right',
            clean_up_tokenization_spaces=True,
            use_fast=False
            )
    config['model']['vocab_size'] = multiple_eight(tokenizer.vocab_size)

    print(config)

    model = make_model(config, device=device)
    loss_fn = nn.NLLLoss(reduction='mean').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    dataset = CSVDataset(
            './HLT/datasets/wmt14-050-tokenized-256.npz',
            from_dump=True
            )

    wandb_callback = WandbCallback(
            project_name='HLT',
            entity='marco-pampaloni',
            config=config,
            tags=['test']
            )
    checkpoint_callback = CheckpointCallback(
            path=ARTIFACTS_PATH + '/checkpoint.pt',
            save_best=True,
            metric='val_loss',
            sync_wandb=True,
            debug=True
            )
    lr_scheduler_callback = LRSchedulerCallback(
            optimizer,
            config=config['training']['lr_scheduler']
            )

    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']

    # TODO: val_dataset batchs_size (no backward phase, so can be 2-3x bigger)
    training_loop = CustomTrainingLoop(
            dataset,
            loss_fn,
            optimizer,
            train_p = 0.99,
            val_p = 0.01,
            test_p = 0.0,
            random_split = False,
            mixed_precision = True,
            batch_size = batch_size,
            shuffle = True,
            device = device,
            num_workers = 4,
            pad_token_id = tokenizer.pad_token_id,
            val_metrics = {'perplexity': seq2seq_perplexity},
            callbacks = [
                wandb_callback,
                checkpoint_callback,
                lr_scheduler_callback,
                ProgressbarCallback(epochs=epochs, width=20)
                ]
            )

    print_summary(model, print_model=True)

    # TODO: project name, user and checkpoint filename
    if args.checkpoint:
        print(f'Checkpoint path provided: {args.checkpoint}')
        print(f'Resuming...')
        checkpoint = download_wandb_checkpoint(f'marco-pampaloni/HLT/{args.checkpoint}', 'checkpoint.pt', device=device)
        training_loop.load_state(model, checkpoint)

    training_loop.run(model, epochs=epochs)

if __name__ == '__main__':
    main()
