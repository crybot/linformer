import torch
from torch import nn, Tensor
from transformers import AutoTokenizer
from datasets import CSVDataset
from training import TrainingLoop 
from training.callbacks import ProgressbarCallback, CheckpointCallback
from training.callbacks import WandbCallback, LRSchedulerCallback
from evaluation.metrics import perplexity
from evaluation.utils import extract_probs
from utils import to_device, print_summary, set_random_state, download_wandb_checkpoint
from utils import make_model
import yaml
import argparse
from text.utils import trim_batch_pad_tokens, encoder_decoder_inputs

# TODO: Dropout as in the paper
# TODO: Label smoothing

def next_multiple(n: int, k: int) -> int:
    return (n + k - 1) // k * k

def seq2seq_perplexity(pred, inputs):
    _, _, tgt, _, _, tgt_mask = encoder_decoder_inputs(*inputs)

    tgt = tgt[tgt_mask]
    return perplexity(extract_probs(pred, tgt))

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

        # Only take output distributions for non-masked tokens: this flattens
        # the tensor so that pred has shape (B * N * self.model.classes) where
        # N is the total number of non masked tokens.
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

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint file')
    parser.add_argument('--config', type=str, help='Path to the config file')
    return parser.parse_args()

def main(args):
    set_random_state(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ARTIFACTS_PATH = './artifacts'
    config = load_config(args.config)

    tokenizer = AutoTokenizer.from_pretrained(
            './models/facebook/bart-base',
            padding_side='right',
            clean_up_tokenization_spaces=True,
            use_fast=False
            )
    config['model']['vocab_size'] = next_multiple(tokenizer.vocab_size, 8)

    print(config)

    model = make_model(config, device=device)
    loss_fn = nn.NLLLoss(reduction='mean').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    dataset = CSVDataset(
            './datasets/wmt14-050-tokenized-256.npz',
            from_dump=True,
            fraction = 1.0
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

    training_loop = CustomTrainingLoop(
            model,
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
            val_metrics = {
                'perplexity': seq2seq_perplexity
                },
            callbacks = [
                wandb_callback,
                checkpoint_callback,
                lr_scheduler_callback,
                ProgressbarCallback(epochs=epochs, width=20)
                ]
            )

    print_summary(model, print_model=True)

    if args.checkpoint:
        print(f'Checkpoint path provided: {args.checkpoint}')
        print(f'Resuming...')
        checkpoint = download_wandb_checkpoint(f'./{args.checkpoint}', 'checkpoint.pt', device=device)
        training_loop.load_state(model, checkpoint)

    model = training_loop.run(epochs=epochs)

if __name__ == '__main__':
    args = parse_args()
    if not args.config:
        raise ValueError("No config file provided")
    main(args)
