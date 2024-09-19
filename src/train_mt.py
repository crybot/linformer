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

class CustomTrainingLoop(TrainingLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, inputs, *args, **kwargs) -> tuple[Tensor, Tensor]:
        src, tgt, src_mask, tgt_mask = inputs
        pred = self.model(src, tgt, src_mask, tgt_mask, log=True)

        # TODO: refactor masking
        tgt_mask = tgt_mask[..., 1:].bool()
        masked_pred = pred[tgt_mask].view(-1, self.model.classes)
        masked_tgt = tgt[..., 1:][tgt_mask].reshape(-1)
        loss = self.loss_fn(masked_pred, masked_tgt)

        return pred, loss

def seq2seq_perplexity(pred, inputs):
        _, tgt, _, tgt_mask = inputs
        tgt_mask = tgt_mask[..., 1:].bool()
        masked_pred = pred[tgt_mask]
        masked_tgt = tgt[..., 1:][tgt_mask]

        return perplexity(extract_probs(masked_pred, masked_tgt))

def main():
    dim = 512
    mlp_dim = 1024
    n_heads = 8
    n_layers = 3
    batch_size = 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 10

    attn = MultiHeadAttention(dim, n_heads)
    encoder = TransformerEncoder(TransformerEncoderLayer(dim, mlp_dim, attn), n_layers=n_layers)
    decoder = TransformerDecoder(TransformerDecoderLayer(dim, mlp_dim, attn), n_layers=n_layers)

    tokenizer = AutoTokenizer.from_pretrained('./HLT/models/facebook/bart-base', padding_side='right', clean_up_tokenization_spaces=True)
    transformer = NLPTransformer(encoder = encoder, decoder = decoder, padding = True, vocab_size = tokenizer.vocab_size)
    model = LanguageModelingHead(transformer).to(device)
    loss_fn = nn.NLLLoss(reduction='mean')

    dataset = CSVDataset('./HLT/datasets/wmt14_translate_de-en_test.csv', src_key = 'en', tgt_key='de', tokenizer=tokenizer, device=device)
    opt = make_optimizer(torch.optim.Adam, lr=1e-4)

    training_loop = CustomTrainingLoop(
            dataset,
            loss_fn,
            opt,
            train_p = 0.8,
            val_p = 0.1,
            test_p = 0.1,
            random_split = False,
            mixed_precision = False,
            batch_size = batch_size,
            shuffle = True,
            pin_memory = False,
            device = device,
            num_workers = 0,
            val_metrics = {'perplexity': seq2seq_perplexity},
            callbacks = [
                ProgressbarCallback(epochs=epochs, width=20)
                ]
            )

    training_loop.run(model, epochs=epochs)

if __name__ == '__main__':
    main()
