import os
import torch
import wandb
import argparse
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import CSVDataset
from evaluation.metrics import perplexity
from evaluation.utils import extract_probs
from utils import to_device
from utils import load_model_from_wandb_checkpoint
from sacrebleu import corpus_bleu
from text.utils import encoder_decoder_inputs, trim_batch_pad_tokens

def evaluate(model, dataset, tokenizer, out_file = None):
    batch_size = 100
    dl = DataLoader(dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=4,
            prefetch_factor=8,
            persistent_workers=False)

    input_sentences = []
    candidates = []
    references = []
    log_probs = torch.tensor([], device='cuda')

    for batch, inputs in enumerate(dl):
        print(f'Processing batch {batch + 1}/{len(dl)}')
        inputs = to_device(*trim_batch_pad_tokens(inputs, tokenizer.pad_token_id), device='cuda')
        enc_in, dec_in, tgt, enc_mask, dec_mask, tgt_mask = encoder_decoder_inputs(*inputs)

        # Generate batch of candidate translations with greedy decoding
        cand = model.generate(enc_in, tokenizer, max_length = 256, src_mask = enc_mask)
        ref = tokenizer.batch_decode(tgt, skip_special_tokens=True)

        candidates.extend(cand)
        references.extend(ref)
        input_sentences.extend(tokenizer.batch_decode(enc_in, skip_special_tokens=True))

        # Accumulate per-token log-probabilities to compute corpus perplexity
        with torch.no_grad():
            pred = model(enc_in, dec_in, enc_mask, dec_mask, log=True)
        pred = pred[tgt_mask]
        tgt = tgt[tgt_mask]

        log_probs = torch.cat([log_probs, extract_probs(pred, tgt).view(-1)])

    references = [[s] for s in references]
    bleu = corpus_bleu(candidates, references).score
    ppl = perplexity(log_probs)

    if out_file:
        print('Saving file...')
        df = pd.DataFrame()
        df['input'] = input_sentences
        df['candidate'] = candidates
        df['reference'] = [r[0] for r in references]
        df.to_csv(out_file, header=True, index=False)
            

    return bleu, ppl

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint file')
    return parser.parse_args()

def main(args):
    max_length = 256
    tokenizer = AutoTokenizer.from_pretrained(
            './HLT/models/facebook/bart-base',
            padding_side='right',
            clean_up_tokenization_spaces=True,
            use_fast=False
            )

    dataset = CSVDataset(
            './HLT/datasets/wmt14_translate_de-en_test.csv',
            src_key = 'en',
            tgt_key='de',
            tokenizer=tokenizer,
            padding='max_length',
            max_length=max_length,
            truncation=True,
            device='cpu'
            )

    print(f'Checkpoint path provided: {args.checkpoint}')
    print(f'Resuming...')
    model = load_model_from_wandb_checkpoint(f'HLT/{args.checkpoint}', device='cuda')

    run = wandb.init(
            id=args.checkpoint,
            project='HLT',
            resume='must',
            reinit=True)

    out_file = os.path.join('./HLT/artifacts', f'{args.checkpoint}_out.csv')
    bleu, ppl = evaluate(model, dataset, tokenizer, out_file)
    wandb.log({'test_bleu': bleu, 'test_perplexity': ppl}, commit=True)

if __name__ == '__main__':
    args = parse_args()
    if not args.checkpoint:
        raise ValueError("No checkpoint provided")

    main(args)
