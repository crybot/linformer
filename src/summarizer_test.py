#!/usr/bin/python3
import torch
import numpy as np
import warnings
from models import Summarizer
from datasets import CSVDataset
from text.utils import random_mask, random_tokens_extract, random_text_extract, mask_fill
from utils import set_random_state


def test_summarization():
    SEED = 42
    torch.backends.cuda.matmul.allow_tf32 = True
    set_random_state(SEED)

    print("Summarization test")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading models")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summarizer = Summarizer(
                pt_lm_path="./HLT/models/facebook/bart-large-cnn",
                fpt_lm_path="./HLT/models/google-bert/bert-large-uncased",
                ).to(device)

    print("Loading dataset")
    dataset = CSVDataset("./HLT/datasets/bbc-text.csv", "text")


    optimizer = torch.optim.AdamW(
            summarizer.parameters(),
            lr=1e-3
            )

    summarizer.train()

    print("Summarization examples")
    for epoch in range(1):
        for text in dataset[:1]:
            # Clear gradients
            summarizer.zero_grad(set_to_none=True)

            mask_token_id = summarizer.inner_tokenizer.mask_token_id
            pad_token_id = summarizer.outer_tokenizer.pad_token_id # TODO: should use inner_tokenizer

            inputs = summarizer.tokenize(text)
            tokens = inputs["input_ids"]
            summary = summarizer.summarize(text, device=device)

            extract = random_tokens_extract(tokens)
            extract = summarizer._re_encode(extract) # TODO: refactor into Summarizer
            masked_extract, mask = random_mask(
                    extract,
                    mask_token_id,
                    mask_p=0.05,
                    pad_token_id=pad_token_id,
                    return_mask=True
                    )

            ms = masked_extract.repeat(1, 1)
            ts = extract.repeat(1, 1)

            # with torch.no_grad():
            #     outputs = summarizer(tokens.to(device), ms.to(device), ts.to(device), append_context=False)

            # print(f'Input: {text}')
            # print('---'*20)
            # print(f'Extract: {summarizer.decode_inner(masked_extract, skip_special_tokens=False)}')
            # print('---'*20)
            # print(f'Summary: {summary}')
            # print('---'*20)
            # print(f'Completion logits: {outputs.logits}')
            # print(f'Loss w/o summary: {outputs.loss}')

            outputs = summarizer(tokens.to(device), ms.to(device), ts.to(device))

            print(f'Loss w/ summary: {outputs.loss}')

            outputs.loss.backward()
            optimizer.step()

            # print(next(summarizer.fpt_lm.parameters()))

            # print(logits.shape)

            # masked_index = (masked_extract == tokenizer.mask_token_id)
            # labels = input_ids.clone()
            # labels[~masked_index] = -100

            # inputs["labels"] = labels
            # inputs = {key: val.to(device) for key, val in inputs.items()}

            # summarizer.train()
            # outputs = pt_lm(**inputs)
            # loss = outputs.loss

            print()

        print("Success")


        # TODO: reconstruction loss and text forward

test_summarization()

