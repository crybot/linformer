import torch
import numpy as np
from torch import Tensor, nn
from transformers import AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoTokenizer
from models.utils import freeze_
from models.straight_through_estimation import StraightThroughEstimator
from text.utils import random_tokens_extract, random_mask
import sys

# TODO: Implement custom DataColletor (see huggingface's DataCollatorForWholeWordMask)
#       instead of manually calling `random_mask`

def repeat_on_dim(t: Tensor, dim: int, n: int) -> Tensor:
    """
    Return a tensor with elements along dimension `dim` repeated `n` times
    """
    dims_repeated = [1]*t.dim() # 1 for each dimension of t
    dims_repeated[dim] = n  # repeat dim-th dimension n times
    return t.repeat(*dims_repeated)

class SummarizationLoss(nn.Module):
    # TODO
    """ Mask-filling with length penalty loss """
    def __init__(self):
        pass

    def forward(self, input, target):
        pass


class Summarizer(nn.Module):
    def __init__(self, pt_lm_path, fpt_lm_path, re_encode=True):
        """
        Arguments:

        pt_lm_path: 
        fpt_lm_path: 
        re_encode: If `True`, decode pt_lm's output and re-encode it with
                   fpt_lm's tokenizer. Since this operation is
                   non-differentiable, the re-encoding process is wrapped with
                   a StraightThroughEstimator, which propagates back the
                   gradients without altering them
        """

        super().__init__()
        self.re_encode = re_encode

        # TODO: models paths
        # Pretrained LM fine-tuned on summarization
        self.pt_lm = AutoModelForSeq2SeqLM.from_pretrained(pt_lm_path)
        self.outer_tokenizer = AutoTokenizer.from_pretrained(pt_lm_path)

        # Pretrained LM with frozen weights
        self.fpt_lm = AutoModelForMaskedLM.from_pretrained(fpt_lm_path)

        self.inner_tokenizer = AutoTokenizer.from_pretrained(
                fpt_lm_path,
                padding_side="right" # because bart uses absolute positional encoding #TODO: arg
                )

        freeze_(self.fpt_lm)

    def _re_encode(self, token_ids, discretize=False):
        # TODO: match special tokens (<pad>, <mask>, <s>, ecc.)
        #       with inner tokenizer (just set self.inner_tokenizer.pad_token =
        #       "<pad>", e.g.)

        tokens = self.inner_tokenizer(
                self.outer_tokenizer.batch_decode(
                    token_ids,
                    skip_special_tokens=False
                    ),
                return_tensors="pt",
                padding=True
                )["input_ids"].to(token_ids.device)
        if not discretize:
            tokens = tokens.float()
        return tokens


    def generate(self, input_ids, device="cpu", max_length=150, skip_eos=False, decode=False):
        """ A StraightThroughEstimator implementation of HuggingFace's generation """
        logits = self.pt_lm(input_ids.to(device)).logits

        def predict(logits):
            predicted_ids = logits.argmax(dim=-1)
            predicted_ids = predicted_ids[..., :max_length]
            if not skip_eos:
                eos_index = (predicted_ids == self.outer_tokenizer.eos_token_id).int().argmax(dim=-1)
                if eos_index:
                    predicted_ids = predicted_ids[..., :eos_index+1]

            if decode:
                return self.decode_outer(predicted_ids)
            return predicted_ids

        return StraightThroughEstimator.apply(logits, predict)



    def tokenize(self, text):
        return self.outer_tokenizer(
                text,
                return_tensors="pt",
                padding=True)

    def decode_outer(self, token_ids, **kwargs):
        return self.outer_tokenizer.batch_decode(token_ids, **kwargs)

    def decode_inner(self, token_ids, **kwargs):
        return self.inner_tokenizer.batch_decode(token_ids, **kwargs)

    # TODO:
    # TODO: should we just use a tokenized dataset?
    def make_inputs(self, text, device="cuda") -> tuple[Tensor, Tensor]:
        # TODO: refactor
        mask_token_id = self.inner_tokenizer.mask_token_id
        pad_token_id = self.outer_tokenizer.pad_token_id # TODO: should use inner_tokenizer

        inputs = self.tokenize(text)
        tokens = inputs["input_ids"].to(device)
        summary = self.summarize(text, device=device)

        extract = random_tokens_extract(tokens)
        extract = self._re_encode(extract, discretize=True)
        masked_extract, mask = random_mask(
                extract,
                mask_token_id,
                mask_p=0.05,
                pad_token_id=pad_token_id,
                return_mask=True
                )
        # ms = masked_extract.repeat(1, 1)
        # ts = extract.repeat(1, 1)

    # TODO: make masked and target optional and use make_inputs()
    def forward(
            self,
            inputs: Tensor,
            masked: Tensor,
            target: Tensor,
            append_context: bool = True
            ) -> Tensor:
        # TODO: should we just provide the extract and randomly mask it?
        """
        Produce mask-filling logits for the provided `masked` extract of `input_ids`

        inputs: (input_ids, attention_mask) where input_ids is the full text T
        masked: masked extract of input_ids
        target: ground-truth of `masked`
        append_context: if True, provide the MLM with a summary of the unmasked text
        """

        # TODO: ensure masked is encoded with the inner_tokenizer

        device = inputs.device

        # A simple separator (</s><s>)
        div = torch.tensor([[2, 0]]).to(device) # TODO: use a new token
        div = repeat_on_dim(div, 0, masked.shape[0])

        context_ids = masked

        # Prepend the generated summary to the input
        if append_context:
            # do_sample=False -> greedy decoding: is this thing differentiable?
            # TODO: max_length should be proportional to the full text
            # TODO: this might be non-differentiable since it produces integers
            # summary_ids = self.pt_lm.generate(inputs, do_sample=False, max_length=150)

            summary_ids = self.generate(inputs, max_length=150, device=device)
            
            # Re-encode output of pt_lm to match the tokenizer of fpt_lm
            if self.re_encode:
               summary_ids = StraightThroughEstimator.apply(summary_ids, self._re_encode)
                # summary_ids = self._re_encode(summary_ids)

            # summary_ids = summary_ids.float()
            # summary_ids.requires_grad = True

            summary_ids = repeat_on_dim(summary_ids, 0, masked.shape[0])

            context_ids = torch.cat([summary_ids, context_ids], dim=-1) # TODO: add a unique token to divide summary from masked extract
            target = torch.cat([summary_ids, target], dim=-1) # TODO: add a unique token to divide summary from masked extract

        mask = context_ids == self.inner_tokenizer.mask_token_id

        labels = target.clone()
        labels[~mask] = -100 # NLLLoss ignore_index=-100
        labels.to(device, dtype=torch.long)

        # TODO: inputs should be the masked extract + summary context
        #       labels should be the unmasked extract
        full_inputs = {"input_ids": context_ids, "labels": labels.long()} 
    
        print(full_inputs)
        outputs = self.fpt_lm(**full_inputs)
        outputs.requires_grad = True

        return outputs

    def summarize(self, text: list[str], device="cpu") -> list[str]:
        """ Produce a summary for each input text in the batch """

        input_ids = self.tokenize(text)["input_ids"].to(device)

        with torch.no_grad():
            output_ids = self.pt_lm.generate(input_ids, num_beams=2, max_length=150)
            output = self.decode_outer(output_ids)

        return output

    def train(self):
        pass


