# Linformer: Self-Attention with Linear Complexity 
This repository holds a PyTorch implementation of Linformer (Wang et al., 2020) and offers a comparison against a vanilla Transformer.

# Dataset
You can the pretokenized dataset from [here](https://huggingface.co/datasets/crybot/WMT14-en-de-tokenized), or you can download the original raw dataset from 
this [Kaggle repository](https://www.kaggle.com/datasets/mohamedlotfy50/wmt-2014-english-german?select=wmt14_translate_de-en_train.csv). For the latter, you need to run the pretokenization phase with:

```bash
./build.sh && ./run.sh src/pretokenize_dataset.py
```

# Usage
The implementation is conveniently dockerized so that all dependencies can be replicated on any machine with a compatible CUDA installation.
Just run
```bash
./build.sh
```
to build the image.

The `./run.sh` script then let's you choose which script to run.

## Training
To train the model, provide a YAML configuration file holding the training settings to the `./run.sh` script. Under `configs/` you can find three example configurations:
- `lin_k32_training.yaml`
- `lin_k64_training.yaml`
- `vanilla_training.yaml`

To train a Linformer with `k=32`:
```bash
./run.sh src/train_my.py --config configs/lin_k32_training.yaml
```

Training logs validation metrics to Weights & Biases and the build script looks for a `~/.wandb_secret` file containing your API key under your home directory by default. You can change this behaviour by editing `build.sh`.

You can also disable Weights & Biases logging altogether by commenting out the callbacks in `src/train_my.py`, but you will not be able to automatically checkpoint the model for later evaluation.
The callbacks to comment out are: `wandb_callback, checkpoint_callback`.

## Comparing performance
To compare the inference performance of different models you can use `src/compare_performance.py`. For example, running the following command tests the inference times of a Linformer with `k=128` with an encoder-only architecture:
```bash
./run.sh src/compare_performance.py --config configs/perf_lin_k128_encoder_only.yaml
```

## Evaluation
Evaluating the models requires access to a Weights & Biases account. By default the scripts look for a project named `HLT`, but you can change it by editing `src/evaluate.py`.
Running `./run.sh src/evaluate.py --checkpoint <wandb-checkpoint>` downloads a checkpoint from Weights & Biases and evaluates it on the test dataset, computing BLEU and perplexity metrics.
