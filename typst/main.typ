#import "@preview/cetz:0.3.0"
#import "@local/cetz-plot:0.1.0": plot, chart
#import "@preview/showybox:2.0.1": showybox

#set cite(form: "normal", style: "alphanumeric")
#set figure(placement: auto)

#set heading(numbering: "1.")
#set page(
  paper: "us-letter",
  header: align(right)[],
  numbering: "1",
)
#set par(justify: true)
#set text(
  font: "New Computer Modern",
  size: 11pt,
)

#let appendix(body) = {
  set heading(numbering: "A.", supplement: [Appendix])
  counter(heading).update(0)
  body
}

#align(center, text(17pt)[
  *Evaluating Linformer's performance on the WMT14 EN-DE machine
translation task*
])
#grid(
  columns: (1fr),
  align(center)[
    Marco Pampaloni \
    Department of Computer Science \
    #link("m.pampaloni2@studenti.unipi.it")
  ]
)



= Introduction <introduction>
The Transformer architecture, since its introduction in 2017 @vaswani2017, has revolutionized the field of natural
language processing (NLP), reaching state of the art in various downstream tasks. Despite its massive parallelism
capabilities, the Transformer struggles with longer sequence lengths due to its attention mechanism, which scales
quadratically with the number of input tokens. This is a problem at both training and inference time.

For this reason, there has been a huge research effort in recent years to develop faster attention mechanisms, either
exploiting the IO properties of the hardware these models run on @flashAttention2022, or by approximating the result of
scaled dot product attention (SDPA).

The #emph[Linformer] architecture @linformer2020 is an example of the latter approach. The authors first empirically
show that the attention matrix is often low-rank, meaning that it could be approximated with an SVD decomposition by
only keeping the largest singular values. This would of course introduce additional asymptotical complexity to the
method, so the authors propose the adoption of linear projections on the keys and values matrices $K , V$ to reduce
their dimensionality and drive the computational complexity of the attention mechanism from $O(n^2)$ to $O(k n)$. The
authors further show that the choice of $k$ does not depend on the sequence length $n$, so that the scaling can be
considered linear in $n$.

The standard multi head attention (MHA) introduced by @vaswani2017 is computed as follows

$
"MultiHead"(Q , K , V) & = "Concat"("head"_1 , dots.h , "head"_h) W^O\
"where" "head"_i  & = "Attention"(Q W^Q , K W^K , V W^V)\
& = underbrace("softmax"(frac(Q W^Q (K W^K)^T, sqrt(d_"model"))), P_i) V W^V
$

The Linformer attention first projects the key and value matrices into a space of dimension $bb(R)^(k times d)$ with
projection matrices $E , F in bb(R)^(k times n)$ and then computes MHA as before:

$
overline("head")_i = "Attention"(Q , E_i K W_i^K , F_i V W_i^V)
$

This produces an attention matrix $overline(P)_i in bb(R)^(n times k)$, which is computed in time linear with $n$. Since
the projection matrices $E , F$ are fixed in size before training, an obvious donwside of this approach is that the
maximum sequence length $n$ has to be known beforehand and the model cannot then scale to inputs with more tokens than
this value. One workaround is to set the maximum sequence length $n$ to a large number and handle shorter inputs by
slicing the projection matrices along their columns before multiplying them with the inputs $K$ and $V$.

This work aims at replicating the results of @vaswani2017 and @linformer2020 and comparing the two architecture on the
WMT14 EN-DE machine translation task.

== Masking <masking>
In a standard Transformer architecture, sequences are usually batched together in order to exploit the model’s
parallelism. This requires padding to be applied to the batch, but pad tokens should not contribute to the attention’s
output. This can be achieved by masking the attention matrix, setting zeroing out its elements in correspondence with
input pad tokens. This if often done by setting each corresponding element to $- oo$ prior to applying the row-wise
softmax.

In Linformer this method cannot be applied, as the attention matrix is linearly projected to a lower-dimensional space.
Instead, we apply masking by zeroing out elements in $Q , K , V$ corresponding to pad tokens. This ensures that the pad
tokens do not contribute to the final attention result.

== Causal masking <causal-masking>
The standard Transformer was trained on the WMT14 EN-DE machine translation dataset, adopting an encoder-decoder
architecture. The self-attention mechanism in the decoder’s layers need to employ causal masking in order for future
(right) tokens not to contribute to the output of past positions. In the Transformer this again can be achieved by
masking, in this case by adding an upper triangular matrix filled with $-oo$ values to the pre-softmax attention
matrix.

Linformer, which was originally developed for encoder-only architectures, does not allow for causal masking, because
however you mask the resulting pre-activation attention matrix, future tokens leak into the past due to the linear
projections of the $K$ and $V$ matrices.

The Linformer attention mechanism thus cannot be applied in the self-attention layers of the decoder, while it can
safely be used in the cross-attention stages because of the lack of causal requirements. This hinders the full scaling
potential of the encoder-decoder Linformer architecture, which is empirically shown in
@training-time and @inference-time.

= Data <data>
As in @vaswani2017, we used the WMT14 EN-DE @bojar-etal-2014-findings dataset comprised of about 4.5 million sencente
pairs. For ease of use, we used the data hosted on a Kaggle repository
#footnote("https://www.kaggle.com/datasets/mohamedlotfy50/wmt-2014-english-german"), which conveniently collects all the
english to german sentences in a single CSV file for training. Additionally, a validation (dev) and test dataset are
provided. The validation set has not been used.

Because of resources constraints, we could not use or preprocess the entire dataset while keeping it in memory. Applying
preprocessing on the fly during training would have slowed down the experiment significantly, rendering its results
meaningless. Instead, we used half of the original training dataset and used the last 1% of it as a validation set.

We used the pretrained Huggingface #footnote("https://www.huggingface.co") implementation of the BART @bart2019
tokenizer, which employs Byte Pair Encoding (BPE) and was applied to the dataset prior to training as a preprocessing
step. The vocabulary of the tokenizer included about 50k tokens, in contrast to the vocabulary size of 37k from
@vaswani2017. We truncated each sentence in the dataset to 256 tokens, padding on the right when necessary to enable
training in batches. This should not cause meaningful degradation in model performance since the average sequence length
after tokenization of the training dataset is much lower for both source and target sentences.
@tab-data-stats shows post-tokenization statistics for the training dataset without padding applied.

#figure(
  table(
    columns: 4,
    [*Input*], [*Average Length*], [*Min Length*], [*Max Length*],
    [source],  [31], [3], [10864],
    [target],  [59], [3], [8318],
  ),
  caption: [Statistics for the training dataset, reporting average (rounded up to the nearest integer), minimum and
  maximum sequence lengths after tokenization for source and target sequences.]
) <tab-data-stats>

During training, each batch is randomly sampled and trimmed to the length of the longest non-padded sequence within the
batch, significantly improving performance for batches with many short sequences (and, consequently, a high number of
<pad> tokens).


The validation dataset is only used to log metrics during training and for scheduling the learning rate. The test
dataset is instead used during the last evaluation step of each experiment.


= Architecture <architecture>

This work focused on two architectures: the vanilla Transformer model and the Linformer. Both employed an
encoder-decoder structure following @vaswani2017:

- 6 stacked encoder layers composed by Multi-Head self-attention, followed by a pointwise multi layer perceptron;
- #box[6 stacked decoder layers composed by Masked Multi-Head self-attention, followed by Multi-Head cross-attention and
a pointwise multi layer perceptron.]

In the standard Transformer model, each attention layer employs the scaled dot-product attention (SDPA) introduced by
@vaswani2017, both for the encoder's self-attention and the decoder's self-attention and cross-attention.
The Linformer instead uses the linearized attention mechanism proposed by @linformer2020 everywhere except in the
self-attention stage of the decoder's layers, because of causality requirements (See @causal-masking).

The implementation of Linformer we provide employs shared projection matrices $E, F$ across heads, but different for
each layer. This is one of the variants proposed by the authors @linformer2020.

Both the standard Transformer and the Linformer adopt residual connections and Layer Normalization applied after each
attention block and feed forward layer as in @vaswani2017. Sinusoidal positional encoding and learned embeddings have
been used for both architectures.

@tab-hyper shows the hyperparameters that have been set for both models, including every variant of Linformer tested.

#figure(
  [
    #table(
      align: left,
      columns: 3,
      [*Parameter*], [*Value*], [*Description*],
      [$d_"model"$], [512], [Size of each embedding vector],
      [$h$], [8], [Number of heads in multi-head attention (MHA)],
      [$d_k, d_v$], [64], [Inner dimension of key and value vectors per head],
      [$d_"mlp"$], [2048], [Hidden layer dimension of each pointwise MLP]
    )
  ],
  caption: [Hyperparameters shared by the tested models]
) <tab-hyper>

One last notable change made to the architectures is the choice of the vocabulary size: BART tokenizer's vocabulary has
been resized to the next multiple of 8 in order to fully exploit the Tensor Cores
#footnote[#link("https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/")] of the Nvidia GPU used during
training (See @hardware), which accelerate matrix products when their sizes are divisible by 8.

== Text generation <text-generation>
In contrast to @vaswani2017, autoregressive machine translation was implemented through greedy decoding
@sutskever2014sequencesequencelearningneural. We limited the output's length to 256 tokens, but terminate early when
each sequence in the batch has produced an EOS token.

== Implementation <implementation>
The Transformer and Linformer models have been implemented using PyTorch `2.5.0a0+872d972e41` following the details presented in
@vaswani2017 and @linformer2020. Inspired by PyTorch's and Hugginface's APIs, we adopted a compositional approach to
model definition that allows to easily swap attention mechanisms between Transformer architectures created using a
general backbone structure.

On top of the standard Transformer, a series of useful wrappers have been implemented that allow the model to
transparently tokenize input strings, embed tokens, apply positional encodings and produce output token distributions.

The following code defines an encoder-decoder Transformer architecture equivalent to that described in @architecture.

#showybox(
  frame: (body-color: black.lighten(99%)),
  ```python
  attn = MultiHeadAttention(dim=512, n_heads=8)
  encoder = TransformerEncoder(TransformerEncoderLayer(dim=512, mlp_dim=2048, attn), n_layers=6)
  decoder = TransformerDecoder(TransformerDecoderLayer(dim=512, mlp_dim=2048, attn), n_layers=6)
  transformer = NLPTransformer(encoder = encoder, decoder = decoder, vocab_size = 50272)
  transformer = LanguageModelingHead(transformer)
  ```
)

The ```python NLPTransformer()``` module internally allocates learnable embeddings and applies positional encoding. The
```python LanguageModeleingHead()``` module outputs a probability distribution over the vocabulary defined by the
tokenizer and it provides basic generational functionalities such as greedy decoding. To define a Linformer model one
can simply swap the ```python MultiHeadAttention()``` module with ```python LinformerAttention()```. Note that although
the defined APIs share similarities with those of PyTorch, they have been implemented from scratch, along with
everything else except the BART tokenizer.

= Hardware <hardware>
The experiments have been carried out locally on a system running a single Nvidia RTX 3090 GPU with 24GB of GDDR6X VRAM
and an Intel i7 4770k CPU overclocked at 4.4GHz. The system's memory amounted to 16GB of DDR3 RAM.
== CPU bound <cpu-bound>
Given the dated system components, the experiments were bottlenecked by the CPU, which could not keep the GPU usage at
100% most of the time during training, hovering near the 96-98% range of utilization instead.

= Experiments <experiments>
Each experiment was conducted independently on the same machine within a Docker container based on Nvidia's PyTorch NGC
#footnote[#link("https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch")] version 24.08, which provides an
optimized set of libraries for efficient training and inference on GPUs. Additionally, mixed precision training
#footnote[#link("https://developer.nvidia.com/automatic-mixed-precision")] was employed throughout, reducing memory
usage and significantly speeding up computations.

We trained each model using a negative log-likelihood loss function with teacher forcing, computed pointwise across the
batch and averaged. At the end of each epoch, we logged the loss and perplexity score calculated on the validation
dataset. For both the logging of training and validation metrics, as well as model checkpointing, custom callback
routines were implemented using Weights & Biases #footnote[#link("https://wandb.ai/site/")]. These callbacks enabled
automatic storage of metrics, system information and hyperparameters, and saved model weights whenever validation
metrics improved during training.

Since training sequence lengths varied due to padding trimming (see @data), we used a batch size of 88 input sequences
for each run. Using the numbers reported in @tab-data-stats, this results in an average number of respectively #(31*88)
source and #(59*88) target tokens per batch. In the worst case the maximum number of tokens per batch is #(256*88),
which is close to what was used in @vaswani2017. This allowed for high memory utilization without running into
allocation errors.

The learning rate schedule followed a similar approach to @vaswani2017, starting with a linear warmup over 4,000 steps
from $10^(-7)$ to $10^(-4)$. After the warmup, we applied a cosine annealing strategy, gradually decreasing the learning
rate to $10^(-6)$ by the final epoch. Each experiment ran for 10 training epochs.

= Results and Analysis <results-and-analysis>
The following section presents the performance results of Linformer variants against a standard Transformer on the WMT14
EN-DE task.
== Model performance <model-performance>

@tab-performance shows that the Linformer performs comparably to the standard Transformer model on both tested metrics,
scoring worse perplexities on the test dataset, but showing similar BLEU scores. Even though perplexity intuitively
drops as the parameter $k$ grows, the BLEU score seems to worsen. This variation is somewhat expected since
@vaswani2017 actually performed their BLEU evaluations on the test dataset with an ensemble of models computed by
averaging many training checkpoints, ultimately lowering variance across experiments.
#figure(
  placement: top,
  [#table(
    columns: 3,
    [*Model*], [*PPL (test)*], [*BLEU (test)*],
    [Transformer], [*3.41*], [29.92],
    [Linformer (k=32)], [3.96], [*30.08*],
    [Linformer (k=64)], [3.84], [27.74],
  )],
  caption: [Linformer performance against a vanilla Transformer model on the WMT14 EN-DE (test) dataset. The Linformer
  has slightly worse perplexity than the Transformer, but their BLEU scores are comparable.],
) <tab-performance>


The perplexity scores evaluated at each training epoch are illustrated in @perplexity-curves for both the standard
Transformer and each tested Linformer variant.

#figure(
  image("figures/perplexity_dev.svg", height: 220pt),
  caption: [Perplexity curves over training epochs computed over the validation (dev) dataset.]
) <perplexity-curves>

== Translation examples <translation-examples>
In the following section, we provide a selection of translations generated from test samples, showcasing the performance
of each model that was evaluated. This aims to offer a comparison of the translation quality produced by the different
models tested. More translation examples can be found in @appendix-a.

#let example_box = showybox.with(
  width: 100%,
  frame: (
    body-color: white.darken(1%),
    title-color: black.lighten(20%),
    // border-color: red.darken(50%),
    radius: 5pt,
  ),
  title-style: (
    // color: black,
    weight: "regular",
    align: center
  ),
  shadow: (
    offset: 0pt,
  )
)

//NOTE: this function is deliberately redundant (in case I decide to change how to display the examples later on)
#let load_examples(path, n: 2, start: 0, title: [Title]) = {
  set text(size: 8pt)
  let examples = csv(path, row-type: dictionary).slice(start, count: n)
  let boxes = ()

  for (input, candidate, reference) in examples {
    boxes.push(
      example_box(
        title: title,
        [*Input*: ] + input,
        [*Candidate*: ] + candidate,
        [*Reference*: ] + reference,
      )
    )
  }

  grid(
    columns: n,
    align: center,
    row-gutter: 1em,
    ..boxes
  )
}

#let transformer_example(index: 0) = {
  load_examples("./artifacts/xf9i0ea3_out.csv", n: 1, start: index, title: [Transformer])
}

#let linformer_example(k, index: 0) = {
  if k == 32 {
    load_examples("./artifacts/ud1t16uq_out.csv", n: 1, start: index, title: [Linformer $k=32$])
  }
  else {
    load_examples("./artifacts/s3edn4nb_out.csv", n: 1, start: index, title: [Linformer $k=64$])
  }
}

// Show a single example for each model in a grid like [[lin32, lin64], [transformer]]
#let examples_grid(index: 0, p: 100%) = {
  grid(
    columns: 2,
    align: center,
    gutter: 1em,
    linformer_example(32, index: index),
    linformer_example(64, index: index),
    grid.cell(
      colspan: 2,
      transformer_example(index: index),
    )
  )
}

#box({
  [_Example Translation \#1_:]
  examples_grid(index: 0)
  linebreak()
})

#box({
  [_Example Translation \#2_:]
  examples_grid(index: 1)
})

== Training time <training-time>
The time required to fully train a model is of paramount importance when considering the cost of scaling models to
billions of parameters and large context windows. Although fully applying a "linearized" attention mechanism throughout
an encoder-decoder architecture is not possible, Linformer should still provide a measurable improvement in the
wall-clock times required to train it compared to a standard Transformer. @tab-training-time shows the average time
required to compute a training batch (including time required to compute validation steps).
#figure(
  table(
    columns: 3, 
    [ *Model*], [*Time (s/batch)*], [*Total Duration (hr)*],
    [Transformer], [0.221], [14.3],
    [Linformer ($k = 32$)], [*0.205*], [*13.2*],
    [Linformer ($k = 64$)], [0.209], [13.4],
  ),
  caption: [Time required by the various model variants to compute a training step (forward and backward passes) and
  total experiment duration.]
) <tab-training-time>

Linformer marginally improves training times, reducing them by about 7% when using $k=32$ and by 5% when $k=64$.

== Inference time <inference-time>

#let plot_data(labels: (), tick-step: 1.0, decimals: 2, size: (8, 8), ..csv_files) = {
  let data = ()
  let ticks = none
  for path in csv_files.pos() {
    let times = csv(path, row-type: array)

    // times.at(0) is the CSV index
    let ys = times.at(1).map(x => float(x))
    let xs = range(times.at(1).len())
    ticks = xs.zip(times.at(0))
    data.push(xs.zip(ys))
  }

  return cetz.canvas({
    plot.plot(
      size: size,
      x-label: "Seq. length/batch size",
      y-label: "Time (s)",
      x-tick-step: none,
      y-decimals: decimals,
      y-tick-step: tick-step,
      x-ticks: ticks, {
        for i in range(data.len()) {
          let label = none
          if i < labels.len() { label = labels.at(i) }
          plot.add(data.at(i), label: label, mark: "o")
        }
      }
    )
  })
}

To experimentally verify the computational efficiency of the Linformer architecture, we ran inference for each model
variant on batches of randomly generated data, varying the sequence length while keeping the total number of input
tokens per batch constant. After a warmup phase, each inference step was repeated 10 times and the execution times were
averaged. The same architectures described in @architecture were used; however, in this case, we were not restricted to
testing only trained models, as the output was discarded. The results of this analysis are summarized in
@fig-enc-dec-times.

#figure(
  scale(
    plot_data(
      size: (8, 5),
      tick-step: auto,
      labels: ("Transformer", "Linformer, k=32"),
      "./artifacts/perf_vanilla.csv",
      "./artifacts/perf_lin_k32.csv"
    ), x: 80%, y: 80%
  ),
  caption: [Comparison of scaling times between Linformer and Transformer with an encoder-decoder architecture.
  Different choices of the parameter $k$ result in the same scaling, with differences in execution time falling within
  margin of error. This is because of the decoder's bottleneck which still requires the exact attention mechanism to be
  carried out.]
) <fig-enc-dec-times>

Despite Linformer's greater efficiency, the curves in @fig-enc-dec-times indicate that the adopted encoder-decoder
architecture does not scale linearly with the input sequence length, and in fact exhibits the same computational
complexity as the Transformer model, up to a multiplicative constant. This is due to the use of the standard MHA in the
decoder's self-attention layers.

In order to reveal the general scaling behaviour of the Linformer model, we carry out the same tests as before with an
encoder-only architecture. The results are shown in @fig-enc-times.

#figure(
  grid(
    columns: 1,
    scale(
      plot_data(
        size: (8,5),
        tick-step: auto,
        decimals: 3,
        labels: ("Transformer", "Linformer, k=128"),
        "./artifacts/perf_vanilla_encoder_only.csv",
        "./artifacts/perf_lin_k128_encoder_only.csv",
      ), x: 80%, y: 80%
    ),
    scale(
      plot_data(
        size: (8, 5),
        tick-step: 0.0011,
        decimals: 3,
        labels: ("Linformer, k=32", "Linformer, k=64", "Linformer, k=128"),
        "./artifacts/perf_lin_k32_encoder_only.csv",
        "./artifacts/perf_lin_k64_encoder_only.csv",
        "./artifacts/perf_lin_k128_encoder_only.csv",
      ), x: 80%, y: 80%
    ),
  ),
  caption: [Scaling times of Linformer and Transformer with an encoder-only architecture. On the Left the standard
  Transformer is compared against the Linformer with $k=128$. Linformer maintains constant execution time while varying
  the sequence. On the Right, various choices of parameters $k$ are shown.] 
) <fig-enc-times>

The wall-time plots clearly show that, as an encoder-only architecture, Linformer's computational complexity does not
depend on the sequence length, which drastically improves inference performance as the context $n$ grows, compared to a
standard Transformer implementation.


= Conclusions <conclusions>

Linformer's performance showed to be comparable with that of a standard Transformer on the WMT14 english to german
machine translation task, while providing measurable efficiency gains over the former model, although they were limited
by the decoder's bottleneck, which still required the use of the full attention matrix due to causality.

Investigating encoder-only architectures adopting the Linformer attention mechanism showed significantly faster
inference speeds over the vanilla Transformer model, particularly for longer sequences, suggesting that encoding tasks
such as sentence classification might be a better benchmark for such models. 

Encoder-decoder Transformers requiring causality masking could probably benefit more from other linearization
approaches.

#pagebreak()
#bibliography("biblio.bib")
#pagebreak()

#show: appendix
= Translation Examples <appendix-a> \
\
#for i in range(2, 12) {
  box({
    [_Example_ \##(i+1)]
    examples_grid(index: i)
    linebreak()
  }
)
}
