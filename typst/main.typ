#import "@preview/cetz:0.3.0"
#import "@local/cetz-plot:0.1.0": plot, chart
#import "@preview/showybox:2.0.1": showybox

#set heading(numbering: "1.")
#set page(
  paper: "us-letter",
  header: align(right)[],
  numbering: "1",
)
#set par(justify: true)
#set text(
  // font: "New Computer Modern",
  size: 10pt,
)

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

=== Causal masking <causal-masking>
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
Sections #link(<sec:training>)[7.2] and #link(<sec:inference>)[7.3]. // TODO: fix references

= Prior work <prior-work>
#strong[TODO]

= Data <data>
As in @vaswani2017, we used the WMT14 EN-DE @bojar-etal-2014-findings dataset comprised of about 4.5 million sencente
pairs. For ease of use, we used the data hosted on a Kaggle repository
#footnote("https://www.kaggle.com/datasets/mohamedlotfy50/wmt-2014-english-german"), which conveniently collects all the
english to german sentences in a single CSV file for training. Additionally, a validation (dev) and test dataset are
provided. The validation set has not been used.

Because of resources constraints, we could note use or preprocess the entire dataset while keeping it in memory.
Applying preprocessing on the fly during training would have slowed down the experiment significantly, rendering its
results meaningless. Instead, we used half of the original training dataset and used the last 1% of it as a validation
set.

We used the pretrained Huggingface #footnote("https://www.huggingface.co") implementation of the BART @bart2019
tokenizer, which employs Byte Pair Encoding (BPE) and was applied to the dataset prior to training as a preprocessing
step. The vocabulary of the tokenizer included about 50k tokens, in contrast to the vocabulary size of 37k from
@vaswani2017. We truncated each sentence in the dataset to 256 tokens, padding on the right when necessary to enable
training in batches. Each batch is randomly sampled from the training dataset and trimmed to the length of the longest
non-padded sequence within the batch, significantly improving performance for batches with many short sequences (and,
consequently, a high number of <pad> tokens).

The validation dataset is only used to log metrics during training and for scheduling the learning rate. The test
dataset is instead used during the last evaluation step of each experiment.

#strong[TODO]
- Average sequence length in training dataset

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

Both the standard Transformer and the Linformer adopt residual connections and Layer Normalization applied after each
attention block and feed forward layer as in @vaswani2017. Sinusoidal positional encoding and learned embeddings have
been used for both architectures.

@tab-hyper shows the hyperparameters that have been set for both models, including every variant of Linformer tested.

#figure(
  [
    #table(
      columns: 3,
      [Hyperparameter], [Value], [Description],
      [$d_"model"$], [512], [Size of each embedding vector],
      [$h$], [8], [Number of heads in multi-head attention (MHA)],
      [$d_k, d_v$], [64], [Inner dimension of key and value vectors per head],
      [$d_"mlp"$], [2048], [Hidden layer dimension of each pointwise MLP]
    )
  ],
  caption: [Hyperparameters shared by the tested models]
) <tab-hyper>

// - $d_"model" = 512$: the size of each embedding vector;
// - $h = 8$: the number of heads in MHA;
// - $d_k = d_v = d_"model" / h = 64$: the inner dimension of key and value vectors for each head in MHA;
// - $d_"mlp" = 2048$: the hidden layer dimension of each pointwise MLP.

One last notable change made to the architectures is the choice of the vocabulary size: BART tokenizer's vocabulary has
been resized to the next multiple of 8 in order to fully exploit the Tensor Cores #footnote[tensor cores: #strong[TODO]]
of the Nvidia GPU (See @hardware) used during training, which accelerate matrix products when their sizes are divisible
by 8.

#strong[TODO]:
- Architecture diagram

#cetz.canvas({
  import cetz.draw: *
  rect((0, 0), (2, 1), radius: 0.1, fill: orange, name: "enc-mha")
  content((rel: (-1, -0.5)), [Hello World])
})


= Hardware <hardware>
The experiments have been carried out locally on a system running a single Nvidia RTX 3090 GPU with 24GB of GDDR6X VRAM
and an Intel i7 4770k CPU overclocked at 4.4GHz. The system's memory amounted to 16GB of DDR3 RAM.
== CPU bound <cpu-bound>
Given the dated system components, the experiments were bottlenecked by the CPU, which could not keep the GPU usage at
100% most of the time during training, hovering near the 96-98% range of utilization instead.

#strong[TODO]:
- Show CPU/GPU utilization graph during training (wandb)

= Experiments <experiments>

#strong[TODO]:
- Docker container (NVCR)
- Mixed precision training
- Weights and Biases
- Logged metrics and validation
- Training times (maybe drop later section)

= Results and Analysis <results-and-analysis>
The following section presents the performance results of Linformer variants against a standard Transformer on the WMT14
EN-DE task.
== Model performance <model-performance>

@tab-performance shows that the Linformer performs comparably to the standard Transformer model on both tested metrics,
scoring worse perplexities on the test dataset, but showing similar BLEU scores. Even though perplexity intuitively
drops as the parameter $k$ grows, the BLEU score seems to worsen. This variation is somewhat expected since
@vaswani2017 actually performed their BLEU evaluations on the test dataset with an ensemble of models computed by
averaging many training checkpoints, ultimately lowering.
#figure(
  [#table(
    columns: 3,
    [Model], [PPL (test)], [BLEU (test)],
    [Transformer], [#strong[3.41]], [29.92],
    [Linformer (k=32)], [3.96], [#strong[30.08]],
    [Linformer (k=64)], [3.84], [27.74],
  )],
  caption: [Linformer performance against a vanilla Transformer model on the WMT14 EN-DE (test) dataset. The Linformer
  has slightly worse perplexity than the Transformer, but their BLEU scores are comparable.],
) <tab-performance>

#strong[TODO]:
- Show perplexity validation curves
- Show training and validation loss curves

== Translation examples <translation-examples>

*TODO*:
- Example translations for Linformer variants
- Automatically load strings from CSV files
- Refactor into function

#showybox(
  [*Example Translation (Transformer) *],
  [*Input*: The school yard renovation was originally planned back in 2008/2009, however, high unplanned expenses meant that the
  work had to be pushed back.],
  [*Candidate*: Die Renovierungsarbeiten waren ursprünglich im Jahr 2008/2009 geplant, hingegen mit hohen ungeplanten Ausgaben, die
  zurückzufahren mussten.],
  [*Reference*: Ursprünglich war die Schulhofsanierung sogar schon in den Jahren 2008/2009 geplant, doch hohe unplanmäßige Ausgaben
  brachten eine Verschiebung.]
)

== Training time <sec:training>
== Inference time <sec:inference>

//TODO: size argument
#let new_plot_data(labels: (), tick-step: 1.0, decimals: 2, ..csv_files) = {
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
      size: (8,5),
      x-label: "Seq. length / batch size",
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

#figure(
  scale(
    new_plot_data(
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
)

#figure(
  grid(columns: 2,
    scale(
      new_plot_data(
        tick-step: auto,
        decimals: 3,
        labels: ("Transformer", "Linformer, k=128"),
      "./artifacts/perf_vanilla_encoder_only.csv",
        "./artifacts/perf_lin_k128_encoder_only.csv",
      ), x: 60%, y: 60%
    ),
    scale(
      new_plot_data(
        tick-step: 0.0011,
        decimals: 3,
        labels: ("Linformer, k=32", "Linformer, k=64", "Linformer, k=128"),
        "./artifacts/perf_lin_k32_encoder_only.csv",
        "./artifacts/perf_lin_k64_encoder_only.csv",
        "./artifacts/perf_lin_k128_encoder_only.csv",
      ), x: 60%, y: 60%
    ),
  ),
  caption: [Scaling times of Linformer and Transformer with an encoder-only architecture. On the Left the standard
  Transformer is compared against the Linformer with $k=128$. Linformer maintains constant execution time while varying
  the sequence. On the Right, various choices of parameters $k$ are shown.] 
)


= Conclusions <conclusions>

#bibliography("biblio.bib")

