#set heading(numbering: "1.")
#set page(
  paper: "us-letter",
  header: align(right)[],
  numbering: "1",
)
#set par(justify: true)
#set text(
  // font: "New Computer Modern",
  size: 11pt,
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
Sections #link(<sec:training>)[7.2] and #link(<sec:inference>)[7.3].

= Prior work <prior-work>
#strong[TODO]

= Data <data>
#strong[TODO]

- Tokenizer used: BART byte-encoding \~50k vocab size
- Dataset description (WMT14 EN-DE): 50% used for memory reasons
- Max length set to 256
- Padding on the right
- Padding trimmed within batches
- Average sequence length in training dataset
- Training, validation, test split
- Use of test dataset

= Architecture <architecture>
= Hardware <hardware>
== CPU bound <cpu-bound>
= Experiments <experiments>
== Docker container (NVCR) <docker-container-nvcr>
== Mixed precision training <mixed-precision-training>
= Results and Analysis <results-and-analysis>
== Model performance <model-performance>
#strong[TODO]

#figure(
  align(center)[#table(
    columns: 3,
    align: (col, row) => (center, center, center,).at(col),
    inset: 6pt,
    [Model], [PPL (test)], [BLEU (test)],
    [Transformer], [#strong[3.41]], [29.92],
    [Linformer (k\=32)], [3.96], [#strong[30.08]],
    [Linformer (k\=64)], [3.84], [27.74],
  )],
  caption: [Linformer performance against a vanilla Transformer model on the WMT14 EN-DE (test) dataset. The Linformer
  has slightly worse perplexity than the Transformer, but their BLEU scores are comparable.],
) <tab:performance>

== Training time <sec:training>
== Inference time <sec:inference>
= Conclusions <conclusions>

#bibliography("biblio.bib")
