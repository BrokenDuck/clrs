# CLRS in PyTorch

A PyTorch fork of [google-deepmind/clrs](https://github.com/google-deepmind/clrs),
with published CLRS-30 dataset loading, the original NumPy reference algorithms,
and an additional hinted min-plus repeated-squaring APSP task.
The upstream base revision is recorded in `UPSTREAM_COMMIT`.
Apache-2.0 licensing and upstream attribution are preserved.

## Installation

Python 3.12+ is supported; the parent project uses Python 3.14.

```bash
pip install -e '.[test]'
OMP_NUM_THREADS=1 pytest -q
```

The runtime uses PyTorch and NumPy. JAX, Haiku, Optax, Chex, TensorFlow, and
TensorFlow Datasets are not dependencies. The standalone `tfrecord` package
supplies the protobuf definitions for reading the published archive.

## Dataset access

```python
import clrs

# Exact published records; downloads the official ~86 MB archive on first use.
loader, count, spec = clrs.create_dataset(
    './data/clrs', 'floyd_warshall', 'train', batch_size=8,
    source='published')
feedback = next(iter(loader))

# New trajectories with reproducible sampling and a provenance-keyed NPZ cache.
loader, count, spec = clrs.create_dataset(
    './data/clrs', 'min_plus_apsp', 'train', batch_size=8,
    source='generated', num_samples=1000, length=16, seed=1)
```

`create_dataset` returns `(torch.utils.data.DataLoader, count, spec)`. Loaders
are finite; iterate with ordinary Python, restarting each epoch. Training
loaders shuffle by default; validation/test do not. `loader_seed` controls
shuffle order, while `seed` controls generated examples. `num_workers` supports
parallel loading without duplicating samples. One dataset has one graph size;
create separate loaders for different sizes.

`Feedback(features, outputs)` and `Features(inputs, hints, lengths)` retain
CLRS metadata. Each `DataPoint` has `name`, `location`, `type_`, and `data`.
Tensor inputs/outputs are `[batch, ...]`; hints are `[time, batch, ...]` and
padded within a batch. Integer `lengths` excludes padding. Numeric precision is
retained in the loader; model encoders use float32 and pointers become integer
indices. Use `clrs.to_torch(feedback, device)` to transfer a batch. Custom
DataLoaders should use `clrs.collate_feedback`.

For direct NumPy sampling, `clrs.build_sampler(name, num_samples, seed=...,
length=...)` remains available, including `num_samples=-1` for online sampling.
All original tasks remain available. `CLRS_30_ALGS` stays unchanged; select
`min_plus_apsp` explicitly.

Published data uses the archive's values, feature shapes, and dtypes. In
particular, TFRecord FloatList values have float32 storage precision even when
metadata declares float64. Published quickselect uses its historical spec,
without the later `pivot` hint: always use the spec returned by the loader.
Generated data uses the current spec. No missing published hints are invented.
`download=False` requires already-extracted records; `source='generated'` never
substitutes for published data. Generated caches include split, seed, size,
sampler options, format version, and upstream revision; bump `CACHE_VERSION`
when changing generation semantics. Published records are indexed by file offset
and read on demand. The downloaded archive is retained, and only requested task
splits are extracted.

## Min-plus APSP and hints

`clrs.algorithms.min_plus_apsp(A)` returns `(Pi, probes)`. It implements the
[min-plus repeated-squaring algorithm in MIT lecture 11](https://ocw.mit.edu/courses/6-046j-design-and-analysis-of-algorithms-spring-2015/312f4a419009b58f8147b75975db4347_MIT6_046JS15_lec11.pdf).
Starting with paths of at most one edge, each round computes
`D_new[i,j] = min_k(D_old[i,k] + D_old[k,j])` using immutable previous-round
matrices. The path bound doubles until it covers `n-1` edges. Runtime is
O(n³ log n); the reduction uses O(n²) scratch space and the hints occupy
O(n² log n).

The input/output contract follows CLRS Floyd–Warshall:

| Stage | Probes |
| --- | --- |
| Input | `pos`: node scalar; `A`: edge weights; `adj`: edge mask |
| Hint | `Pi_h`: edge predecessor pointer; `D`: edge distance; `msk`: reachability; `m`: graph scalar path bound |
| Output | `Pi`: edge predecessor pointer |

One initialization snapshot is followed by one snapshot per complete square,
including the final state. The number of hints is
`1 + ceil(log2(max(1, n-1)))`. No inner-loop hint steps are emitted.
`Pi[i,j]` is the predecessor of j on a shortest path from i. Unreachable pairs
and diagonals retain the source index i. Existing paths win ties; otherwise
intermediate vertices are considered in ascending order.

Off-diagonal zero means no edge, as in CLRS Floyd–Warshall. The diagonal distance
is zero. Internally unreachable distances are infinite; probes instead use zero
plus `msk=0`, so neural inputs remain finite. Negative edges are supported under
the precondition of no negative cycles; general negative-cycle detection is not
provided. Default sampling reuses Floyd–Warshall's nonnegative, undirected graph
distribution; direct algorithm calls also support directed graphs. Empty,
non-square, nonfinite matrices and negative self-loops are rejected.

## Models and training

```python
feedback = next(iter(loader))
model = clrs.models.BaselineModel(
    spec, feedback, clrs.get_processor_factory('pgn'),
    hidden_dim=32, encode_hints=True, decode_hints=True,
    hint_teacher_forcing=0.5, device='cpu', seed=0)
loss = model.feedback(feedback)
predictions, predicted_hints = model.predict(feedback.features)
metrics = clrs.evaluate(feedback.outputs, predictions)
```

Pass lists of specs and dummy batches for multi-task learning, then select
`algorithm_index` in `feedback`, `predict`, or `forward`. Task encoders and
decoders are separate; the processor is shared. `forward` exposes differentiable
raw logits for custom training loops. For explicit PyTorch optimization, call
`model.loss(feedback)` and use its total and per-probe losses.

All 19 upstream factory options are implemented: Deep Sets, GAT/GATv2 with
masked or full adjacency, both memory networks, MPNN, PGN, masked PGN, and
upstream gated/triplet variants. D2 GATv2 is available as a class. The memory
network retains the original discrete embedding design; feature-to-index
conversion is not differentiable. This port does not claim numerical identity
between framework initializers or reproduce published benchmark scores.

Training supports optional LSTM state, dropout, noisy teacher forcing, hard/soft
hint feedback, processor freezing, gradient clipping, and checkpointed Adam
state. Evaluation consumes only the initial ground-truth hint; subsequent hints
are autoregressive. Trajectory lengths still determine the number of steps,
as in CLRS. PyTorch generators replace JAX random keys; calls no longer accept
an RNG key as their first argument. `save_model()` / `restore_model()` use torch
state dictionaries and include optimizer and generator state. JAX checkpoints
are not compatible.

```bash
python -m clrs.examples.run --algorithms min_plus_apsp --steps 100
python -m clrs.examples.run --algorithms bfs,min_plus_apsp --processor-type triplet_pgn --steps 100
python -m clrs.examples.run --algorithms floyd_warshall --source published --steps 100
```

The example reports validation and larger-graph test metrics and writes a
checkpoint. Use `--device cuda` when available. `--num-samples`, `--eval-samples`,
`--train-length`, and `--test-length` customize generated datasets; published
splits retain their fixed examples. `--restore` resumes weights and optimizer;
data iterator position and live chunk state restart.

For truncated BPTT, `create_chunked_dataset(..., chunk_length=...)` returns a
repeating iterator and spec; `BaselineModelChunked` consumes its feedback.
Inputs, hints, and outputs all have `[time, batch, ...]` axes, with `is_first`
and `is_last` markers. Pending transitions cross chunk boundaries before their
losses are computed, retaining training gradients at those boundaries. Hidden
states are detached between chunks and reset at sample boundaries. Call
`reset_state()` when starting a new stream. The CLI enables this with
`--chunk-length`. Direct `chunkify` drops incomplete final chunks of finite
streams; the dataset wrapper repeats full batches.

CLRS-Text formatters and the Hugging Face generator remain available. The text
creation utility now returns a PyTorch IterableDataset of Python dictionaries.

## Validation

The tests cover all reference algorithms, sampler determinism, published record
and cache decoding, all processors, full/chunked losses, future-hint isolation,
checkpoint restoration, unused multi-task heads, and a tiny APSP overfit run.
Min-plus tests check bounded-path distances and reconstructed predecessor paths,
including ties, disconnected graphs, signed edges, and singleton graphs.

To additionally check real published records after extracting them:

```bash
CLRS_PUBLISHED_TEST_DIR=/path/to/data OMP_NUM_THREADS=1 pytest -q
```

## Citation

To cite the CLRS Algorithmic Reasoning Benchmark:

```latex
@article{deepmind2022clrs,
  title={The CLRS Algorithmic Reasoning Benchmark},
  author={Petar Veli\v{c}kovi\'{c} and Adri\`{a} Puigdom\`{e}nech Badia and
    David Budden and Razvan Pascanu and Andrea Banino and Misha Dashevskiy and
    Raia Hadsell and Charles Blundell},
  journal={arXiv preprint arXiv:2205.15659},
  year={2022}
}
```

To cite the CLRS-Text Algorithmic Reasoning Language Benchmark:

```latex
@article{deepmind2024clrstext,
  title={The CLRS-Text Algorithmic Reasoning Language Benchmark},
  author={Larisa Markeeva and Sean McLeish and Borja Ibarz and Wilfried Bounsi
    and Olga Kozlova and Alex Vitvitskyi and Charles Blundell and
    Tom Goldstein and Avi Schwarzschild and Petar Veli\v{c}kovi\'{c}},
  journal={arXiv preprint arXiv:2406.04229},
  year={2024}
}
```
