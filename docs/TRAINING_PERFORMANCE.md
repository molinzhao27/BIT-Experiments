# Training performance without changing the experiment

This work preserves model weights/architecture, floating-point precision, data splits and batch order,
the loss objective, optimizer hyperparameters, training budget, selection metric and decoding protocol.
Quantization, TF32 changes, different batch sizes and shortened training are excluded. Normal floating-point
rounding is allowed for the explicitly experimental selective AR head.

**Rollout:** the previous runtime is still the default. The optimization flags below are ready for validation,
but have not been installed in active GPU code copies or enabled in queued experiments. Both available CUDA
GPUs were occupied during development. CPU measurements below are component benchmarks, not estimates of
production GPU throughput. Do not restart an ongoing scientific run to apply these changes.

## Runtime overhead reductions

Add `--runtime optimized` to an otherwise identical `scripts/train_stage2.py` command:

- Reuse up to 32 invariant decoder attention masks. Cache keys include length, block size, batch size, head
  count and concrete device; B/H dimensions and attention kernels remain the same. Device/dtype moves clear
  the cache. Masks stay out of checkpoints, and inference-created cached tensors remain safe for backward.
- Skip the unused vocabulary projection during diffusion prefix prefill when a partial prefix block remains.
  The decoder and KV-cache update still run. Block-aligned prefixes retain the original head computation.
- Allocate common padded batches once. Preserve BIT's dataset, sampler, loader RNG, workers, dtypes, values,
  ordering and ownership. Unusual padding and dtype cases fall back to BIT's collator.
- Read detached training loss/count scalars in chunks of 64 instead of synchronizing twice every batch.
  Convert individual values to Python and sum in the original order. Optimizer, scheduler, accumulation,
  evaluation and checkpoint calls retain BIT's ordering, including its existing accumulation behavior.

`--runtime reference` restores the previous workloads. It is also explicit in the existing scripts that
intercept `Trainer.train` for inspection or fast-step validation. Generation outside the optimized runtime
keeps the original prefill workload by default.

PyTorch recommends [reusing invariant BlockMasks](https://pytorch.org/blog/flexattention/) because construction
can be expensive. We preserve their original dimensions rather than changing the compiled attention layout.
Worker/prefetch settings stay unchanged: [DataLoader worker creation consumes RNG state](https://docs.pytorch.org/docs/2.7/data.html#randomness-in-multi-process-data-loading),
so merely adding persistent workers would not preserve the current random stream.

## Selective AR vocabulary head (experimental)

For ordinary Qwen2/Aero autoregressive models, add both `--runtime optimized --selective-ar-head` after the
GPU check below passes. The transformer still processes the full causal sequence with the same dropout.
Only hidden states whose next-token target is supervised go through the frozen vocabulary head. This avoids
allocating `[batch, sequence, vocabulary]` logits for ignored neural-prefix and padding positions.

The summed cross-entropy objective and contrastive term are unchanged; smaller GEMMs and CE reductions can
change rounding. This is not quantization or frozen-encoder caching. Unsupported architectures, quantized
models, stochastic/trainable heads, audio projection and incompatible losses are rejected.

Teacher-forced outputs become compact `preds[N, vocab]` and `targets[N]`. Stage 2's training/evaluation uses
the loss and count; generation is unchanged. Consumers needing full sequence logits must retain the original
forward. The candidate is independent of the packed diffusion fast step.

## Validation and measurement

Use BIT's pinned training environment (torch 2.7.1, transformers 4.48.2, PEFT 0.11.1), not the newer project
analysis environment. Initialize the `external/bit` submodule first.

```bash
python -m pytest tests/test_exact_masks.py tests/test_exact_data.py tests/test_decode_prefill.py \
  tests/test_training_runtime.py tests/test_selective_ar.py tests/test_profile_training_runtime.py -q
python scripts/benchmark_training_runtime.py --output results/profile/runtime_cpu.json
python scripts/benchmark_selective_ar.py --output results/profile/selective_ar_cpu.json
```

The runtime tests compare real multi-step LoRA/dropout gradients, optimizer state and RNG; training-loop tests
compare against the actual vendored BIT loop over multiple shuffled epochs, accumulation settings and checkpoint
choices. Decode checks cover greedy, nucleus, editing, remasking, traces and scores. Collation checks include
non-native byteorder, structured padding, NaN payloads, signed zeros and three-epoch loader RNG/order.

On an **idle GPU under the existing GPU-lock wrapper**, run the bounded production gate:

```bash
python scripts/profile_training_runtime.py --llm aero-1-audio-diffusion \
  --init checkpoints/ssl_human/BEST_core --steps 8 --rounds 3
python scripts/profile_training_runtime.py --llm default --selective-ar-head \
  --init checkpoints/ssl_human/BEST_core --steps 8 --rounds 3
```

`--llm default` uses BIT's original Aero AR loader; registry names select the other supported decoders.
The gate alternates modes with identical real batches, parameters, optimizer/scheduler state and all RNGs.
Every replay shape is warmed before timing. It compares losses, supervised counts, gradients, parameter updates,
optimizer state and RNG, records software/GPU/module hashes, and saves no checkpoints. Tolerances are recorded;
do not loosen them automatically to make a failure pass. Training replay timing excludes data fetch and transfer,
which are reported separately; it is not a complete epoch benchmark. Require both a passing gate and a meaningful
speed improvement before rollout. Then check a short training run on validation data only, with the same
configuration and evaluation cadence. Test data must not select runtime settings.

## Packed diffusion and follow-up work

The prior packed `fast_step` remains independently gated by `scripts/fast_step_gate.py`. Its training failure
was not explained by the eval-mode check, and the static-compilation fix still needs target-GPU evidence.
The user's relaxed exactness requirement permits different dropout draws in principle, but does not establish
that this implementation trains correctly. No fast-step registry entry or queued launcher was changed here.

Further candidates needing isolated GPU experiments: fused AdamW, pinned nonblocking transfer/prefetch,
encoder compilation, and newer attention backends. Preserve the current environment and CPU-affinity isolation
while comparing them. Do not infer CUDA results from these CPU measurements.
