# 3_datasets_dataloaders notes

## Dataset vs DataLoader
- **Dataset** = "what is sample i?": `__len__` + `__getitem__(i)` → one `(x, y)`. File reading, decoding, transforms live here. Knows nothing about batches/order/processes.
- **DataLoader** = delivery pipeline: batching (`collate_fn`, default = `stack`), shuffling (`Sampler` picks indices), parallelism (`num_workers`), `pin_memory`.
- Tutorial: FashionMNIST 60k/10k, 28×28 gray; batch → `[64, 1, 28, 28]`, labels `[64]`. `CustomImageDataset` (CSV + `decode_image`) is only defined, never used.

## num_workers (measured, 4 cores, see `3b_dataloader_benchmark.py`)
- Default `num_workers=0` = loading in the main process, GPU waits.
- Slow decode (2 ms/sample): 4.35s → 2.20s → 1.20s for 0/2/4 workers — linear, CPU-bound.
- Even FashionMNIST: 2.63s → 1.08s (per-sample PIL → tensor is Python, slow).
- `time.perf_counter()`: high-resolution clock for measuring durations (only differences are meaningful). `t = time.perf_counter(); work(); dt = time.perf_counter() - t`. Prefer over `time.time()` (lower resolution, can jump when system clock changes). For GPU work, `torch.cuda.synchronize()` before reading the end time.
- Benchmark uses busy-wait, not `sleep` (sleep frees the CPU and would flatter workers); `synchronize()` at the end for honest timing.

## Am I data-bound?
- Test: step time with **one fixed batch already on GPU** (pure compute ceiling) vs **real loader**.
- Measured: fixed 1.3 ms/step; real nw=0 134 ms (107×), nw=4 40 ms (32×) → GPU idle ~99%.
- Real ≈ fixed → compute-bound → model-side optimizations (AMP, `torch.compile`) pay off. Otherwise they buy nothing.
- Other signs: `nvidia-smi` util swinging 0↔100%; CPU cores pegged; profiler gaps between kernels; time `next(it)` vs step separately.
- Fixes (cheap → expensive): more workers / `persistent_workers` / `prefetch_factor` → cheaper `__getitem__` (pre-decode, pre-resize, cache) → faster storage format → GPU decode/augment (DALI, `decode_jpeg(device="cuda")`) → more CPU per GPU.

## Interview topics
- **Workers:** start ≈ CPU cores, measure; each is a process (RAM). `persistent_workers=True` avoids respawn per epoch; `prefetch_factor` default 2.
- **Processes not threads:** GIL. Gotchas: same NumPy seed in every worker (`worker_init_fn`); don't open file handles / DB connections in `__init__` before fork.
- **Map-style vs `IterableDataset`:** random access vs stream; streaming → shard per worker and per GPU yourself.
- **`collate_fn`:** default stack breaks on variable sizes (boxes, lidar points per frame) → custom collate: pad or return lists.
- **GPU-side data work:** batched GPU augmentation, GPU JPEG decode, DALI — matters with many high-res cameras.
- **DDP:** `DistributedSampler`; call `sampler.set_epoch(e)` or every epoch has the same shuffle.
- **Storage:** millions of small JPEGs are slow → WebDataset tar shards, LMDB, Parquet; cache decoded tensors.
- **Interview line:** "Before optimizing the model, I check if I'm data-bound by comparing step time with a fixed GPU batch vs the real loader."
