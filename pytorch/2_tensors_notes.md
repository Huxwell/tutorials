# 2_tensors notes

- **Strides:** memory is a flat line; a tensor is shape + one stride per dim. `.T` / `permute` just swap strides — no data moves.
- **Contiguous:** reading the tensor row by row visits memory in order. After `.T`, it doesn't.
- **`view`:** only computes new strides, never copies. Splitting dims always works; merging only if the dims are back-to-back in memory.
- **`reshape`:** view when possible, copy otherwise.
- **`contiguous()`:** copies into normal order if needed, otherwise no-op.
- **Not in-place:** `view`/`reshape`/`contiguous` return new tensors → `x = x.reshape(...)`. Only `_` methods modify in place.
- **`reshape(6)`** = flatten. `reshape(-1)` = flatten, infer size.
- **`expand` vs `repeat`:** `expand` = stride 0, zero memory; `repeat` copies. Never write in place into an expanded tensor.
- **`cat` vs `stack`:** `cat` joins along an existing dim, `stack` adds a new one.
- **Async GPU:** calls only queue work. `.item()`, `.cpu()`, `print` force a sync → stall the pipeline.
- **Timing:** without `torch.cuda.synchronize()` you measure only launch time (1 ms vs 196 ms real).
- **Pinned memory + `non_blocking=True`:** revisit in the future.
