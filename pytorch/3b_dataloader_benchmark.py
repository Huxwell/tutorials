# Extra for 3_datasets_dataloaders.py:
#   Part 1: how num_workers affects DataLoader throughput
#   Part 2: "am I data-bound?" check - real loader vs one fixed batch already on GPU
import time
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import v2


class SlowDS(Dataset):
    """Fake dataset where each sample costs ~2 ms of CPU (like JPEG decode + augment)."""
    def __len__(self):
        return 2048

    def __getitem__(self, i):
        t = time.perf_counter()
        while time.perf_counter() - t < 0.002:  # busy-wait = burn CPU, like real decoding
            pass
        return torch.randn(3, 64, 64), i % 10


fashion = datasets.FashionMNIST(
    root="data", train=True, download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)


def bench_loader(ds, n_batches, **loader_kwargs):
    """Time how long it takes to deliver n_batches to the GPU."""
    dl = DataLoader(ds, batch_size=64, **loader_kwargs)
    t = time.perf_counter()
    for k, (x, y) in enumerate(dl):
        x = x.to("cuda", non_blocking=True)
        if k + 1 == n_batches:
            break
    torch.cuda.synchronize()  # wait until the last copy really finished
    return time.perf_counter() - t


print("=== Part 1: num_workers ===")
for name, ds, n in [("slow decode", SlowDS(), 32), ("FashionMNIST", fashion, 300)]:
    for nw in [0, 2, 4]:
        sec = bench_loader(ds, n, num_workers=nw, pin_memory=True)
        print(f"{name:12s} num_workers={nw}: {sec:.2f}s")


print("=== Part 2: data-bound check ===")
model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 64 * 64, 512), nn.ReLU(), nn.Linear(512, 10)).cuda()
opt = torch.optim.SGD(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
N_STEPS = 30


def train_step(x, y):
    loss = loss_fn(model(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()


def time_real_loader(num_workers):
    dl = DataLoader(SlowDS(), batch_size=64, num_workers=num_workers, pin_memory=True)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for k, (x, y) in enumerate(dl):
        train_step(x.cuda(non_blocking=True), y.cuda(non_blocking=True))
        if k + 1 == N_STEPS:
            break
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / N_STEPS


def time_fixed_batch():
    x = torch.randn(64, 3, 64, 64, device="cuda")  # one batch, reused every step
    y = torch.randint(0, 10, (64,), device="cuda")
    for _ in range(5):  # warmup
        train_step(x, y)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(N_STEPS):
        train_step(x, y)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / N_STEPS


gpu_only = time_fixed_batch()
print(f"fixed batch on GPU (pure compute): {gpu_only*1000:.1f} ms/step")
for nw in [0, 4]:
    real = time_real_loader(nw)
    print(f"real loader, num_workers={nw}:     {real*1000:.1f} ms/step  -> {real/gpu_only:.0f}x slower than compute")
