import torch
import timeit
from torch import nn
from contextlib import nullcontext

def mixed_precision_bench():
    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float32)
    print(s)
    s = torch.tensor(0,dtype=torch.float16)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float16)
    print(s)
    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float16)
    print(s)
    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        x = torch.tensor(0.01,dtype=torch.float16)
        s += x.type(torch.float32)
    print(s)

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden: int = 8192):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden, bias=False)
        self.ln = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, out_features, bias=False)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x

def check_precision():
    device = torch.device('cuda:0')
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        model = ToyModel(100, 10, hidden=1024).to(device)
        x = torch.randn(32, 100, device=device)
        y = model(x)
        print(f"params precision: {next(model.parameters()).dtype}")
        print(f"fc1 precision: {model.fc1(x).dtype}")
        print(f"layernorm precision: {model.ln(y).dtype}")
        print(f"output precision: {y.dtype}")
        print(f"loss precision: {y.mean().dtype}")
        y.mean().backward()
        print(f"grad precision: {next(model.parameters()).grad.dtype}")

def _bench(batch_size, precision: torch.dtype) -> list[float]:
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    model = ToyModel(2048, 2048, hidden=8192).to(device)
    x = torch.randn(batch_size, 2048, device=device)
    ctx = torch.autocast(device_type="cuda", dtype=precision) if precision != torch.float32 else nullcontext()
    with ctx:
        # warmup
        for _ in range(5):
            model.zero_grad(set_to_none=True)
            y = model(x)
            y.mean().backward()
        torch.cuda.synchronize()
        times = []
        for _ in range(100):
            torch.cuda.synchronize()
            start = timeit.default_timer()
            model.zero_grad(set_to_none=True)
            y = model(x)
            y.mean().backward()
            torch.cuda.synchronize()
            end = timeit.default_timer()
            times.append(end - start)
    return times

def _calc_mean_std(times):
    mean = sum(times) / len(times)
    std = (sum((x - mean) ** 2 for x in times) / len(times)) ** 0.5
    return mean, std

if __name__ == "__main__":
    # batch_size = 128
    for batch_size in [1, 8, 16, 32, 64, 128, 256]:
        print(f"Running mixed precision benchmark with batch size {batch_size}...")

        # benchmark full fp32
        times = _bench(batch_size, torch.float32)
        mean, std = _calc_mean_std(times)
        print(f"full fp32: {mean*1e3:.2f} ± {std*1e3:.2f} ms")

        # benchmark full fp16
        times = _bench(batch_size, torch.float16)
        mean, std = _calc_mean_std(times)
        print(f"full fp16: {mean*1e3:.2f} ± {std*1e3:.2f} ms")

        # benchmark bf16
        times = _bench(batch_size, torch.bfloat16)
        mean, std = _calc_mean_std(times)
        print(f"full bf16: {mean*1e3:.2f} ± {std*1e3:.2f} ms")
