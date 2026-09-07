import time
import torch
import timeit
from cs336_basics.model import CausalMultiHeadSelfAttention

# analytical calculations of expected memory usage for activations, gradients, and optimizer states
def _calc_memory(batch, seq_len, d_model, bit_size=32):
    linear_mem = batch * seq_len * d_model
    scores_mem = batch * seq_len**2 
    # 4 because qkv + output projection, and scores for attention
    return (4 * linear_mem + scores_mem) * (bit_size / 8) # bytes

def _calc_bwd_memory(batch, seq_len, d_model, bit_size=32):
    return _calc_memory(batch, seq_len, d_model, bit_size) * 2  # activations + gradients

def _bench(d_model, seq_len, n_heads=1, batch=8, warmup=5, test_itr=100, compiled=False) -> list[float]:
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    model = CausalMultiHeadSelfAttention(d_model, n_heads).to(device)
    x = torch.randn(batch, seq_len, d_model, device=device)
    if compiled:
        model = torch.compile(model)

    # warmup
    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        y = model(x)
        y.mean().backward()
    torch.cuda.synchronize()

    # Test iterations
    times = []
    fwd_memories = []
    bwd_memories = []
    for _ in range(test_itr):
        torch.cuda.synchronize()
        start = timeit.default_timer()
        model.zero_grad(set_to_none=True)

        torch.cuda.reset_peak_memory_stats(device)
        y = model(x)
        torch.cuda.synchronize()
        fwd_memories.append(torch.cuda.max_memory_allocated(device))

        torch.cuda.reset_peak_memory_stats(device)
        y.mean().backward()
        torch.cuda.synchronize()
        end = timeit.default_timer()
        times.append(end - start)
        bwd_memories.append(torch.cuda.max_memory_allocated(device))
    return times, fwd_memories, bwd_memories

if __name__ == "__main__":
    for d_model in [16, 32, 64, 128]:
        for seq_len in [4096, 8192]: #, 16384]:
            for compiled in [False, True]:
                mem_bytes = _calc_memory(batch=8, seq_len=seq_len, d_model=d_model)
                back_mem_bytes = _calc_bwd_memory(batch=8, seq_len=seq_len, d_model=d_model)
                print(
                    f"Expected memory usage for d_model={d_model}, "
                    f"seq_len={seq_len}: "
                    f"Forward {mem_bytes/1024**2:.2f} MB "
                    f"| Backward {back_mem_bytes/1024**2:.2f} MB"
                )
                try:
                    times, fwd_memories, bwd_memories = _bench(d_model, seq_len, compiled=compiled)
                except RuntimeError as e:
                    print(f"RuntimeError for d_model={d_model}, seq_len={seq_len}: {e}")
                    continue
                mean = sum(times) / len(times)
                std = (sum((x - mean) ** 2 for x in times) / len(times)) ** 0.5
                mean_ms = mean * 1e3
                std_ms = std * 1e3
                fwd_mem_mb = sum(fwd_memories) / len(fwd_memories) / 1024**2
                bwd_mem_mb = sum(bwd_memories) / len(bwd_memories) / 1024**2
                print(
                    f"Compiled: {compiled:<5} | "
                    f"d_model: {d_model:<5} | "
                    f"seq_len: {seq_len:<5} | "
                    f"time: {mean_ms:>6.2f} ± {std_ms:>5.2f} ms | "
                    f"fwd_mem: {fwd_mem_mb:>7.2f} MB | "
                    f"bwd_mem: {bwd_mem_mb:>7.2f} MB"
                )
                time.sleep(1)  # Sleep for a second to re-alloc gpu
