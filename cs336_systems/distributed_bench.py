import time
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def all_reduce(rank, world_size, elements: torch.Tensor, dtype: torch.dtype, timing):
    n_warm = 3
    n_test = 5
    setup(rank, world_size)
    data = torch.rand(elements).to(dtype)
    # print(f"rank {rank} data (before all-reduce): {data[:5]}")
    for _ in range(n_warm): # warmup
        dist.all_reduce(data, async_op=False)

    dist.barrier()
    start = time.perf_counter()
    for _ in range(n_test):
        dist.all_reduce(data, async_op=False)
    # torch.cuda.synchronize()
    end = time.perf_counter()
    timing[rank] = (end - start) / n_test
    dist.destroy_process_group()
    # print(f"rank {rank} data (after all-reduce): {data[:5]}")

if __name__ == "__main__":
    dtype = torch.float32
    tensor_sizes = torch.tensor([1, 10, 100, 1000], dtype=int) * 10e3
    world_size = [2, 4, 6]

    for mbs in tensor_sizes:
        tensor_elements = (mbs // dtype.itemsize).to(int)
        for n_gpus in world_size:
            timing = torch.zeros((n_gpus,), dtype=torch.float32)
            timing.share_memory_()
            mp.spawn(fn=all_reduce, args=(n_gpus, tensor_elements.item(), dtype, timing), nprocs=n_gpus, join=True)

            print(f"Megabytes: {mbs} | n_proc: {n_gpus} | time: {timing.max()}")