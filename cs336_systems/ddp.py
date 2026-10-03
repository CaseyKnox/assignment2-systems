import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

class DDP(torch.nn.Module):

    def __init__(self, module: torch.nn.Module, n_devices: int = 4):
        super(DDP, self).__init__()
        self.module = module
        self.n_devices = n_devices

        for param in module.parameters():
            dist.broadcast(param.data, src=0)

    @staticmethod
    def setup(rank: int, world_size: int):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def forward(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)

    def backward(self):
        self.module.backward()

    def finish_gradient_synchronization(self):
        for param in self.module.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, async_op=False, op=dist.ReduceOp.AVG)
