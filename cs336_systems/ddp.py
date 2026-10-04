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
        slow = True
        if slow:
            for param in self.module.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, async_op=False, op=dist.ReduceOp.AVG)
        else:
            grads = torch._utils._flatten_dense_tensors([param.grad for param in self.module.parameters() if param.grad is not None])
            dist.all_reduce(grads, async_op=False, op=dist.ReduceOp.AVG)
            unflattened_grads = torch._utils._unflatten_dense_tensors(grads, [param.grad for param in self.module.parameters() if param.grad is not None])
            for param, grad in zip([param for param in self.module.parameters() if param.grad is not None], unflattened_grads):
                param.grad.copy_(grad)
            

class DDPOverlap(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, n_devices: int = 4):
        super(DDPOverlap, self).__init__()
        self.module = module
        self.n_device = n_devices
        self.handles = []

        def _hook(self, param):
            # Async all-reduce the gradient as soon as it is created
            # store the handle for later synchronization
            handle = dist.all_reduce(param.grad, async_op=True, op=dist.ReduceOp.AVG)
            self.handles.append(handle)

        for param in module.parameters():
            dist.broadcast(param.data, src=0)
            # register a hook to all_reduce the gradient as they are created
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(lambda p: _hook(self, p))

    
    def forward(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)
    
    def backward(self):
        self.module.backward()

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

