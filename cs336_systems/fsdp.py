import os
import torch
import torch.distributed as dist
from collections import defaultdict
# from torch.nn.modules import Linear, Embedding
from cs336_basics.model import Embedding, Linear

class FSDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        """
        Given an instantiated PyTorch nn.Module to be parallelized, construct an FSDP module that
        will handle weight all-gathers and gradient reduce-scatters. Make sure that your hooks or your
        module wrappers all-gather the weights in time for the forward pass. To limit memory use,
        only start gathering after the layer two before the current one has completed its forward pass.
        In the backward pass, your hooks or module wrappers should all-gather to have the weights
        available for the computation. When the gradients are available, they should be reduce-
        scattered to the appropriate ranks. Make sure to free the gathered weights after use. When
        compute_dtype is provided, cast the weights to that dtype before communicating or using them
        for compute, while keeping master weights and optimizer updates in FP32.
        """
        super(FSDP, self).__init__()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.compute_dtype = compute_dtype
        self.module = module
        self.handles = []

        for mod in self.module.modules():
            for param in mod.parameters():
                # start with identical weights before sharding
                dist.broadcast(param.data, src=0)

            # Norms are owned by everyone since they're small
            if not isinstance(mod, (Linear, Embedding)):
                def _hook(param):
                    handle = dist.all_reduce(param.grad, async_op=True, op=dist.ReduceOp.SUM)
                    self.handles.append((handle, param))

                for param in mod.parameters(recurse=False):
                    if param.requires_grad:
                        param.register_post_accumulate_grad_hook(_hook)

                continue

            # split the params into world_size chunks
            chunk_size = len(mod.weight.data) // self.world_size
            start_ptr = chunk_size * self.rank

            # Replace the module's full data weight with a chunk
            chunk = mod.weight.data[start_ptr : start_ptr + chunk_size].clone()
            mod.weight.data = chunk

            # FORWARD
            def make_fwd_prehook(dt):
                # TODO: add dtype
                def hook(mod, inp):
                    # gather the entire layer's weights from all ranks
                    mod._saved_fp32 = mod.weight.data
                    shape = list(mod.weight.data.shape)
                    shape[0] = shape[0] * self.world_size
                    full_tensor = torch.empty(shape, device=mod.weight.data.device)
                    dist.all_gather_into_tensor(full_tensor, mod.weight.data)
                    mod.weight.data = full_tensor
                    mod.weight.data = mod.weight.data.to(dt)

                return hook

            def fwd_hook(mod, inp, out):
                # reset layer's weights to chunk
                mod.weight.data = mod._saved_fp32

            mod.register_forward_pre_hook(make_fwd_prehook(compute_dtype))
            mod.register_forward_hook(fwd_hook)

            # BACKWARD
            def make_bwd_prehook(dt):
                def backward_prehook(mod, grad_output):
                    mod._saved_fp32_bwd = mod.weight.data

                    # Gather the data from all ranks
                    shape = list(mod.weight.data.shape)
                    shape[0] = shape[0] * self.world_size
                    full_tensor = torch.empty(shape, device=mod.weight.data.device)
                    dist.all_gather_into_tensor(full_tensor, mod.weight.data)

                    mod.weight.data = full_tensor.to(dt)
                    mod.weight.grad = None

                return backward_prehook

            def make_grad_hook(m, is_linear):
                def hook(param):
                    # restore tensor
                    if is_linear and hasattr(m, "_saved_fp32_bwd"):
                        m.weight.data = m._saved_fp32_bwd
                        del m._saved_fp32_bwd

                    if param.grad is not None:
                        # param.grad = param.grad.to(torch.float32)
                        shape = list(param.grad.shape)
                        shape[0] = shape[0] // self.world_size
                        out = torch.empty(shape, device=param.grad.device)
                        # average not supported by GLOO, use sum / world size
                        handle = dist.reduce_scatter_tensor(out, param.grad.to(torch.float32), async_op=True, op=dist.ReduceOp.SUM)
                        self.handles.append((handle, param))
                        param.grad = out

                return hook

            if isinstance(mod, Linear):
                mod.register_full_backward_pre_hook(make_bwd_prehook(compute_dtype))

            mod.weight.register_post_accumulate_grad_hook(make_grad_hook(mod, isinstance(mod, Linear)))

        
    @staticmethod
    def setup(rank: int, world_size: int):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def forward(self, *inputs, **kwargs): 
        """
        Calls the wrapped module’s forward() method with the
        """
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        """
        When called, wait for asynchronous communication calls to finish on the GPU.
        """
        for handle, param in self.handles:
            handle.wait()
            param.grad /= self.world_size
        self.handles.clear()

    def all_gather(self):
        d = dict()
        for name, mod in self.module.named_modules():
            if not hasattr(mod, "weight"): 
                continue
            if isinstance(mod, (Linear, Embedding)):
                shape = list(mod.weight.data.shape)
                shape[0] = shape[0] * self.world_size
                full_tensor = torch.empty(shape, device=mod.weight.data.device)
                dist.all_gather_into_tensor(full_tensor, mod.weight.data)
                d[f"{name}.weight"] = full_tensor
            else:
                d[f"{name}.weight"] = mod.weight.data

        return d