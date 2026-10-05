import torch
import torch.distributed as dist
from torch.optim import Optimizer
from typing import Type, Any


class OptimizerStateSharding(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs
        self.counter = 0
        # All parameters in the model
        self.all_params = []
        # Mapping from parameter memory id to the owner rank
        self.param_mem_to_owner = {}
        # Wrapped optimizer holding only this rank's shard (created lazily)
        self.optimizer = None

        # Calls self.add_param_group for each group in params
        super().__init__(params, defaults=kwargs)

    def step(self, closure=None, **kwargs):
        loss = None
        if self.optimizer is not None:
            loss = self.optimizer.step(closure, **kwargs)
        elif closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Each rank broadcasts the parameters it updated
        for p in self.all_params:
            dist.broadcast(p.data, src=self.param_mem_to_owner[id(p)])
        return loss

    def add_param_group(self, param_group: dict[str, Any]):
        # Register the full group so the wrapper's param_groups stay complete
        super().add_param_group(param_group)
        group = self.param_groups[-1]

        # Round robin parameter assignment across optimizer shards
        owned = []
        for p in group['params']:
            if id(p) in self.param_mem_to_owner:
                continue
            owner = self.counter % self.world_size
            self.param_mem_to_owner[id(p)] = owner
            self.all_params.append(p)
            if owner == self.rank:
                owned.append(p)
            self.counter += 1

        if not owned:
            return
        shard_group = {k: v for k, v in group.items() if k != 'params'}
        shard_group['params'] = owned
        if self.optimizer is None:
            self.optimizer = self.optimizer_cls([shard_group], **self.optimizer_kwargs)
        else:
            self.optimizer.add_param_group(shard_group)
