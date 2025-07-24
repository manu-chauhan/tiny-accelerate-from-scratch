import torch

import torch.optim as optim
import torch.distributed as dist


class ZeRO2Optimizer(optim.Optimizer):
    def __init__(self, params,
                 optimizer_class=optim.Adam,
                 lr=0.001,
                 **kwargs):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.optimizer = optimizer_class(params, lr=lr, **kwargs)
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.partition_params()

    def partition_params(self):
        self.param_to_rank_map = {}

        for param_group in self.optimizer.param_groups:
            for param_idx, param in enumerate(param_group['params']):
                correct_rank = param_idx % self.world_size
                self.param_to_rank_map[param] = correct_rank

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group_idx, group in enumerate(self.optimizer.param_groups):
            for param_idx, param in enumerate(group['params']):
                if param.grad is None:
                    continue

                # # all reduce here
                # dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG)

                dist.reduce_scatter(out=param.grad.data,)

                # now update only THIS rank's optim states
                if self.param_to_rank_map[param] == self.rank:
                    state = self.optimizer.state[param]
                    grad = param.grad

                    # for adam update moment and variance
                    if 'exp_avg' in state and 'exp_avg_sq' in state:
                        beta1, beta2 = group['betas'][0], group['betas'][1]

                        state['exp_avg'].mul_(beta1).add_(grad, alpha=(1 - beta1))
                        state['exp_avg'] /= (1 - beta1 ** (state.get('step', 0) + 1))

                        state['exp_avg_sq'].mul_(beta2).add_(grad ** 2, alpha=(1 - beta2))
                        state['exp_avg_sq'] /= (1 - beta2 ** (state.get('step', 0) + 1))

                # now all ranks must broadcast current param from OWNER to all other ranks
                for k, v in self.optimizer.state[param].items():
                    if torch.is_tensor(v):
                        dist.broadcast(v, src=self.param_to_rank_map[param])

                # actual update here
                with torch.no_grad():

                    lr = group['lr']

                    step = state['exp_avg'] / (torch.sqrt(state['exp_avg_sq'] + group['eps']))

                    param.data.add_(step, alpha=-lr)

        return loss
