import torch

import torch.optim as optim


class MyAdam(optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)  # m_t, mean
                    state['exp_avg_sq'] = torch.zeros_like(p.data)  # v_t, variance

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                beta1, beta2 = group['betas']

                state['step'] += 1

                # update moments

                # eq: m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
                exp_avg.mul_(beta1).add_(grad, alpha=(1 - beta1))

                # eq: v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
                exp_avg_sq.mul_(beta2).add_(grad ** 2, alpha=(1 - beta2))

                # corrected versions
                # eq: m̂_t = m_t / (1 - β₁^t)
                exp_avg.div_(1 - (beta1 ** state['step']))
                # eq: v̂_t = v_t / (1 - β₂^t)
                exp_avg_sq.div_(1 - (beta2 ** state['step']))

                # update: θ_t = θ_{t-1} - α * m̂_t / (sqrt(v̂_t) + ε)
                p.data -= group['lr'] * exp_avg / (torch.sqrt(exp_avg_sq) + group['eps'])
