import torch
import torch.nn as nn
import torch.distributed as dist


class Bucket:
    def __init__(self, params, grad_type, process_group):
        self.params = params
        self.grad_type = grad_type
        self.process_group = process_group
        self.main_grads = [p.main_grad for p in params]
        self.handle = None
        self.ready_params = set()

    def mark_ready(self, param):
        self.ready_params.add(param)
        if len(self.ready_params) == len(self.params):
            self.sync()
            self.ready_params.clear()

    def sync(self):
        if self.handle is not None:
            self.handle.wait()

        flat_grads = [g.view(-1) for g in self.main_grads if g is not None]

        if flat_grads:
            bucket_tensor = torch.cat(flat_grads).to(self.grad_type)
            self.handle = dist.all_reduce(bucket_tensor, group=self.process_group, async_op=True, op=dist.ReduceOp.SUM)
            bucket_tensor /= dist.get_world_size(self.process_group)

            # now copy synced grads back
            offset = 0
            for g in self.main_grads:
                if g is not None:
                    numel = g.numel()
                    g.copy_(bucket_tensor[offset:offset + numel].view_as(g))
                    offset += numel


class BucketManager:
    def __init__(self, params, process_group, bucket_size_elements, grad_type):
        self.buckets = []
        self.process_group = process_group
        self.grad_type = grad_type
        current_bucket_params = []
        current_size = 0

        for p in params:
            if p.requires_grad:
                p.main_grad = torch.zeros_like(p, dtype=grad_type)
                numel = p.numel()
                if current_size + numel > bucket_size_elements:
                    self.buckets.append(
                        Bucket(params=current_bucket_params, grad_type=grad_type, process_group=process_group))
                    current_bucket_params = [p]
                    current_size = numel
                else:
                    current_size += numel
                    current_bucket_params.append(p)

        # just a check for any more params
        if current_bucket_params:
            self.buckets.append(Bucket(current_bucket_params, grad_type=grad_type, process_group=process_group))

    def mark_param_as_ready(self, param):
        for bucket in self.buckets:
            if param in bucket.params:
                # bucket.sync()
                bucket.mark_ready(param)
                break

    def wait(self):
        for bucket in self.buckets:
            if bucket.handle is not None:
                bucket.handle.wait()


class DataParallelBucket(nn.Module):
    def __init__(self, module, bucket_cap_size_mb=25, grad_type=torch.float32):
        super().__init__()
        self.module = module

        self.require_backward_grad_sync = True

        grad_element_size = torch.tensor([], dtype=grad_type).element_size()

        bucket_size_elements = bucket_cap_size_mb * 1024 * 1024 // grad_element_size

        self.bucket_manager = BucketManager(list(module.parameters()), dist.group.WORLD, bucket_size_elements,
                                            grad_type)

        self._register_accumulate_grad_hooks()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def _register_accumulate_grad_hooks(self):
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_hook(self._make_param_hook(param))

    def _make_param_hook(self, param):
        def param_hook(grad):
            if param.requires_grad and grad is not None:
                param.main_grad.add_(grad.to(param.main_grad.dtype))

                if self.require_backward_grad_sync:
                    self.bucket_manager.mark_param_as_ready(param=param)
            return grad

        return param_hook

    def synchronize(self):
        self.bucket_manager.wait()

        for p in self.module.parameters():
            if p.requires_grad:
                p.grad = p.main_grad.to(p.dtype)
