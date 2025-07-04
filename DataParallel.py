import torch.distributed as dist
import torch.nn as nn
import torch
class ManualNaiveDataParallel:
    """
    A manual naive data parallel implementation that uses torch.distributed for gradient synchronization.

    Example usage:
    def main(rank, world_size):

        setup(rank, world_size)

        # Hyperparameters
        input_dim, hidden_dim, output_dim = 20, 10, 5
        batch_size = 16
        epochs = 2

        # Build model
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ).to(rank)

        my_ddp = MyManualDDPOverlap(model)


        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        for epoch in range(epochs):
            print(f"[Rank {rank}] 🚀 Starting epoch {epoch}")
            sampler.set_epoch(epoch)  # Shuffle differently each epoch

            for batch_idx, (bx, by) in enumerate(loader):
                bx, by = bx.to(rank), by.to(rank)

                optimizer.zero_grad()
                preds = model(bx)
                loss = criterion(preds, by)
                loss.backward()  # Hooks are called during backward to start async all-reduce
                my_ddp.synchronize_grads()  # Wait for all async operations and average gradients
                optimizer.step()

    """

    def __init__(self, module):
        self.module = module
        self.worker_handles = []
        self.register_hooks()

    def register_hooks(self):
        """
        Register backward hooks for all params with gradients

        """

        def all_reduce_hook(grad):
            if grad is not None:
                work = dist.all_reduce(grad, op=dist.ReduceOp.SUM, async_op=True)
                self.worker_handles.append(work)
            return grad  # return the grad unchanged for autograd

        for p in self.module.parameters():
            if p.requires_grad:
                p.register_hook(all_reduce_hook)

    def synchronize_grads(self):
        """Wait for all workers to finish their all-reduce operations."""
        for work in self.worker_handles:
            work.wait()
        self.worker_handles.clear()

        world_size = dist.get_world_size()

        for p in self.module.parameters():
            if p.grad is not None:
                p.grad /= world_size

class DataParallelBucket(nn.Module):
    def __init__(self, module, bucket_cap_size_mb=25, grad_type=torch.float32):
        super().__init__()
        self.module = module
        self.require_backward_grad_sync = True # whether to sync grads during backward pass or not, useful during accumulating gradients
        each_grad_size = 2 if grad_type in [torch.float16, torch.bfloat16] else 4  # size in bytes for each grad type
        self.bucket_cap_size_mb = bucket_cap_size_mb * 1024 * 1024 // each_grad_size # calculate number of grads in one bucket
        self.register_backward_hook()
        self._post_backward_hook_callback_set = False # whether the callback for wait gradient synchronization is set
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def backward(self, input_tensor, output_tensor, output_tensor_grad):
        return self.module.backward(input_tensor, output_tensor, output_tensor_grad)

    def register_backward_hook(self):


