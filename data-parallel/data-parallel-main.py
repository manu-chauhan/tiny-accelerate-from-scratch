import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

import bucket


def setup(rank, world_size):
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size, init_method="tcp://127.0.0.1:12355")
    torch.cuda.set_device(f"cuda: {rank}")


def cleanup():
    dist.destroy_process_group()


def get_deep_model(input_dim, hidden_dim, output_dim, depth, device):
    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(depth):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers).to(device)


def train(rank, world_size):
    if rank == 0:
        global_start_time = time.time()
    setup(rank, world_size)
    input_dim, hidden_dim, output_dim = 512 + 128 + 64 + 128, (1024 * 2) + 64 + 8 + 2, 256
    depth = 81
    batch_size = (1024 * 1024 * 1024 * 2) + (1024 * 2000 * 100)
    epochs = 210

    device = torch.device(f"cuda:{rank}")

    model = get_deep_model(input_dim, hidden_dim, output_dim, depth, device).to(device)

    dp_model = bucket.DataParallelBucket(model, bucket_cap_size_mb=2, grad_type=torch.float32)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    criterion = nn.MSELoss()

    x = torch.randn(1000, input_dim)
    y = torch.randn(1000, output_dim)

    dataset = TensorDataset(x, y)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        start = time.time()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)  # Use model, not ddp_model
            loss = criterion(out, by)
            loss.backward()  # Triggers async all-reduce via hooks
            dp_model.synchronize()  # Wait and average gradients
            optimizer.step()
        end = time.time()
        print(f"[Rank {rank}] Epoch {epoch} complete. Time taken: {end - start:.2f}s")

    cleanup(rank)

    # Log total time from rank 0 only
    if rank == 0:
        total_time = time.time() - global_start_time
        print(f"\n✅ Total time taken (all ranks): {total_time:.2f} seconds\n")


# Main execution
if __name__ == "__main__":
    world_size = 1
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
