import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from accelerate import SimpleAccelerator

# Create a dummy dataset
data = torch.randn(100, 10)
labels = torch.randn(100, 10)
dataset = TensorDataset(data, labels)
data_loader = DataLoader(dataset, batch_size=16)

# Define a simple model and optimizer
model = nn.Linear(10, 10)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Initialize our accelerator
accelerator = SimpleAccelerator(mixed_precision="fp16")

# Prepare everything
model, optimizer, data_loader = accelerator.prepare(model=model,
                                                    optimizer=optimizer,
                                                    dataloader=data_loader)

# Training loop with gradient accumulation
for epoch in range(5):
    with accelerator.accumulate(model, steps=3) as accum:
        for batch in data_loader:
            inputs, targets = batch
            with accelerator.autocast():
                outputs = model(inputs)
                loss = nn.MSELoss()(outputs, targets)

            # Backward pass with accumulation
            if accum.backward(loss):
                accelerator.step(optimizer)
                optimizer.zero_grad()

    print(f"Epoch {epoch + 1} completed")

# Save a checkpoint
accelerator.save_checkpoint(model, optimizer, "model_checkpoint.pt")
