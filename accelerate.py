import contextlib
import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader


class SimpleAccelerator:
    def __init__(self,
                 mixed_precision=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.is_distributed = torch.cuda.device_count() > 1

        self.mixed_precision = mixed_precision
        self.scaler = torch.amp.GradScaler() if mixed_precision == 'fp16' else None

        if self.is_distributed:
            torch.distributed.init_process_group(backend='nccl')
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.local_rank = 0

    def prepare(self, *, model, optimizer, dataloader):
        model = model.to(self.device)

        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank])
            dataloader = self._wrap_dataloader(dataloader)

        return model, optimizer, dataloader

    def _wrap_dataloader(self, dataloader):
        if self.is_distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataloader.dataset,
                                                                      num_replicas=torch.distributed.get_world_size(),
                                                                      rank=self.local_rank,
                                                                      shuffle=True)

            return DataLoader(dataloader.dataset,
                              batch_size=dataloader.batch_size,
                              num_workers=dataloader.num_workers,
                              pin_memory=True,
                              sampler=sampler)

        return dataloader

    def autocast(self):
        if self.mixed_precision == 'fp16':
            return torch.amp.autocast(device_type=self.device.type, dtype=torch.float16)
        elif self.mixed_precision == 'bf16':
            return torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16)
        else:
            return contextlib.nullcontext()

    def step(self, optimizer):
        if self.scaler:
            self.scaler.step(optimizer)
            self.scaler.update()

        else:
            optimizer.step()

    def clip_gradients(self, model, optimizer, max_norm=1.0):
        if self.scaler:
            # Unscales the gradients of optimizer's assigned params in-place
            self.scaler.unscale_(optimizer)

        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    def backward(self, loss):
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def accumulate(self, model, steps=2):
        class AccumulateContext:
            def __init__(self, accelerator, model, steps):
                self.accelerator = accelerator
                self.model = model
                self.steps = steps
                self.step_count = 0

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

            def backward(self, loss):
                print(f"step: {self.step_count}, loss: {loss.item()}")
                loss = loss / self.steps  # scale the loss here first
                self.accelerator.backward(loss)
                self.step_count += 1
                if self.step_count % self.steps == 0:
                    print(f"Updating model after {self.steps} steps")
                    return True  # now update
                return False

        return AccumulateContext(self, model, steps)

    @staticmethod
    def save_checkpoint(model, optimizer, path="checkpoint.pth"):
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()}, path)

    @staticmethod
    def load_checkpoint(self, model, optimizer, path="checkpoint.pth"):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            raise FileNotFoundError(f"Checkpoint file {path} not found.")
