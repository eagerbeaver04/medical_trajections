#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
from torch.utils.data import Dataset

from utils.gpu_check import get_device


# In[2]:


device, CUDA = get_device()


# In[3]:


from structures.medical_sequence import MedicalSequence


# In[4]:


def print_head_of_sequences(sequences: list[MedicalSequence]):
    print_size = min(len(sequences), 5)
    for seq in sequences[:print_size]:
        print(seq)


# In[5]:


from dataset.sequence_generator import SequenceGenerator
from dataset.medical_sequence_dataset import MedicalSequenceDataset


# In[6]:


dataset = MedicalSequenceDataset.from_generator(
    generator=SequenceGenerator(),
    n_sequences=100,
    max_steps=8,
    pad_cabinet_token_id=0,
    pad_to_max_length=True,
    encode_condition_pad_for_model=True,
)


# In[7]:


print(len(dataset))
print(dataset.summary())


# In[ ]:


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out

def train_one_epoch(
    model: TransformerWrapper,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()

    total_loss = 0.0
    total_cabinet_loss = 0.0
    total_condition_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad()

        losses = model.compute_losses(batch)
        loss = losses["loss"]

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cabinet_loss += losses["cabinet_loss"].item()
        total_condition_loss += losses["condition_loss"].item()
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "cabinet_loss": total_cabinet_loss / max(n_batches, 1),
        "condition_loss": total_condition_loss / max(n_batches, 1),
    }

@torch.no_grad()
def evaluate(
    model: TransformerWrapper,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_cabinet_loss = 0.0
    total_condition_loss = 0.0
    n_batches = 0

    cabinet_correct = 0
    cabinet_total = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        losses = model.compute_losses(batch)

        total_loss += losses["loss"].item()
        total_cabinet_loss += losses["cabinet_loss"].item()
        total_condition_loss += losses["condition_loss"].item()
        n_batches += 1

        cabinet_logits = losses["cabinet_logits"]      # [B, Tk, V]
        cabinet_targets = batch["cabinets"]            # [B, Tk]
        cabinet_mask = batch["cabinet_mask"].bool()    # [B, Tk]

        preds = cabinet_logits.argmax(dim=-1)
        cabinet_correct += (preds[cabinet_mask] == cabinet_targets[cabinet_mask]).sum().item()
        cabinet_total += cabinet_mask.sum().item()

    return {
        "loss": total_loss / max(n_batches, 1),
        "cabinet_loss": total_cabinet_loss / max(n_batches, 1),
        "condition_loss": total_condition_loss / max(n_batches, 1),
        "cabinet_acc": cabinet_correct / max(cabinet_total, 1),
    }

