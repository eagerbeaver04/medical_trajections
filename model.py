#!/usr/bin/env python
# coding: utf-8

# 

# In[11]:


import torch
import numpy as np
from torch.utils.data import Dataset

from dataset_generator import MedicalSequenceDataset
from sequence_generator import MedicalSequenceGenerator, MedicalSequence
from transisition_model import TransitionModel


# In[12]:


N = 4

start_probs = [0.4, 0.3, 0.2, 0.1]

cab_matrix = [
    [0.00, 0.388, 0.409, 0.153],   # 0.00+0.40+0.40+0.18 = 0.98
    [0.184, 0.00, 0.409, 0.347],   # 0.20+0.40+0.38 = 0.98
    [0.30, 0.10, 0.00, 0.51],   # 0.30+0.10+0.58 = 0.98
    [0.53, 0.183, 0.137, 0.00]    # 0.60+0.20+0.18 = 0.98
]
end_probs  = [0.05, 0.06, 0.09, 0.15]
# survive_probs= [0.03, 0.03, 0.05, 0.05]
rng = np.random.default_rng()        # no seed = system entropy
model = TransitionModel(N)
model.build_from_matrices(start_probs, cab_matrix, end_probs)

generator = MedicalSequenceGenerator(model)
sequences: list[MedicalSequence] = generator.generate_many(50000)


# In[13]:


def print_head_of_sequences(sequences: list[MedicalSequence]):
    print_size = min(len(sequences), 5)
    print("Token ID sequences:")
    for seq in sequences[:print_size]:
        seq.print_by_tokens()

    print("Named sequences:")
    for seq in sequences[:print_size]:
        seq.print_by_keys()


# In[14]:


print_head_of_sequences(sequences)


# In[15]:


dataset = MedicalSequenceDataset(sequences)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

