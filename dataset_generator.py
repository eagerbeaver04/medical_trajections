from torch.utils.data import Dataset
import torch
from sequence_generator import MedicalSequence
from events import Event, PaddingEvent
from patient_statuses import IStatus, PaddingStatus

class MedicalSequenceDataset(Dataset):
    def __init__(self, sequences: list[MedicalSequence], max_len=None):
        self.sequences = sequences
        if max_len is None:
            max_len = max(len(seq) for seq in sequences)
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    # def __getitem__(self, idx):
    #     seq: MedicalSequence = self.sequences[idx]
    #     seq.pad(self.max_len)
    #     print(seq)
    #     mask = [1] * len(seq) + [0] * (self.max_len - len(seq))
    #     return {
    #         'input_ids': torch.tensor(seq.to_tokens(), dtype=torch.long),
    #         'attention_mask': torch.tensor(mask, dtype=torch.long)
    #     }