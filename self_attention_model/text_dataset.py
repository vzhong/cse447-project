from matplotlib.pyplot import text
import torch
import data_util
from pathlib import Path


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, sequence_length: int, path: Path):
        super().__init__()
        self.sequence_length = sequence_length
        self.indexer = data_util.SymbolIndexer()
        with open(path) as text_data:
          self.data = text_data.read()

    def __len__(self) -> int:
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx: int) -> torch.ByteTensor:
        if idx >= len(self):
            raise IndexError
        return torch.ByteTensor([self.indexer.to_index(s) for s in self.data[idx:idx + self.sequence_length]])
