import abc

from torch.utils.data import Dataset


class HoVerDatasetBase(Dataset, abc.ABC):
    @abc.abstractmethod
    def __len__(self):
        return NotImplemented

    @abc.abstractmethod
    def __getitem__(self, idx):
        return NotImplemented

    @abc.abstractmethod
    def load_all_data(self):
        return NotImplemented

    @abc.abstractmethod
    def load_data(self, idx):
        return NotImplemented
