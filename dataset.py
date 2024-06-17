import copy
from torch.utils.data import Dataset


class GeneratorDataset(Dataset):
    def __init__(self, rid_data):
        self.rid_data = copy.deepcopy(rid_data)
        self.time_data = [list(range(len(x))) for x in rid_data]

    def __len__(self):
        return len(self.rid_data)
    
    def __getitem__(self, index):
        return (self.rid_data[index], self.time_data[index])
    
class DiscriminatorDataset(Dataset):
    def __init__(self, real_rid_data, fake_rid_data):
        self.rid_data = copy.deepcopy(real_rid_data) + copy.deepcopy(fake_rid_data)
        self.labels = [1 for _ in range(len(real_rid_data))] + [0 for _ in range(len(fake_rid_data))]

    def __len__(self):
        return len(self.rid_data)

    def __getitem__(self, index):
        return (self.rid_data[index], self.labels[index])
