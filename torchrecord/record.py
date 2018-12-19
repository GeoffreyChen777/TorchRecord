from .dataloader import DataLoader
from .dataset import TRDataset
from .sampler import TRSampler
from .transforms import default_transform

def TorchRecord(record_path='', record_num=1, shuffle=True, batch_size=1, num_workers=1, transform=default_transform):
    dataset = TRDataset(record_path=record_path, record_num=record_num, shuffle=shuffle, batch_size=batch_size, transform=transform)
    sampler = TRSampler(record_path=record_path, shuffle=shuffle, batch_size=batch_size, record_num=record_num, num_workers=num_workers)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers)
