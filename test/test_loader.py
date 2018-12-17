from torchrecord import TRDataset
from tqdm import tqdm
import torch.utils.data as data
from test.conv_dataset import CUB200
from torchrecord.transforms import default_transform
from torchrecord import TRSampler


dataset = TRDataset(transform=default_transform)
sampler = TRSampler('./testdb', shuffle=True, batch_size=32)
loader = data.DataLoader(dataset, batch_sampler=sampler, num_workers=2)
for i, batch in enumerate(tqdm(loader)):
    pass

dataset = CUB200()
loader = data.DataLoader(dataset, batch_size=32, num_workers=2, shuffle=True)
for i, batch in enumerate(tqdm(loader)):
    pass

