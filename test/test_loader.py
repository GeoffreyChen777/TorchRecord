from torchrecord import TRDataset
from tqdm import tqdm
import torch.utils.data as data
from torchrecord.transforms import default_transform
from torchrecord import TRSampler
from test.conv_dataset import CUB200
import torch


dataset = TRDataset(transform=default_transform)
sampler = TRSampler('./testdb', shuffle=True, batch_size=32, record_num=4)
loader = data.DataLoader(dataset, batch_sampler=sampler, num_workers=2)
for epoch in range(10):
    for i, batch in enumerate(tqdm(loader)):
        pass


# dataset = CUB200()
# loader = data.DataLoader(dataset, num_workers=2, batch_size=32, shuffle=True)
# for epoch in range(10):
#     for i, batch in enumerate(tqdm(loader)):
#         pass