from torchrecord import RecordLoader
from tqdm import tqdm
import torch.utils.data as data
from conv_dataset import CUB200

dbloader = RecordLoader(db_path='./testdb', num_workers=2)
dataset = CUB200()
origin_loader = data.DataLoader(dataset, batch_size=32,
                                shuffle=True, num_workers=2, collate_fn=dataset.collate_fn)

# for i, batch in enumerate(tqdm(dbloader)):
#     pass

for i, batch in enumerate(tqdm(origin_loader)):
    pass
