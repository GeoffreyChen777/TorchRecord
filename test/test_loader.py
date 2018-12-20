from torchrecord import TorchRecord
from tqdm import tqdm


loader = TorchRecord('./testdb', 4, True, 32, 2)

for epoch in range(10):
    check_sum = 0
    for i, batch in enumerate(tqdm(loader)):
        check_sum += batch[1].sum().item()
    print(check_sum)


# dataset = CUB200()
# loader = data.DataLoader(dataset, num_workers=4, batch_size=32, shuffle=True)
# for epoch in range(10):
#     for i, batch in enumerate(tqdm(loader)):
#         pass