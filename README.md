# TorchRecord

![](https://img.shields.io/badge/torchrecord-v0.1.0-blue.svg)

TorchRecord can merge the small files including images and labels into one or multiple big record file to improve the copying and reading performance.

TorchRecord use LMDB as the storage database. A specific writer and loader can be used to write and read the record.

## Reading Performance Benchmark

Dataset: CUB200 datasets(11788 jpg images)

### Load Image and transform them to tensor (batch_size: 32)

Env: Intel(R) Xeon(R) CPU E5-2603 0 @ 1.80GHz 4core with 32 GB Mem

- num_workers = 2:
    ```
    Conventional:    100%|██████████████████| 369/369 [00:42<00:00,  8.72it/s]
    TRLoader:        100%|██████████████████| 369/369 [00:21<00:00, 16.91it/s]
    ```

- num_workers = 4:
    ```
    Conventional:    100%|██████████████████| 369/369 [00:22<00:00, 16.16it/s]
    TRLoader:        100%|██████████████████| 369/369 [00:13<00:00, 26.73it/s]
    ```
## Installation

```bash
pip install torchrecord
```
**pillow-simd** which is a faster folk of Pillow is recommended.

## Demo

```python
import os
import random
from torchrecord import default_data_process_func
from torchrecord import Writer, TorchRecord
# =====================================================
# Make data list (txt or csv), one data item per line.
# The template of data_list:
#
# path/to/image/img1.jpg 1
# path/to/image/img2.jpg 2
# ...
# =====================================================

with open('./data_list.txt', 'w') as writer:
    for p, d, fl in os.walk('./testdata'):
        for f in fl:
            if f.endswith('jpg'):
                writer.write("{} {}\n".format(os.path.join(p, f), random.randint(0, 10)))


# =====================================================
# Use torchrecord.Writer to write the torchrecord.
# data_list: the data list
# output_dir: the path of the torchrecord
# db_num: split the origin dataset to n subset
# shuffle: if it is True, the writer will shuffle all the data before writing them to the torchrecord
# data_process_func: the function for processing the data item
# =====================================================

writer = Writer(data_list='./data_list.txt', output_dir='./test_torchrecord', 
                db_num=4, shuffle=True, data_process_func=default_data_process_func)
writer.write()


# =====================================================
# Create a TorchRecord which is very similar to the dataloader of PyTorch
# =====================================================

loader = TorchRecord(record_path='./testdb', record_num=4, shuffle=True, batch_size=32, num_workers=4)

for i, batch in enumerate(loader):
    pass

```

## About Memory Usage

LMDB is a Lightning Memory-Mapped Database. It will load the data into your memory. If the memory of your PC is insufficient, the loading speed might be slow due to the memory cache replacement. In order to solve this problem, TorchRecord use `record_num` to split dataset into several sub record. TRrecord will load all the sub record one by one and close trained record to release the memory.

## The Detail of the Writer

- data_process_func:

    We list all the data path and label in the data_list file, and use the `data_process_func` to process the data and labels.

    You can define your own `data_process_func` and send them as a parameter to initialize the Writer.

    The `default_data_process_func` are as follows:

    ```python
    def default_data_process_func(data):
        data = data.split(' ')
        tensor_protos = TensorProtos()

        img = Image.open(data[0]).convert("RGB")
        img_tensor = tensor_protos.protos.add()
        img_tensor.dims.extend(img.size)
        img_tensor.data_type = 3
        img_tensor.byte_data = img.tobytes()

        label_tensor = tensor_protos.protos.add()
        label_data = str.encode(data[1])
        label_tensor.data_type = 3
        label_tensor.byte_data = label_data
        return tensor_protos
    ```

    The `data` parameter is one line of your data_list file. 

    We use the TensorProtos which is a *Buffer Protocols* (designed by Google) object as the data structure. You can save your data in TensorProtos
    and return it. Then, the writer will save the TensorProtos into LMDB.
    

## The Detail of the TorchRecord

The TorchRecord are composed with a TRSampler, a TRDataset ,and a Dataloader. You can read the torchrecord simply by the TorchRecord object. 

The TRSampler is used to sample the key index from the torchrecord(lmdb). `full_shuffle` is under developing.
```python
class TRSampler(data.Sampler):
    def __init__(self, record_path, record_num, shuffle=False, full_shuffle=False, batch_size=1, num_workers=1):
```

Then, the dataloader will distribute the key index sampled by the TRSampler to TRDataset. TRDataset will seek the raw byte string of the key index from lmdb and parse it to the TensorProtos. After that, the transform will be applied to it. Finally, TRDataset will return the image tensor and the label to the dataloader. The dataloader will use the `collate_fn` to construct the batch.

```python
class TRDataset(data.Dataset):
    def __init__(self, record_path='', transform=default_transform, record_num=1, shuffle=False, batch_size=1, proto=TensorProtos):
```

TRDataset use a item pool(size: min(4*batch_size, 64)) to shuffle the sampled sequence. TRDataset will put the sampled item to the pool and then random get one from the pool.

- transform

    This is the transform function for decode the byte string to the tensor_protos. You can also apply any torchvision transforms here.

    The default transform is like this:

    ```python
    from PIL import Image
    import torchvision.transforms as tvt


    trans = tvt.Compose([
        tvt.Resize((224, 224)),
        tvt.ToTensor()
    ])

    def default_transform(tensor_protos):
        img_proto = tensor_protos.protos[0]
        img = Image.frombytes(mode='RGB', size=tuple(img_proto.dims), data=img_proto.byte_data)
        img = trans(img)
        label = int(tensor_protos.protos[1].byte_data)
        return img, label
    ```

You can alos create your own dataloader with the TRSampler, a TRDataset manually.
