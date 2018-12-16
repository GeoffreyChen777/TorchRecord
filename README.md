# TorchRecord

![](https://img.shields.io/badge/torchrecord-v0.0.1-blue.svg)

TorchRecord can merge the small files into one or multiple big file to improve the reading performance.

TorchRecord use LMDB as the storage format. A specific writer and loader can be used to write and read the record.

## Reading Performance Benchmark

- Conventional Dataloader with CUB200 datasets(11788 jpg images):

    num_worker = 2, batch_size = 32
    
    100%|██████████████████| 369/369 [00:42<00:00,  8.78it/s]
    
- TorchRecord loader with CUB200 datasets:

    num_worker = 2, batch_size = 32
    
    100%|██████████████████| 369/369 [00:15<00:00, 23.80it/s]
    
## Demo

```python
import os
import random
from torchrecord import Writer
from torchrecord import RecordLoader
from torchrecord import default_data_process_func
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
# Create a RecordLoader and use them as the original dataloader in PyTorch
# record_path: torchrecord path
# batch_size: batch size
# num_workers: the number of workers
# shuffle: if it is True, the loader will shuffle the data before reading them
# =====================================================

loader = RecordLoader(record_path='./test_torchrecord', 
                      batch_size=32, num_workers=2, 
                      shuffle=True)

for i, batch in enumerate(loader):
    pass

```

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
## The Detail of the RecordLoader
```python
class RecordLoader(object):
    def __init__(self, 
         record_path,
         batch_size=32, 
         collate_fn=default_collate_fn, 
         num_workers=1,
         shuffle=True, 
         dataq_maxsize=200, 
         transform=default_transform, 
         dataset=None
    )
```

- dataq_maxsize

    This is the size of the data queue. All the workers will put the images to the data queue. Then we pop 4*batch_size images to the data_pool in the main processing and randomly select batch_size images to construct the batch.
    
- collate_fn

    This is the function for constructing the batch.
    
    The default_collate_fn is like this:
    
    ```python
    def default_collate_fn(data_group):
        return data_group
    ```
    
- transform

    This is the transform function for decode the tensor_protos. You can also do the torchvision.transform here.
    
    The default transform is like this:
    
    ```python
    def default_transform(tensor_protos):
        img_proto = tensor_protos.protos[0]
        img = Image.frombytes(mode='RGB', size=tuple(img_proto.dims), data=img_proto.byte_data)
        label = int(tensor_protos.protos[1].byte_data)
        return img, label
    ```
    
- dataset

    This is the iterable dataset to open the lmdb for workers. You can find the definition in this repository.

