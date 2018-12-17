import torch.utils.data as data
from torchrecord import TensorProtos
from torchrecord.transforms import default_transform


class TRDataset(data.Dataset):
    def __init__(self, transform=default_transform, proto=TensorProtos):
        self.transform = transform
        self.proto = proto

    def parse_byte(self, byte_str):
        tensor_protos = self.proto()
        tensor_protos.ParseFromString(byte_str)
        item = self.transform(tensor_protos)
        return item

    def __getitem__(self, item):
        item = self.parse_byte(item)
        return item

    def __len__(self):
        return 1

