import numpy as np
import torch.utils.data as data
from PIL import Image
import os


def default_loader(img_path):
    img = Image.open(img_path)
    img = img.convert("RGB")
    return img


class CUB200(data.Dataset):
    def __init__(self, loader=default_loader):
        self.loader = loader
        self.image_index, self.label_idx = self._load_index()

    def __getitem__(self, item_index):
        image_path = os.path.join(self.image_index[item_index])
        img = self.loader(image_path)
        label = self.label_idx[item_index]
        return img, label

    def __len__(self):
        return len(self.image_index)

    @staticmethod
    def collate_fn(batch):
        imgs, labels = [], []
        for batch_block in batch:
            imgs.append(batch_block[0])
            labels.append(batch_block[1])
        return imgs, labels

    def _load_index(self):
        data_path = os.path.join('./data_list.txt')
        with open(data_path, 'r') as reader:
            data_index = reader.readlines()

        image_index = [x.strip().split(' ')[0] for x in data_index]
        label_index = [int(x.strip().split(' ')[1]) for x in data_index]

        return image_index, label_index
