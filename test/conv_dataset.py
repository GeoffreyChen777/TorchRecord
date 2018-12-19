import numpy as np
import torch.utils.data as data
from PIL import Image
import os
import torchvision.transforms as tt
import torch
import torchvision.transforms as tt


t = tt.Compose([
    tt.Resize((224, 224)),
    tt.ToTensor(),
    tt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def default_loader(img_path):
    img = Image.open(img_path)
    img = t(img.convert("RGB"))
    return img


class CUB200(data.Dataset):
    def __init__(self, loader=default_loader):
        self.loader = loader
        self.image_index, self.label_idx = self._load_index()
        self.count = 0

    def __getitem__(self, item_index):
        image_path = os.path.join('../data/datasets/CUB_200/CUB_200_2011/images/', self.image_index[item_index].replace('testdata/', ''))
        img = self.loader(image_path)
        label = torch.tensor(int(self.label_idx[item_index]))
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
