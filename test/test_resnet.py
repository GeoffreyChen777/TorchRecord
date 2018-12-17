import torch
import sys
import torch.nn as nn
from torchvision.models.resnet import resnet18
import torch.utils.data as data
from torchrecord import TRDataset, TRSampler
import time


class Timer(object):
    def __init__(self, iter_length):
        self.start_time = 0
        self.iter_length = iter_length

    def start(self):
        self.start_time = time.time()

    def stamp(self, step):
        time_duration = time.time() - self.start_time
        rest_time = time_duration / (step+1) * (self.iter_length - step - 1)
        cur_hour, cur_min, cur_sec = self.convert_format(time_duration)
        rest_hour, rest_min, rest_sec =  self.convert_format(rest_time)
        log_string = "[{}:{}:{} < {}:{}:{}]".format(cur_hour, cur_min, cur_sec, rest_hour, rest_min, rest_sec)
        return log_string

    def stop(self):
        time_duration = time.time() - self.start_time
        cur_hour, cur_min, cur_sec = self.convert_format(time_duration)
        log_string = "[{}:{}:{}]".format(cur_hour, cur_min, cur_sec)
        self.start_time = 0
        return log_string

    @staticmethod
    def convert_format(sec):
        hour = "{:02}".format(int(sec // 3600))
        minu = "{:02}".format(int(sec // 60))
        sec = "{:02}".format(int(sec % 60))
        return hour, minu, sec


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet = resnet18(pretrained=True).to(device)
resnet.fc = nn.Linear(resnet.fc.in_features, 200).to(device)

cri = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

dataset = TRDataset()
sampler = TRSampler('./testdb', shuffle=True, batch_size=64)
loader = data.DataLoader(dataset, batch_sampler=sampler, num_workers=2)

timer = Timer(len(loader))

total_loss = 0
running_corrects = 0

for epoch in range(50):
    print("================ Epoch {} ================".format(epoch))
    timer.start()
    for i, batch in enumerate(loader):
        optimizer.zero_grad()
        img, label = batch[0].to(device), batch[1].to(device) - 1
        output = resnet(img)

        loss = cri(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 20 == 19:
            print("{} | Loss: {:.4}".format(timer.stamp(i), total_loss/20))
            total_loss = 0

        _, preds = torch.max(output, 1)
        running_corrects += torch.sum((preds == label).float()).item()
    print("Acc. {:.4}".format((1.*running_corrects)/(1.*len(loader))))
    running_corrects = 0
