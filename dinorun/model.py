import configparser

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .settings import settings

config = configparser.ConfigParser()
config.read('./config.ini')


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=settings['img_channels'],
                                  out_channels=32,
                                  kernel_size=(8, 8),
                                  stride=(4, 4))
        self.conv2d_2 = nn.Conv2d(in_channels=32,
                                  out_channels=64,
                                  kernel_size=(4, 4),
                                  stride=(2, 2))
        self.conv2d_3 = nn.Conv2d(in_channels=64,
                                  out_channels=64,
                                  kernel_size=(3, 3),
                                  stride=(1, 1))
        self.fc = nn.Linear(in_features=256, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=settings['num_actions'])

    def forward(self, x: Tensor) -> Tensor:
        conv1 = F.relu(F.max_pool2d(F.pad(self.conv2d_1(x), pad=(1, 1, 1, 1)),
                                    kernel_size=(2, 2)))
        conv2 = F.relu(F.max_pool2d(F.pad(self.conv2d_2(conv1), pad=(3, 3, 3, 3)),
                                    kernel_size=(2, 2)))
        conv3 = F.relu(F.max_pool2d(F.pad(self.conv2d_3(conv2), pad=(1, 1, 1, 1)),
                                    kernel_size=(2, 2)))
        flatten = conv3.view(-1, 64 * 2 * 2)
        fc = self.fc(flatten)
        out = self.fc2(fc)
        return out
