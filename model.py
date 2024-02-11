import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torchvision
from typing import Optional, Union
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):

    def __init__(self, n_class, n_seq, n_feature):
        super().__init__()

        self.layer1 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(n_feature)
        self.act1 = nn.ReLU()

        self.layer2 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(n_feature)
        self.act2 = nn.ReLU()

        self.fc = nn.Linear(n_feature*n_seq, n_class)

    def forward(self, x):
        b,t,c = x.size()

        out = self.layer1(x.permute(0,2,1)).to(device)
        
        out = self.act1(self.bn1(out))

        out = self.layer2(out)
        out = self.act2(self.bn2(out))

        logits = self.fc(out.reshape(b,-1))

        return logits
    
class DNN(nn.Module):

    def __init__(self, n_seq, n_feature, n_class):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(n_seq*n_feature, 2048)
        self.act1 = nn.ReLU()
        self.dropout = nn.Dropout()

        self.fc2 = nn.Linear(2048, 1024)
        self.act2 = nn.ReLU()

        self.fc3 = nn.Linear(1024, 512)
        self.act3 = nn.ReLU()

        self.fc4 = nn.Linear(512, 256)
        self.act4 = nn.ReLU()

        self.fc5 = nn.Linear(256, 128)
        self.act5 = nn.ReLU()

        self.fc6 = nn.Linear(128, n_class)
        

    def forward(self, x):
        b, t, c = x.size()
        x = x.reshape(b, -1)
        out = self.dropout(self.act1(self.fc1(x)))
        out = self.act2(self.fc2(out))
        out = self.act3(self.fc3(out))
        out = self.act4(self.fc4(out))
        out = self.act5(self.fc5(out))
        out = self.fc6(out)

        return out