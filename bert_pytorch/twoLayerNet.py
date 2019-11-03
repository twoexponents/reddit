import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoLayerNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.bn1 = nn.BatchNorm1d(H)
        #self.linear2 = nn.Linear(H, D_out)
        self.linear2 = nn.Linear(H, H)
        self.bn2 = nn.BatchNorm1d(H)
        self.linear3 = nn.Linear(H, D_out)

    def forward(self, input):
        out = self.bn1(F.relu(self.linear1(input)))
        out = self.bn2(F.relu(self.linear2(out)))
        out = F.softmax(F.relu(self.linear3(out)), dim=1)
        return out

class LSTM(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(LSTM, self).__init__()
        self.D_in = D_in
        self.H = H
        self.lstm = nn.LSTM(D_in, H)
        self.linear1 = nn.Linear(H, H)
        self.bn1 = nn.BatchNorm1d(H)
        self.linear2 = nn.Linear(H, H)
        self.bn2 = nn.BatchNorm1d(H)
        self.linear3 = nn.Linear(H, D_out)

    def forward(self, input):
        out, _ = self.lstm(input)
        out = self.bn1(F.relu(self.linear1(out)))
        out = self.bn2(F.relu(self.linear2(out)))
        out = F.softmax(F.relu(self.linear3(out)), dim=1)
        return out

