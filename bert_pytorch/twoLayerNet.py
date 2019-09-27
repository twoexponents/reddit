import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoLayerNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        # add batch normalization soon
        self.linear1 = nn.Linear(D_in, H)
        self.bn1 = nn.BatchNorm1d(H)
        self.linear2 = nn.Linear(H, H)
        self.bn2 = nn.BatchNorm1d(H)
        self.linear3 = nn.Linear(H, D_out)

    def forward(self, input):
        out = F.relu(self.bn1(self.linear1(input)))
        out = F.relu(self.bn2(self.linear2(out)))
        out = F.softmax(self.linear3(out), dim=1)
        return out

