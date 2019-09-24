import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoLayerNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        # add batch normalization soon
        self.layer1 = nn.Linear(D_in, H)
        self.layer2 = nn.Linear(H, D_out)
        #self.layer1 = 

    def forward(self, input):
        out = F.relu(self.layer1(input))
        out = torch.sigmoid(self.layer2(out))
        return out

