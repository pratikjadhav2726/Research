import copy
import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np
class DeepNN(nn.Module):
    def __init__(self):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        # self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the images
        x = self.dropout(F.relu(self.fc1(x)))
        # x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.fc5(x)
        return x
    
class LRPDropout(nn.Module):
    def __init__(self,dropoutrate=0.5):
        super(LRPDropout, self).__init__()
        self.mask = None
        self.rate=dropoutrate

    def forward(self, x):
        if self.training:
            if self.mask is None:
              return x
            # During training, apply dropout
            # print(self.mask.shape,x.shape)
            output = (x * self.mask )/ (1 - self.rate)
        else:
            # During evaluation, don't apply dropout
            output = x
        return output



    def update_mask(self, lrp_values, percentile=50):
        percentile=100-self.rate*100
        # calculate the threshold based on LRP values
        threshold = np.percentile((torch.abs(lrp_values).cpu().numpy()), percentile)

        # create a binary mask based on the threshold
        self.mask = (torch.abs(lrp_values) < threshold).float().to("cuda")
        # print(self.mask)

    def show_mask(self):
        print(self.mask)





    
