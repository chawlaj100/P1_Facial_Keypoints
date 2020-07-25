## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # Image size = 224*224 ->
        self.conv1 = nn.Conv2d(1, 32, 5) # (224 - 5)/1 + 1 = 220 -> (32, 220, 220)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2) # 46 -> (32, 110, 110)
        self.drop1 = nn.Dropout(p=0.2)
        
        self.conv2 = nn.Conv2d(32, 64, 4) # (110 - 4)/1 + 1 = 43 -> (64, 107, 107)
        self.drop2 = nn.Dropout(p=0.2) # after pooling -> (64, 53, 53)
        
        self.conv3 = nn.Conv2d(64, 128, 3) # (53 - 3)/1 + 1 = 19 -> (128, 51, 51)
        self.drop3 = nn.Dropout(p=0.2) # after pooling -> (128, 25, 25)
        
        self.dense1 = nn.Linear(80000,1000) # 128*25*25 = 80000
        self.drop4 = nn.Dropout(p=0.2)
        
        self.dense2 = nn.Linear(1000,500)
        self.drop5 = nn.Dropout(p=0.2)
        
        self.dense3 = nn.Linear(500,136)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.drop1(self.pool(self.act(self.conv1(x))))
        
        x = self.drop2(self.pool(self.act(self.conv2(x))))
        
        x = self.drop3(self.pool(self.act(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        
        x = self.drop4(self.act(self.dense1(x)))
        
        x = self.drop5(self.act(self.dense2(x)))
        
        out = self.dense3(x)
        return out
