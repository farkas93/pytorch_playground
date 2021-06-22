import torch
from torch import nn


# Define model
class VGGLikeNetwork(nn.Module):
    def __init__(self):
        super(VGGLikeNetwork, self).__init__()
        self.stack = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Image will be 14x14
            
            nn.Flatten(),
            nn.Linear(14*14*64, 4096),
            nn.ReLU(),
            nn.Linear(4096, 10),
            nn.ReLU()
        )

    def forward(self, x):
        logits = self.stack(x)
        return logits