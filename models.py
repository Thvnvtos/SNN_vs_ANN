import torch
import torch.nn as nn

class ANN(nn.Module):
    # in_c : input channels, fm : feature maps, ks : kernel size
    def __init__(self, in_c = 1,fm = [8,16], ks=[5,5]):
        super().__init__()
        
        self.convLayer1 = nn.Sequential(
            nn.Conv2d(in_c, fm[0], kernel_size=ks[0]),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
            )
        self.dim1 = (28 - ks[0] + 1) // 2
        self.convLayer2 = nn.Sequential(
            nn.Conv2d(fm[0],fm[1], kernel_size=ks[1]),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
            )
        self.dim2 = (self.dim1 - ks[1] + 1) // 2
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fm[1]*self.dim2**2, 10),
            nn.Softmax(dim=1)
            )

    def forward(self, x):
        x = self.convLayer1(x)
        x = self.convLayer2(x)
        return self.fc(x)


class SNN(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass