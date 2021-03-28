import torch
import torch.nn as nn
from spikingjelly.clock_driven import neuron, functional, surrogate, layer

class ANN(nn.Module):
    # in_c : input channels, fm : feature maps, ks : kernel size
    def __init__(self, in_c = 1,fm = [12,64], ks=[5,5]):
        super().__init__()
        
        self.convLayer1 = nn.Sequential(
            nn.Conv2d(in_c, fm[0], kernel_size=ks[0], bias=False),
            nn.BatchNorm2d(fm[0]),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
            )
        self.dim1 = (28 - ks[0] + 1) // 2
        self.convLayer2 = nn.Sequential(
            nn.Conv2d(fm[0],fm[1], kernel_size=ks[1], bias=False),
            nn.BatchNorm2d(fm[1]),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
            )
        self.dim2 = (self.dim1 - ks[1] + 1) // 2
        self.fc = nn.Sequential(
            nn.Flatten(),
            # check using dropout
            nn.Linear(fm[1]*self.dim2**2, 10, bias=False),
            nn.Softmax(dim=1)
            )

    def forward(self, x):
        x = self.convLayer1(x)
        x = self.convLayer2(x)
        return self.fc(x)


class SNN(nn.Module):
    def __init__(self, in_c = 1,fm = [12,64], ks=[5,5], tau = 2.0, T = 10, v_threshold = 1.0, v_reset = 0.0):
        super().__init__()
        self.T = T

        self.static_conv = nn.Sequential(
            nn.Conv2d(in_c, fm[0], kernel_size=ks[0], bias=False),
            nn.BatchNorm2d(fm[0])
            )

        self.dim1 = (28 - ks[0] + 1) // 2
        self.conv = nn.Sequential(
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(fm[0], fm[1], kernel_size=ks[1], bias=False),
            nn.BatchNorm2d(fm[1]),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2)
            )

        self.dim2 = (self.dim1 - ks[1] + 1) // 2
        self.fc = nn.Sequential(
            nn.Flatten(),
            # check using dropout
            nn.Linear(fm[1]*self.dim2**2, 10, bias=False),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            )

    def forward(self, x):
        x = self.static_conv(x)
        out_spikes_counter = self.fc(self.conv(x))
        for t in range(1, self.T):
            out_spikes_counter += self.fc(self.conv(x))

        return out_spikes_counter / self.T