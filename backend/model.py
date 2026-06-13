pip

import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class CNNtoSNNConverter(nn.Module):
    def __init__(self, beta=0.95, slope=25):
        super(CNNtoSNNConverter, self).__init__()
        
        
        self.cnn_extractor = nn.Sequential(
            
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )
        spike_grad = surrogate.fast_sigmoid(slope=slope)
        self.spike_converter = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)

    def forward(self, x):
        spatial_features = self.cnn_extractor(x)
        spikes, membrane_potential = self.spike_converter(spatial_features)
        return spikes, membrane_potential

