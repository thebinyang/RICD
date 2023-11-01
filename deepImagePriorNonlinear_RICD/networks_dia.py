# -*- coding: utf-8 -*-
"""
Original Code Author: Sudipan Saha.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from deform_conv import DeformConv2D
import numpy as np



# CNN model 4 channel
def conv_fixup_init(layer, scaling_factor=1):
    # here we assume that kernel is squared matrix
    k = layer.kernel_size[0]
    n = layer.out_channels
    sigma = np.sqrt(2 / (k * k * n)) * scaling_factor
    layer.weight.data.normal_(0, sigma)
    return layer

class ModelDeepImagePriorHyperspectralNonLinear2Layer(nn.Module):
    def __init__(self, numberOfImageChannels, nFeaturesIntermediateLayers):
        super(ModelDeepImagePriorHyperspectralNonLinear2Layer, self).__init__()

        kernelSize = 3
        paddingSize = int((kernelSize - 1) / 2)
        self.conv1 = nn.Conv2d(numberOfImageChannels, nFeaturesIntermediateLayers, kernel_size=kernelSize, stride=1,
                               padding=paddingSize)
        self.conv1.weight = torch.nn.init.kaiming_uniform_(self.conv1.weight)

        self.conv2 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize,
                               stride=1, padding=paddingSize)
        self.conv2.weight = torch.nn.init.kaiming_uniform_(self.conv2.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)

        return x


class ModelDeepImagePriorHyperspectralNonLinear3Layer(nn.Module):
    def __init__(self, numberOfImageChannels, nFeaturesIntermediateLayers):
        super(ModelDeepImagePriorHyperspectralNonLinear3Layer, self).__init__()

        kernelSize = 3
        paddingSize = int((kernelSize - 1) / 2)
        self.conv1 = nn.Conv2d(numberOfImageChannels, nFeaturesIntermediateLayers, kernel_size=kernelSize, stride=1,
                               padding=paddingSize)
        self.conv1.weight = torch.nn.init.kaiming_uniform_(self.conv1.weight)

        self.conv2 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize,
                               stride=1, padding=paddingSize)
        self.conv2.weight = torch.nn.init.kaiming_uniform_(self.conv2.weight)

        self.conv3 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize,
                               stride=1, padding=paddingSize)
        self.conv3.weight = torch.nn.init.kaiming_uniform_(self.conv3.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)

        return x


class ModelDeepImagePriorHyperspectralNonLinear4Layer(nn.Module):
    def __init__(self, numberOfImageChannels, nFeaturesIntermediateLayers):
        super(ModelDeepImagePriorHyperspectralNonLinear4Layer, self).__init__()

        kernelSize = 3
        paddingSize = int((kernelSize - 1) / 2)
        self.conv1 = nn.Conv2d(numberOfImageChannels, nFeaturesIntermediateLayers, kernel_size=kernelSize, stride=1,
                               padding=paddingSize)
        self.conv1.weight = torch.nn.init.kaiming_uniform_(self.conv1.weight)

        self.conv2 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize,
                               stride=1, padding=paddingSize)
        self.conv2.weight = torch.nn.init.kaiming_uniform_(self.conv2.weight)

        self.conv3 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize,
                               stride=1, padding=paddingSize)
        self.conv3.weight = torch.nn.init.kaiming_uniform_(self.conv3.weight)

        self.conv4 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize,
                               stride=1, padding=paddingSize)
        self.conv4.weight = torch.nn.init.kaiming_uniform_(self.conv4.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)

        return x

# ######################RICD
class ModelDeepImagePriorHyperspectralNonLinear5Layer(nn.Module):
    def __init__(self, numberOfImageChannels, nFeaturesIntermediateLayers):
        super(ModelDeepImagePriorHyperspectralNonLinear5Layer, self).__init__()

        kernelSize = 3
        paddingSize = int((kernelSize - 1) / 2)
        self.conv1 = nn.Conv2d(numberOfImageChannels, nFeaturesIntermediateLayers, kernel_size=kernelSize, stride=1,
                               padding=paddingSize)
        self.conv1.weight = torch.nn.init.kaiming_uniform_(self.conv1.weight)

        self.conv21 = nn.Conv2d(nFeaturesIntermediateLayers, nFeaturesIntermediateLayers, kernel_size=kernelSize, stride=1,
                                padding=paddingSize, dilation=1)
        self.conv22 = nn.Conv2d(nFeaturesIntermediateLayers, nFeaturesIntermediateLayers, kernel_size=kernelSize, stride=1,
                                padding=paddingSize * 2, dilation=2)

        self.conv21.weight = torch.nn.init.kaiming_uniform_(self.conv21.weight)
        self.conv22.weight = torch.nn.init.kaiming_uniform_(self.conv22.weight)

        # nFeaturesIntermediateLayers=nFeaturesIntermediateLayers*2
        self.offsets3 = nn.Conv2d(nFeaturesIntermediateLayers * 2, 18, kernel_size=kernelSize, padding=paddingSize)
        self.offsets3.weight = torch.nn.init.kaiming_uniform_((self.offsets3.weight))

        self.conv3 = DeformConv2D(nFeaturesIntermediateLayers * 2, nFeaturesIntermediateLayers, kernel_size=kernelSize,
                                  padding=paddingSize)
        self.conv4 = nn.Conv2d(nFeaturesIntermediateLayers * 2, nFeaturesIntermediateLayers, kernel_size=kernelSize,
                               stride=1, padding=paddingSize)
        self.conv4.weight = torch.nn.init.kaiming_uniform_(self.conv4.weight)

        self.conv41 = nn.Conv2d(nFeaturesIntermediateLayers, nFeaturesIntermediateLayers, kernel_size=kernelSize,
                                stride=1,padding=paddingSize, dilation=1)
        self.conv42 = nn.Conv2d(nFeaturesIntermediateLayers, nFeaturesIntermediateLayers, kernel_size=kernelSize,
                                stride=1,padding=paddingSize * 2, dilation=2)

        self.conv41.weight = torch.nn.init.kaiming_uniform_(self.conv41.weight)
        self.conv42.weight = torch.nn.init.kaiming_uniform_(self.conv42.weight)

        # nFeaturesIntermediateLayers = nFeaturesIntermediateLayers * 2
        self.offsets5 = nn.Conv2d(nFeaturesIntermediateLayers * 2, 18, kernel_size=kernelSize, padding=paddingSize)
        self.offsets5.weight = torch.nn.init.kaiming_uniform_((self.offsets5.weight))

        self.conv5 = DeformConv2D(nFeaturesIntermediateLayers * 2, nFeaturesIntermediateLayers, kernel_size=kernelSize,
                                  padding=paddingSize)

        self.conv8 = nn.Conv2d(nFeaturesIntermediateLayers * 3, nFeaturesIntermediateLayers, kernel_size=kernelSize,
                               stride=1,padding=paddingSize)
        self.conv8.weight = torch.nn.init.kaiming_uniform_(self.conv8.weight)


    def forward(self, x):

        x1 = F.relu(self.conv1(x))
        x21 = F.relu(self.conv21(x1))  # nFeaturesIntermediateLayers, nFeaturesIntermediateLayers
        x22 = F.relu(self.conv22(x1))
        x2 = torch.cat((x21, x22), dim=1)
        offsets3 = self.offsets3(x2)
        x3 = self.conv3(x2, offsets3)  # nFeaturesIntermediateLayers * 2, nFeaturesIntermediateLayers
        x3 = F.relu(x3)
        x31 = torch.cat((x1, x3), dim=1)
        x31 = self.conv4(x31)  # nFeaturesIntermediateLayers * 2, nFeaturesIntermediateLayers
        x41 = F.relu(self.conv41(x31))
        x42 = F.relu(self.conv42(x31))
        x4 = torch.cat((x41, x42), dim=1)
        offsets5 = self.offsets5(x4)
        x5 = self.conv5(x4, offsets5)  # nFeaturesIntermediateLayers * 2, nFeaturesIntermediateLayers
        x5 = F.relu(x5)
        x = torch.cat((x1, x3, x5), dim=1)
        x = self.conv8(x)  # nFeaturesIntermediateLayers * 3, nFeaturesIntermediateLayers

        return x


class ModelDeepImagePriorHyperspectralNonLinear6Layer(nn.Module):
    def __init__(self, numberOfImageChannels, nFeaturesIntermediateLayers):
        super(ModelDeepImagePriorHyperspectralNonLinear6Layer, self).__init__()

        kernelSize = 3
        paddingSize = int((kernelSize - 1) / 2)
        self.conv1 = nn.Conv2d(numberOfImageChannels, nFeaturesIntermediateLayers, kernel_size=kernelSize, stride=1,
                               padding=paddingSize)
        self.conv1.weight = torch.nn.init.kaiming_uniform_(self.conv1.weight)

        self.conv2 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize,
                               stride=1, padding=paddingSize)
        self.conv2.weight = torch.nn.init.kaiming_uniform_(self.conv2.weight)

        self.conv3 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize,
                               stride=1, padding=paddingSize)
        self.conv3.weight = torch.nn.init.kaiming_uniform_(self.conv3.weight)

        self.conv4 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize,
                               stride=1, padding=paddingSize)
        self.conv4.weight = torch.nn.init.kaiming_uniform_(self.conv4.weight)

        self.conv5 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize,
                               stride=1, padding=paddingSize)
        self.conv5.weight = torch.nn.init.kaiming_uniform_(self.conv5.weight)

        self.conv6 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize,
                               stride=1, padding=paddingSize)
        self.conv6.weight = torch.nn.init.kaiming_uniform_(self.conv6.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)

        return x


class ModelDeepImagePriorHyperspectralNonLinear7Layer(nn.Module):
    def __init__(self, numberOfImageChannels, nFeaturesIntermediateLayers):
        super(ModelDeepImagePriorHyperspectralNonLinear7Layer, self).__init__()

        kernelSize = 3
        paddingSize = int((kernelSize - 1) / 2)
        self.conv1 = nn.Conv2d(numberOfImageChannels, nFeaturesIntermediateLayers, kernel_size=kernelSize, stride=1,
                               padding=paddingSize)
        self.conv1.weight = torch.nn.init.kaiming_uniform_(self.conv1.weight)

        self.conv2 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize,
                               stride=1, padding=paddingSize)
        self.conv2.weight = torch.nn.init.kaiming_uniform_(self.conv2.weight)

        self.conv3 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize,
                               stride=1, padding=paddingSize)
        self.conv3.weight = torch.nn.init.kaiming_uniform_(self.conv3.weight)

        self.conv4 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize,
                               stride=1, padding=paddingSize)
        self.conv4.weight = torch.nn.init.kaiming_uniform_(self.conv4.weight)

        self.conv5 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize,
                               stride=1, padding=paddingSize)
        self.conv5.weight = torch.nn.init.kaiming_uniform_(self.conv5.weight)

        self.conv6 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize,
                               stride=1, padding=paddingSize)
        self.conv6.weight = torch.nn.init.kaiming_uniform_(self.conv6.weight)

        self.conv7 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize,
                               stride=1, padding=paddingSize)
        self.conv7.weight = torch.nn.init.kaiming_uniform_(self.conv7.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.conv7(x)

        return x


class ModelDeepImagePriorHyperspectralNonLinear8Layer(nn.Module):
    def __init__(self, numberOfImageChannels, nFeaturesIntermediateLayers):
        super(ModelDeepImagePriorHyperspectralNonLinear8Layer, self).__init__()

        kernelSize = 3
        paddingSize = int((kernelSize - 1) / 2)
        self.conv1 = nn.Conv2d(numberOfImageChannels, nFeaturesIntermediateLayers, kernel_size=kernelSize, stride=1,
                               padding=paddingSize)
        self.conv1.weight = torch.nn.init.kaiming_uniform_(self.conv1.weight)

        self.conv2 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize,
                               stride=1, padding=paddingSize)
        self.conv2.weight = torch.nn.init.kaiming_uniform_(self.conv2.weight)

        self.conv3 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize,
                               stride=1, padding=paddingSize)
        self.conv3.weight = torch.nn.init.kaiming_uniform_(self.conv3.weight)

        self.conv4 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize,
                               stride=1, padding=paddingSize)
        self.conv4.weight = torch.nn.init.kaiming_uniform_(self.conv4.weight)

        self.conv5 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize,
                               stride=1, padding=paddingSize)
        self.conv5.weight = torch.nn.init.kaiming_uniform_(self.conv5.weight)

        self.conv6 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize,
                               stride=1, padding=paddingSize)
        self.conv6.weight = torch.nn.init.kaiming_uniform_(self.conv6.weight)

        self.conv7 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize,
                               stride=1, padding=paddingSize)
        self.conv7.weight = torch.nn.init.kaiming_uniform_(self.conv7.weight)

        self.conv8 = nn.Conv2d(nFeaturesIntermediateLayers, int(nFeaturesIntermediateLayers), kernel_size=kernelSize,
                               stride=1, padding=paddingSize)
        self.conv8.weight = torch.nn.init.kaiming_uniform_(self.conv8.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.conv7(x)
        x = F.relu(x)
        x = self.conv8(x)

        return x



