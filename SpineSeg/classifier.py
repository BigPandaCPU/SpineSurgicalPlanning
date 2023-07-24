__copyright__ = \
"""
Copyright &copyright © (c) 2021 Inria Grenoble Rhône-Alpes.
All rights reserved.

This source code is to be used for academic research purposes only.
For commercial uses of the code, please send an email to edmond.boyer@inria.fr and sergi.pujades@inria.fr

"""
__license__ = "CC BY-NC-SA 4.0"
__authors__ = "Di Meng"


import torch.nn as nn 

def conv3d(in_channels, out_channels, kernel_size=3, bias=False, padding=1, stride=1):

    conv3d = []

    conv3d.append(nn.ReplicationPad3d(padding))
    conv3d.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride, bias=bias))

    conv3d.append(nn.BatchNorm3d(out_channels, eps=0.0001, momentum = 0.95))
    conv3d.append(nn.ReLU(inplace=True))

    return conv3d


def downConv3d(in_channels, out_channels, kernel_size=3, pooling=2, bias=False, padding=1):

    downConv = []

    downConv.append(nn.MaxPool3d(pooling))

    conv3d_1 = conv3d(in_channels, out_channels, kernel_size=kernel_size, bias=bias, padding=padding)
    downConv += conv3d_1
    conv3d_2 = conv3d(out_channels, out_channels, kernel_size=kernel_size, bias=bias, padding=padding)
    downConv += conv3d_2

    return downConv



class encoder3d(nn.Module):
    def __init__(self, n_classes, in_channel=1, feature_maps=[16, 32, 64, 128, 256]): # trained with 16, 32, 64, 128, 256
        super(encoder3d, self).__init__()

        self.convStart = nn.Sequential(*(conv3d(in_channel, feature_maps[0]) + conv3d(feature_maps[0], feature_maps[0])))
        self.downConv1 = nn.Sequential(*downConv3d(feature_maps[0], feature_maps[1])) 
        self.downConv2 = nn.Sequential(*downConv3d(feature_maps[1], feature_maps[2]))
        self.downConv3 = nn.Sequential(*downConv3d(feature_maps[2], feature_maps[3]))
        self.downConv4 = nn.Sequential(*downConv3d(feature_maps[3], feature_maps[4]))
        self.downConv5 = nn.Sequential(*downConv3d(feature_maps[4], feature_maps[4]))
        self.downConv6 = nn.Sequential(*downConv3d(feature_maps[4], feature_maps[4]))
        # self.downConv7 = nn.Sequential(*downConv3d(feature_maps[4], feature_maps[4]))
        self.fcn = nn.Linear(feature_maps[4]*(2**3), n_classes)
        # self.outact = nn.Softmax(dim=1)


    def forward(self, x):

        batch_size = x.shape[0]

        x = self.convStart(x) # 128
        x = self.downConv1(x) # 64
        x = self.downConv2(x) # 32
        x = self.downConv3(x) # 16
        x = self.downConv4(x) # 8
        x = self.downConv5(x) # 4
        x = self.downConv6(x) # 2
        # print('check point: ', x.size())
        # x = self.downConv7(x) # 1
        
        x_flat = x.view(batch_size, -1)
        out = self.fcn(x_flat)
        # out = self.outact(out)

        return out 
