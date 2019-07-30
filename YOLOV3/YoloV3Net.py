import torch
import torch.nn as nn
import numpy as np


class Convolution_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, padding=0):
        super().__init__()
        self.conv_layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, ksize, stride, padding),
                                        nn.BatchNorm2d(out_channels),
                                        nn.PReLU())

    def forward(self, x):
        return self.conv_layer(x)


class Residual_Layer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.res_layer = nn.Sequential(Convolution_Layer(in_channels, in_channels // 2, 1),
                                       Convolution_Layer(in_channels // 2, in_channels, 3, 1, 1))

    def forward(self, x):
        return self.res_layer(x) + x


class UpSampling_Layer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=2, mode='nearest')


class ConvolutionSet_Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_set_layer = nn.Sequential(Convolution_Layer(in_channels, out_channels, 1),
                                            Convolution_Layer(out_channels, in_channels, 3, 1, 1),
                                            Convolution_Layer(in_channels, out_channels, 1),
                                            Convolution_Layer(out_channels, in_channels, 3, 1, 1),
                                            Convolution_Layer(in_channels, out_channels, 1))

    def forward(self, x):
        return self.conv_set_layer(x)


class MainNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.outsize52_layer = nn.Sequential(Convolution_Layer(3, 32, 3, 1, 1),
                                             Convolution_Layer(32, 64, 3, 2, 1),
                                             Residual_Layer(64),
                                             Convolution_Layer(64, 128, 3, 2, 1),
                                             Residual_Layer(128),
                                             Residual_Layer(128),
                                             Convolution_Layer(128, 256, 3, 2, 1),
                                             Residual_Layer(256),
                                             Residual_Layer(256),
                                             Residual_Layer(256),
                                             Residual_Layer(256),
                                             Residual_Layer(256),
                                             Residual_Layer(256),
                                             Residual_Layer(256),
                                             Residual_Layer(256))

        self.outsize26_layer = nn.Sequential(Convolution_Layer(256, 512, 3, 2, 1),
                                             Residual_Layer(512),
                                             Residual_Layer(512),
                                             Residual_Layer(512),
                                             Residual_Layer(512),
                                             Residual_Layer(512),
                                             Residual_Layer(512),
                                             Residual_Layer(512),
                                             Residual_Layer(512))

        self.convset1_layer = nn.Sequential(Convolution_Layer(512, 1024, 3, 2, 1),
                                            Residual_Layer(1024),
                                            Residual_Layer(1024),
                                            Residual_Layer(1024),
                                            Residual_Layer(1024),
                                            ConvolutionSet_Layer(1024, 512))

        self.predict1_layer = nn.Sequential(Convolution_Layer(512, 1024, 3, 1, 1),
                                            nn.Conv2d(1024, 225, 1))

        self.up1_layer = nn.Sequential(Convolution_Layer(512, 256, 1),
                                       UpSampling_Layer())

        self.convset2_layer = nn.Sequential(ConvolutionSet_Layer(768, 256))

        self.predict2_layer = nn.Sequential(Convolution_Layer(256, 512, 3, 1, 1),
                                            nn.Conv2d(512, 225, 1))

        self.up2_layer = nn.Sequential(Convolution_Layer(256, 128, 1),
                                       UpSampling_Layer())

        self.predict3_layer = nn.Sequential(ConvolutionSet_Layer(384, 128),
                                            Convolution_Layer(128, 256, 3, 1, 1),
                                            nn.Conv2d(256, 225, 1))

    def forward(self, x):
        h_52 = self.outsize52_layer(x)
        h_26 = self.outsize26_layer(h_52)
        convset1_out = self.convset1_layer(h_26)
        up1_out = self.up1_layer(convset1_out)
        route1_out = torch.cat((up1_out, h_26), dim=1)
        convset2_out = self.convset2_layer(route1_out)
        up2_out = self.up2_layer(convset2_out)
        route2_out = torch.cat((up2_out, h_52), dim=1)
        predict1_out = self.predict1_layer(convset1_out)
        predict2_out = self.predict2_layer(convset2_out)
        predict3_out = self.predict3_layer(route2_out)
        return predict1_out, predict2_out, predict3_out


if __name__ == '__main__':
    yolov3net = MainNet()
    input = torch.Tensor(2, 3, 416, 416)

    p1, p2, p3 = yolov3net(input)
    print(p1.shape)
    print(p2.shape)
    print(p3.shape)
