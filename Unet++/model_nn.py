import torch.nn as nn
import torch

dropout_rate = 0.5


class standard_unit(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size, padding=padding),
                                  # nn.BatchNorm2d(outchannel),
                                  nn.PReLU(),
                                  nn.Dropout2d(dropout_rate),
                                  nn.Conv2d(outchannel, outchannel, kernel_size, padding=padding),
                                  # nn.BatchNorm2d(outchannel),
                                  nn.PReLU(),
                                  nn.Dropout2d(dropout_rate)
                                  )

    def forward(self, input_tensor):
        out = self.conv(input_tensor)
        return out


class out_unit(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size=1):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size),
                                  nn.Sigmoid())

    def forward(self, x):
        x = self.conv(x)
        return x


########################################

"""
Standard UNet++ [Zhou et.al, 2018]
Total params: 9,041,601
"""


class Nest_Net(nn.Module):
    def __init__(self, img_channel, num_class=1, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        nb_filter = [32, 64, 128, 256, 512]

        self.conv1_1 = standard_unit(img_channel, nb_filter[0])

        self.conv2_1 = nn.Sequential(nn.MaxPool2d(2, 2),
                                     standard_unit(nb_filter[0], nb_filter[1]))

        # 可以尝试线性插值接一个1*1卷积压缩通道
        # up1_2 = F.interpolate(conv2_1, 2, mode='bilinear')

        self.up1_2 = nn.ConvTranspose2d(nb_filter[1], nb_filter[0], 2, 2)

        self.conv1_2 = standard_unit(nb_filter[0] * 2, nb_filter[0])

        self.conv3_1 = nn.Sequential(nn.MaxPool2d(2, 2),
                                     standard_unit(nb_filter[1], nb_filter[2]))

        self.up2_2 = nn.ConvTranspose2d(nb_filter[2], nb_filter[1], 2, 2)

        self.conv2_2 = standard_unit(nb_filter[1] * 2, nb_filter[1])

        self.up1_3 = nn.ConvTranspose2d(nb_filter[1], nb_filter[0], 2, 2)

        self.conv1_3 = standard_unit(nb_filter[0] * 3, nb_filter[0])

        self.conv4_1 = nn.Sequential(nn.MaxPool2d(2, 2),
                                     standard_unit(nb_filter[2], nb_filter[3]))

        self.up3_2 = nn.ConvTranspose2d(nb_filter[3], nb_filter[2], 2, 2)

        self.conv3_2 = standard_unit(nb_filter[2] * 2, nb_filter[2])

        self.up2_3 = nn.ConvTranspose2d(nb_filter[2], nb_filter[1], 2, 2)

        self.conv2_3 = standard_unit(nb_filter[1] * 3, nb_filter[1])

        self.up1_4 = nn.ConvTranspose2d(nb_filter[1], nb_filter[0], 2, 2)

        self.conv1_4 = standard_unit(nb_filter[0] * 4, nb_filter[0])

        self.conv5_1 = nn.Sequential(nn.MaxPool2d(2, 2),
                                     standard_unit(nb_filter[3], nb_filter[4]))

        self.up4_2 = nn.ConvTranspose2d(nb_filter[4], nb_filter[3], 2, 2)

        self.conv4_2 = standard_unit(nb_filter[3] * 2, nb_filter[3])

        self.up3_3 = nn.ConvTranspose2d(nb_filter[3], nb_filter[2], 2, 2)

        self.conv3_3 = standard_unit(nb_filter[2] * 3, nb_filter[2])

        self.up2_4 = nn.ConvTranspose2d(nb_filter[2], nb_filter[1], 2, 2)

        self.conv2_4 = standard_unit(nb_filter[1] * 4, nb_filter[1])

        self.up1_5 = nn.ConvTranspose2d(nb_filter[1], nb_filter[0], 2, 2)

        self.conv1_5 = standard_unit(nb_filter[0] * 5, nb_filter[0])

        self.nestnet_output = out_unit(nb_filter[0], num_class)

    def forward(self, img_input):
        conv1_1 = self.conv1_1(img_input)
        conv2_1 = self.conv2_1(conv1_1)
        conv3_1 = self.conv3_1(conv2_1)
        conv4_1 = self.conv4_1(conv3_1)
        conv5_1 = self.conv5_1(conv4_1)

        up1_2 = self.up1_2(conv2_1)
        conv1_2 = torch.cat((up1_2, conv1_1), dim=1)
        conv1_2 = self.conv1_2(conv1_2)

        up2_2 = self.up2_2(conv3_1)
        conv2_2 = torch.cat((up2_2, conv2_1), dim=1)
        conv2_2 = self.conv2_2(conv2_2)

        up1_3 = self.up1_3(conv2_2)
        conv1_3 = torch.cat((up1_3, conv1_1, conv1_2), dim=1)
        conv1_3 = self.conv1_3(conv1_3)

        up3_2 = self.up3_2(conv4_1)
        conv3_2 = torch.cat((up3_2, conv3_1), dim=1)
        conv3_2 = self.conv3_2(conv3_2)

        up2_3 = self.up2_3(conv3_2)
        conv2_3 = torch.cat((up2_3, conv2_1, conv2_2), dim=1)
        conv2_3 = self.conv2_3(conv2_3)

        up1_4 = self.up1_4(conv2_3)
        conv1_4 = torch.cat((up1_4, conv1_1, conv1_2, conv1_3), dim=1)
        conv1_4 = self.conv1_4(conv1_4)

        up4_2 = self.up4_2(conv5_1)
        conv4_2 = torch.cat((up4_2, conv4_1), dim=1)
        conv4_2 = self.conv4_2(conv4_2)

        up3_3 = self.up3_3(conv4_2)
        conv3_3 = torch.cat((up3_3, conv3_1, conv3_2), dim=1)
        conv3_3 = self.conv3_3(conv3_3)

        up2_4 = self.up2_4(conv3_3)
        conv2_4 = torch.cat((up2_4, conv2_1, conv2_2, conv2_3), dim=1)
        conv2_4 = self.conv2_4(conv2_4)

        up1_5 = self.up1_5(conv2_4)
        conv1_5 = torch.cat((up1_5, conv1_1, conv1_2, conv1_3, conv1_4), dim=1)
        conv1_5 = self.conv1_5(conv1_5)

        nestnet_output1 = self.nestnet_output(conv1_2)
        nestnet_output2 = self.nestnet_output(conv1_3)
        nestnet_output3 = self.nestnet_output(conv1_4)
        nestnet_output4 = self.nestnet_output(conv1_5)

        if self.deep_supervision:
            return nestnet_output1, nestnet_output2, nestnet_output3, nestnet_output4

        else:
            return nestnet_output4


if __name__ == '__main__':
    net = Nest_Net(1)
    x = torch.randn((3, 1, 512, 512))
    print(net(x).size())
