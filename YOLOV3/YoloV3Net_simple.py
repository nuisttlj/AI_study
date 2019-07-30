import torch
import torch.nn as nn
import YOLOV3.Cfg as cfg


class Residual_Layer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.res_layer = nn.Sequential(nn.Conv2d(in_channels, in_channels // 2, 1),
                                       nn.BatchNorm2d(in_channels // 2),
                                       nn.PReLU(),
                                       nn.Conv2d(in_channels // 2, in_channels, 3, 1, 1),
                                       nn.BatchNorm2d(in_channels),
                                       nn.PReLU())

    def forward(self, x):
        return self.res_layer(x) + x


def make_layers(cfg, in_channels):
    in_channels = in_channels
    layers = []
    for v in cfg:
        if v == 'U':
            layers += [nn.Upsample(scale_factor=2, mode='nearest')]
        elif v == 'D':
            layers += [nn.Conv2d(in_channels, in_channels * 2, 3, 2, 1), nn.BatchNorm2d(in_channels * 2),
                       nn.PReLU()]
            in_channels = in_channels * 2
        elif v[0] == 'R':
            for _ in range(int(v[1])):
                layers += [Residual_Layer(in_channels)]
        elif v[:2] == 'C1':
            layers += [nn.Conv2d(in_channels, int(v[2:]), 1, 1, 0), nn.BatchNorm2d(int(v[2:])), nn.PReLU()]
            in_channels = int(v[2:])
        elif v[:2] == 'C3':
            layers += [nn.Conv2d(in_channels, int(v[2:]), 3, 1, 1), nn.BatchNorm2d(int(v[2:])), nn.PReLU()]
            in_channels = int(v[2:])
        elif v[0] == 'O':
            layers += [nn.Conv2d(in_channels, int(v[1:]), 1, 1, 0)]
    return nn.Sequential(*layers)


# def extract_shape(x):
#     output = x.reshape(x.size(0), 3, -1, x.size(2), x.size(3))
#     output_bce = output[:, :, :1, :, :]
#     output_mse = output[:, :, 1:5, :, :]
#     output_ce = output[:, :, 5:, :, :]
#     return output_bce, output_mse, output_ce


class YoloV3_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_fms52 = make_layers(cfg.cfg_fms52, cfg.in_channels_fms52)
        self.layers_fms26 = make_layers(cfg.cfg_fms26, cfg.in_channels_fms26)
        self.layers_p1 = make_layers(cfg.cfg_p1, cfg.in_channels_p1)
        self.layers_up1 = make_layers(cfg.cfg_up1, cfg.in_channels_up1)
        self.layers_p2 = make_layers(cfg.cfg_p2, cfg.in_channels_p2)
        self.layers_up2 = make_layers(cfg.cfg_up2, cfg.in_channels_up2)
        self.layers_p3 = make_layers(cfg.cfg_p3, cfg.in_channels_p3)
        # self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        fms_52_out = self.layers_fms52(x)
        fms_26_out = self.layers_fms26(fms_52_out)
        p1_out = self.layers_p1(fms_26_out)
        # p1_out_bce, p1_out_mse, p1_out_ce = extract_shape(p1_out)
        # p1_out_bce = torch.sigmoid(p1_out_bce)
        # p1_out_ce = self.softmax(p1_out_ce)
        # p1_out = torch.cat((p1_out_bce, p1_out_mse, p1_out_ce), dim=2).reshape(p1_out.size(0), -1, p1_out.size(2),
        #                                                                        p1_out.size(3))
        up1_out = self.layers_up1(fms_26_out)
        route1 = torch.cat((up1_out, fms_26_out), dim=1)
        p2_out = self.layers_p2(route1)
        # p2_out_bce, p2_out_mse, p2_out_ce = extract_shape(p2_out)
        # p2_out_bce = torch.sigmoid(p2_out_bce)
        # p2_out_ce = self.softmax(p2_out_ce)
        # p2_out = torch.cat((p2_out_bce, p2_out_mse, p2_out_ce), dim=2).reshape(p2_out.size(0), -1, p2_out.size(2),
        #                                                                        p2_out.size(3))
        up2_out = self.layers_up2(route1)
        route2 = torch.cat((up2_out, fms_52_out), dim=1)
        p3_out = self.layers_p3(route2)
        # p3_out_bce, p3_out_mse, p3_out_ce = extract_shape(p3_out)
        # p3_out_bce = torch.sigmoid(p3_out_bce)
        # p3_out_ce = self.softmax(p3_out_ce)
        # p3_out = torch.cat((p3_out_bce, p3_out_mse, p3_out_ce), dim=2).reshape(p3_out.size(0), -1, p3_out.size(2),
        #                                                                        p3_out.size(3))
        # return p1_out_bce, p1_out_mse, p1_out_ce, p2_out_bce, p2_out_mse, p2_out_ce, p3_out_bce, p3_out_mse, p3_out_ce
        return p1_out, p2_out, p3_out


if __name__ == '__main__':
    yolov3net = YoloV3_Net()
    input = torch.Tensor(2, 3, 416, 416)

    p1, p2, p3 = yolov3net(input)
    print(p1.shape)
    print(p2.shape)
    print(p3.shape)
