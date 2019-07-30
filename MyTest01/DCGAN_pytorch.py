import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class D_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.comp_conv = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1), nn.LeakyReLU(),
                                       nn.Conv2d(64, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(),
                                       nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.LeakyReLU(),
                                       nn.Conv2d(128, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(),
                                       nn.Conv2d(128, 1, 7), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, x):
        net = self.comp_conv(x)
        return net


class G_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.liner = nn.Linear(128, 7 * 7 * 128)
        self.comp_deconv = nn.Sequential(nn.ConvTranspose2d(128, 128, 3, 2, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
                                         nn.ConvTranspose2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
                                         nn.ConvTranspose2d(64, 64, 3, 2, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
                                         nn.ConvTranspose2d(64, 1, 3, 1, 1), nn.Tanh())

    def forward(self, x):
        net = self.liner(x)
        net = net.view(-1, 128, 7, 7)
        net = self.comp_deconv(net)
        return net


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.02)
    elif isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, std=0.02)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, std=0.02)


class Net:
    def __init__(self, isCuda=True):
        self.d_net = D_Net()
        self.g_net = G_Net()
        self.isCuda = isCuda
        if self.isCuda:
            self.d_net.cuda()
            self.g_net.cuda()
        self.d_net.apply(weight_init)
        self.g_net.apply(weight_init)

        self.loss_fn = nn.BCELoss()
        self.d_optimizer = optim.Adam(self.d_net.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.g_optimizer = optim.Adam(self.g_net.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def forward(self, data_input, random_input):
        if self.isCuda:
            data_input = data_input.cuda()
            random_input = random_input.cuda()
        self.d_real_out = self.d_net(data_input)
        self.g_out = self.g_net(random_input)
        self.d_fake_out = self.d_net(self.g_out.detach())
        self.g_fake_out = self.d_net(self.g_out)

    def D_loss(self, d_real_label, d_fake_label):
        if self.isCuda:
            d_real_label = d_real_label.cuda()
            d_fake_label = d_fake_label.cuda()
        d_real_loss = self.loss_fn(self.d_real_out, d_real_label)
        d_fake_loss = self.loss_fn(self.d_fake_out, d_fake_label)
        self.d_loss = d_real_loss + d_fake_loss
        return self.d_loss

    def G_loss(self, g_fake_label):
        if self.isCuda:
            g_fake_label = g_fake_label.cuda()
        self.g_loss = self.loss_fn(self.g_fake_out, g_fake_label)
        return self.g_loss

    def D_backward(self):
        self.d_optimizer.zero_grad()
        self.d_loss.backward()
        self.d_optimizer.step()

    def G_backward(self):
        self.g_optimizer.zero_grad()
        self.g_loss.backward()
        self.g_optimizer.step()


if __name__ == '__main__':
    dataset = datasets.MNIST(root="./minist_torch/", train=True, transform=transforms.ToTensor(), download=True)
    dataloader = DataLoader(dataset=dataset, batch_size=100, shuffle=True, num_workers=5)
    net = Net()
    for _ in range(100000):
        for i, (xs, _) in enumerate(dataloader):
            data_input = (xs - 0.5) * 2
            random_input = torch.rand((100, 128))
            random_input = (random_input - 0.5) * 2
            d_real_label = torch.ones((100, 1, 1, 1))
            d_fake_label = torch.zeros((100, 1, 1, 1))
            g_fake_label = torch.ones((100, 1, 1, 1))
            net.forward(data_input, random_input)
            d_loss_ = net.D_loss(d_real_label, d_fake_label)
            net.D_backward()
            print("D_Loss: ", d_loss_.cpu().detach().numpy())
            random_input = torch.rand((100, 128))
            random_input = (random_input - 0.5) * 2
            net.forward(data_input, random_input)
            g_loss_ = net.G_loss(g_fake_label)
            net.G_backward()
            print("G_Loss: ", g_loss_.cpu().detach().numpy())
            if i % 10 == 0:
                random_input = torch.rand((100, 128))
                random_input = (random_input - 0.5) * 2
                net.forward(data_input, random_input)
                g_out_ = net.g_out.cpu().detach().numpy()
                img_array = g_out_[0].reshape([28, 28]) / 2 + 0.5
                plt.imshow(img_array)
                plt.pause(0.1)
