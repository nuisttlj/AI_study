import torch


class MyNet(torch.nn.Module):

    def __init__(self):
        super(MyNet, self).__init__()

        self.fc1 = None
        self.fc2 = None

    def forward(self, x):
        y = self.fc1(x)
        y = self.fc2(y)
        return y


net = MyNet()
opt1 = torch.optim.Adam(net.fc2.parameters())
opt2 = torch.optim.SGD(net.parameters())


