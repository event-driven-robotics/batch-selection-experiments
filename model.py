import torch.nn as nn

from torch import tanh, sigmoid
from torch.nn.functional import relu


class EV_RotNet(nn.Module):
    def __init__(self, num_bins=9):
        super(EV_RotNet, self).__init__()

        def en_conv(i, o):
            return nn.Conv2d(i, o, kernel_size=3, stride=2, padding=1)

        def res_conv(i):
            return nn.Conv2d(i, i, kernel_size=3, stride=1, padding=1)

        self.en_conv1 = en_conv(num_bins, 64)
        self.en_conv2 = en_conv(64, 128)
        self.en_conv3 = en_conv(128, 256)
        self.en_conv4 = en_conv(256, 512)

        self.res_conv1 = res_conv(512)
        self.res_conv2 = res_conv(512)

        self.gm_conv1 = en_conv(512, 256)
        self.gm_conv2 = en_conv(256, 64)
        self.gm_conv3 = en_conv(64, 16)
        self.gm_conv4 = en_conv(16, 3)

    def forward(self, x):
        out = relu(
            self.en_conv4(
                relu(self.en_conv3(relu(self.en_conv2(relu(self.en_conv1(x))))))
            )
        )

        res = relu(self.res_conv2(relu(self.res_conv1(out))))
        out = res + out

        out = self.gm_conv4(
            relu(self.gm_conv3(relu(self.gm_conv2(relu(self.gm_conv1(out))))))
        )[:, :, 0, 0]

        return out


class EV_Imgs(nn.Module):
    def __init__(self, num_imgs=16):
        super(EV_Imgs, self).__init__()

        def en_conv(i, o):
            return nn.Conv2d(i, o, kernel_size=6, stride=3, padding=0)

        self.en_conv1 = en_conv(num_imgs, 8)
        self.en_conv2 = en_conv(8, 4)
        self.en_conv3 = en_conv(4, 2)
        self.en_conv4 = en_conv(2, 1)

    def forward(self, x):
        out = sigmoid(
            self.en_conv4(
                relu(self.en_conv3(relu(self.en_conv2(relu(self.en_conv1(x))))))
            )
        )

        return out * 400000

model = EV_Imgs()
#model(t).shape

model = EV_RotNet()