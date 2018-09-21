import torch.nn as nn
import numpy as np


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, input_nc=3, conv_dim=64, repeat_num=6):
        super(Discriminator, self).__init__()

        layers = list()
        layers.append(nn.Conv2d(input_nc, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            next_dim = curr_dim * 2
            layers.append(nn.Conv2d(curr_dim, next_dim, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.1))
            curr_dim = next_dim

        self.main = nn.Sequential(*layers)
        self.final = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        h = self.main(x)
        y = self.final(h)
        return y


class Classifier(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, image_size=512, input_nc=3, conv_dim=64, repeat_num=6, class_num=2):
        super(Classifier, self).__init__()

        layers = list()
        layers.append(nn.Conv2d(input_nc, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            next_dim = curr_dim * 2
            layers.append(nn.Conv2d(curr_dim, next_dim, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.1))
            curr_dim = next_dim

        self.main = nn.Sequential(*layers)
        self.final = nn.Linear(int(np.power(image_size / np.power(2, repeat_num), 2)), class_num)

    def forward(self, x):
        h = self.main(x)
        y = self.final(h.view(h.size(0), -1))
        return y