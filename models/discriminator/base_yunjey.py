import torch.nn as nn


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, image_size=512, input_nc=1, label_nc=2, conv_dim=64, n_layers=6):
        super(Discriminator, self).__init__()

        layers = list()
        layers.append(nn.Conv2d(input_nc, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, n_layers):
            next_dim = curr_dim * 2
            layers.append(nn.Conv2d(curr_dim, next_dim, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = next_dim

        layers.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, label_nc, kernel_size=int(image_size / (2 ** n_layers)), bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), -1)
