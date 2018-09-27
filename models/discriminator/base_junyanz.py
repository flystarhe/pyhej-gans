import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_nc, conv_dim=64, n_layers=6, use_sigmoid=False):
        super(Discriminator, self).__init__()

        layers = [nn.Conv2d(input_nc, conv_dim, kernel_size=4, stride=2, padding=1),
                  nn.LeakyReLU(0.1, True)]

        curr_dim = conv_dim
        for n in range(1, n_layers):
            next_dim = curr_dim * 2
            layers += [nn.Conv2d(curr_dim, next_dim, kernel_size=4, stride=2, padding=1),
                       nn.InstanceNorm2d(next_dim, affine=True, track_running_stats=True),
                       nn.LeakyReLU(0.1, True)]
            curr_dim = next_dim

        layers += [nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1),
                   nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True),
                   nn.LeakyReLU(0.1, True)]

        layers += [nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)]

        if use_sigmoid:
            layers += [nn.Sigmoid()]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
