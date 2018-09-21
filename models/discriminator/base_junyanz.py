import torch.nn as nn


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, conv_dim=64, n_layers=3, norm_layer=nn.InstanceNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()

        layers = [nn.Conv2d(input_nc, conv_dim, kernel_size=4, stride=2, padding=1),
                  nn.LeakyReLU(0.2, True)]

        curr_dim = conv_dim
        for n in range(1, n_layers):
            next_dim = curr_dim * 2
            layers += [nn.Conv2d(curr_dim, next_dim, kernel_size=4, stride=2, padding=1),
                       norm_layer(next_dim),
                       nn.LeakyReLU(0.2, True)]
            curr_dim = next_dim

        layers += [nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1),
                   norm_layer(curr_dim),
                   nn.LeakyReLU(0.2, True)]

        layers += [nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1)]

        if use_sigmoid:
            layers += [nn.Sigmoid()]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
