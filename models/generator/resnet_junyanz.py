import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, conv_dim, use_bias):
        super(ResBlock, self).__init__()
        conv_block = list()

        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        residual = x
        out = self.conv_block(x)
        out += residual
        return out


class Generator(nn.Module):
    def __init__(self, input_nc, conv_dim=64, n_blocks=6, use_bias=True):
        super(Generator, self).__init__()

        layers = [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, conv_dim, kernel_size=7, stride=1, padding=0, bias=use_bias),
                  nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True),
                  nn.ReLU(True)]

        curr_dim = conv_dim
        for i in range(2):
            next_dim = curr_dim * 2
            layers += [nn.Conv2d(curr_dim, next_dim, kernel_size=4, stride=2, padding=1, bias=use_bias),
                       nn.InstanceNorm2d(next_dim, affine=True, track_running_stats=True),
                       nn.ReLU(True)]
            curr_dim = next_dim

        for i in range(n_blocks):
            layers += [ResBlock(curr_dim, use_bias=use_bias)]

        for i in range(2):
            next_dim = curr_dim // 2
            layers += [nn.ConvTranspose2d(curr_dim, next_dim, kernel_size=4, stride=2, padding=1, bias=use_bias),
                       nn.InstanceNorm2d(next_dim, affine=True, track_running_stats=True),
                       nn.ReLU(True)]
            curr_dim = next_dim

        layers += [nn.ReflectionPad2d(3),
                   nn.Conv2d(curr_dim, input_nc, kernel_size=7, stride=1, padding=0),
                   nn.Tanh()]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
