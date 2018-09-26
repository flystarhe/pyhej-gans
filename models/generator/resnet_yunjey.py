import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, conv_dim, use_bias):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(conv_dim, conv_dim, kernel_size=1, stride=1, padding=0, bias=use_bias)
        self.norm1 = nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.norm2 = nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True)
        self.conv3 = nn.Conv2d(conv_dim, conv_dim, kernel_size=1, stride=1, padding=0, bias=use_bias)
        self.norm3 = nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        out += residual
        out = self.relu(out)

        return out


class Generator(nn.Module):
    def __init__(self, input_nc=1, label_nc=2, conv_dim=64, n_blocks=6):
        super(Generator, self).__init__()

        layers = list()
        layers.append(nn.ReflectionPad2d(3))
        layers.append(nn.Conv2d(input_nc + label_nc, conv_dim, kernel_size=7, stride=1, padding=0, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            next_dim = curr_dim * 2
            layers.append(nn.Conv2d(curr_dim, next_dim, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(next_dim, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = next_dim

        # Bottleneck layers.
        for i in range(n_blocks):
            layers.append(ResBlock(conv_dim=curr_dim, use_bias=False))

        # Up-sampling layers.
        for i in range(2):
            next_dim = curr_dim // 2
            layers.append(nn.ConvTranspose2d(curr_dim, next_dim, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(next_dim, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = next_dim

        layers.append(nn.ReflectionPad2d(3))
        layers.append(nn.Conv2d(curr_dim, input_nc, kernel_size=7, stride=1, padding=0, bias=False))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
