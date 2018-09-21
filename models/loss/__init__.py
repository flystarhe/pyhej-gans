import torch
import torch.nn as nn


class GANLoss(nn.Module):
    def __init__(self, no_lsgan=True, real_label=1.0, fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(real_label))
        self.register_buffer("fake_label", torch.tensor(fake_label))
        if no_lsgan:
            self.loss = nn.BCELoss()
        else:
            self.loss = nn.MSELoss()

    def get_target_tensor(self, x, is_real):
        if is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(x)

    def __call__(self, x, is_real):
        target_tensor = self.get_target_tensor(x, is_real)
        return self.loss(x, target_tensor)

    def forward(self, x):
        return None
