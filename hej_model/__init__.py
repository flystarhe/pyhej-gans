import os
import torch
from torch.optim import lr_scheduler


def get_scheduler(optimizer, lr_update_step, lr_update_gamma):
    return lr_scheduler.StepLR(optimizer, step_size=lr_update_step, gamma=lr_update_gamma)


def load_net(net, iters, checkpoints_dir, device):
    file_name = os.path.join(checkpoints_dir, "net_{:08d}.pth".format(iters))
    net.load_state_dict(torch.load(file_name, map_location=device))
    net.to(device)


def save_net(net, iters, checkpoints_dir):
    file_name = os.path.join(checkpoints_dir, "net_{:08d}.pth".format(iters))
    if isinstance(net, torch.nn.DataParallel):
        torch.save(net.module.state_dict(), file_name)
    else:
        torch.save(net.state_dict(), file_name)