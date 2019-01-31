import os
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torchvision


# Assume input range is [-1, 1]
class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=34, use_bn=False, use_input_norm=True, device=torch.device("cpu")):
        super(VGGFeatureExtractor, self).__init__()
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485 * 2 - 1, 0.456 * 2 - 1, 0.406 * 2 - 1]).view(1, 3, 1, 1).to(device)
            # [0.485, 0.456, 0.406] if input in range [0,1]
            std = torch.Tensor([0.229 * 2, 0.224 * 2, 0.225 * 2]).view(1, 3, 1, 1).to(device)
            # [0.229, 0.224, 0.225] if input in range [0,1]
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


# Assume input range is [0, 1]
class ResNet101FeatureExtractor(nn.Module):
    def __init__(self, use_input_norm=True, device=torch.device("cpu")):
        super(ResNet101FeatureExtractor, self).__init__()
        model = torchvision.models.resnet101(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485*2-1, 0.456*2-1, 0.406*2-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)
        self.features = nn.Sequential(*list(model.children())[:8])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


def define_F(gpu_ids, use_bn, device):
    # pytorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    model_f = VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn, use_input_norm=True, device=device)
    # netF = ResNet101FeatureExtractor(use_input_norm=True, device=device)
    if gpu_ids:
        model_f = nn.DataParallel(model_f)
    model_f.eval()  # No need to train
    return model_f


def get_scheduler(optimizer, lr_update_step, lr_update_gamma):
    return lr_scheduler.StepLR(optimizer, step_size=lr_update_step, gamma=lr_update_gamma)


def load_net(net, iters, name, checkpoints_dir, device):
    file_name = os.path.join(checkpoints_dir, "net_{:08d}_{}.pth".format(iters, name))
    net.load_state_dict(torch.load(file_name, map_location=device))
    net.to(device)


def save_net(net, iters, name, checkpoints_dir):
    file_name = os.path.join(checkpoints_dir, "net_{:08d}_{}.pth".format(iters, name))
    if isinstance(net, torch.nn.DataParallel):
        torch.save(net.module.state_dict(), file_name)
    else:
        torch.save(net.state_dict(), file_name)
