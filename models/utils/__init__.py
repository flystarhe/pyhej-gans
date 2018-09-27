import os
import codecs
import shutil
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.utils as vutils


def make_dir(target_dir, mode="755", rm=False):
    xpath = Path()
    for part in Path(target_dir).parts:
        xpath /= part
        if not xpath.exists():
            os.system("mkdir -m {} {}".format(mode, str(xpath)))

    if rm:
        shutil.rmtree(str(xpath))
        os.system("mkdir -m {} {}".format(mode, str(xpath)))

    return str(xpath)


def make_parent(target_path, mode="755", rm=False):
    target_dir = os.path.dirname(target_path)
    return make_dir(target_dir, mode, rm)


def print_network(model, name):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print(name)
    print("The number of parameters: [{}]".format(num_params))


def plot_images(tensor):
    # tensor (Tensor or list): 4D mini-batch Tensor of shape (BxCxHxW)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(tensor, padding=2, normalize=True, scale_each=True).cpu(), (1, 2, 0)))


class Logger(object):
    def __init__(self, root):
        self.full_path = os.path.join(root, "log.loss")
        self.data = {}

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self.data.setdefault(k, []).append(v)

    def save(self, curr_iters):
        with codecs.open(self.full_path, "a", "utf-8") as writer:
            message = ["{}:{:.4f}".format(k, np.mean(self.data[k])) for k in sorted(self.data.keys())]
            writer.write("{}#{}\n".format(curr_iters, ",".join(message)))
        self.data = {}
