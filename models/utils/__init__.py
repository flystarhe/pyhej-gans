import os
import shutil
from pathlib import Path


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
