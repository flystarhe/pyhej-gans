import os
import json
import time
import codecs
import numpy as np


def str2bool(v):
    return v.upper() == "TRUE"


def str2list(v):
    return [int(i) for i in v.split(",") if i.isdigit()]


def print_network(model, name):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print("{}: parameters={}".format(name, num_params))


def print_options(opt):
    message = json.dumps(vars(opt), indent=2, sort_keys=True)
    os.makedirs(opt.checkpoints_dir, exist_ok=True)
    file_path = os.path.join(opt.checkpoints_dir, "log.opt")
    with codecs.open(file_path, "a", "utf-8") as writer:
        writer.write("# " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        writer.write(message + "\n")
    print(message)


class Logger(object):
    def __init__(self, checkpoints_dir):
        os.makedirs(checkpoints_dir, exist_ok=True)
        self.path = os.path.join(checkpoints_dir, "log.loss")
        self.data = {}

        self.log(time.strftime("%Y-%m-%d %H:%M:%S"))

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self.data.setdefault(k, []).append(v)

    def log(self, message):
        with codecs.open(self.path, "a", "utf-8") as writer:
            writer.write("# " + message + "\n")

    def save(self, curr_iters):
        with codecs.open(self.path, "a", "utf-8") as writer:
            message = ["{}:{:.6f}".format(k, np.mean(self.data[k])) for k in sorted(self.data.keys())]
            message = "{}# {}".format(curr_iters, ",".join(message))
            writer.write(message + "\n")
            self.data = {}
        return message
