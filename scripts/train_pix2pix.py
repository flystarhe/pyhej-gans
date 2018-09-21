import argparse


def get_options(**kwargs):
    opts = {
        "input_nc": 1,
        "output_nc": 1,
        "gpu_ids": [],
        "isTrain": False,
        "batchSize": 1,
        "ngf": 64,
        "ndf": 64,
        "norm": "instance",
        "dropout": False,
        "init_type": "",
        "n_layers_D": 3,
        "lsgan": True,
        "lr": 2e-4,
        "save_dir": "tmps/task_name",
        "verbose": False,
        "which_epoch": "none",
    }
    for k, v in kwargs.items():
        opts[k] = kwargs[k]
    return argparse.Namespace(**opts)
