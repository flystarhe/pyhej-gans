import os
import sys
import json
import argparse


def str2bool(v):
    return v.upper() == "TRUE"


def str2list(v):
    return [int(i) for i in v.split(",") if i.isdigit()]


class BaseOptions(object):
    @staticmethod
    def print_options(opt):
        message = json.dumps(vars(opt), indent=2)
        file_name = os.path.join(opt.checkpoints_dir, "opt.json")
        with open(file_name, "w") as opt_file:
            opt_file.write(message)
        print(message)

    @staticmethod
    def parse(args=None):
        if args is None:
            args = sys.argv[1:]
        parser = argparse.ArgumentParser()
        parser.add_argument("--gpu_ids", type=str2list, default=[], help="0|1|0,1|0,1,2")
        parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints/experiment_name")
        # for dataset
        parser.add_argument("--json_file", type=str, required=True)
        parser.add_argument("--aligned", type=str2bool, default=True)
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--image_size", type=int, default=512)
        parser.add_argument("--input_nc", type=int, default=1)
        parser.add_argument("--label_nc", type=int, default=0)
        parser.add_argument("--workers", type=int, default=4)
        # for discriminator
        parser.add_argument("--conv_dim_d", type=int, default=64)
        parser.add_argument("--n_layers_d", type=int, default=6)
        parser.add_argument("--use_sigmoid", action="store_true")
        # for generator
        parser.add_argument("--conv_dim_g", type=int, default=64)
        parser.add_argument("--n_blocks_g", type=int, default=6)
        parser.add_argument("--use_bias", action="store_true")
        # for training
        parser.add_argument("--init_type", type=str, default="normal")
        parser.add_argument("--resume_iters", type=int, default=None)
        parser.add_argument("--start_iters", type=int, default=1)
        parser.add_argument("--train_iters", type=int, default=200000)
        parser.add_argument("--model_save", type=int, default=10000)
        parser.add_argument("--lr", type=float, default=0.0002)
        parser.add_argument("--lr_update_step", type=int, default=10000)
        parser.add_argument("--lr_update_gamma", type=float, default=0.5)
        parser.add_argument("--display_freq", type=int, default=1000)
        parser.add_argument("--display_ncols", type=int, default=3)

        opt, _ = parser.parse_known_args(args)

        return opt
