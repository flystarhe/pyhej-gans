import os
import sys

mylibs = ["/home/hejian/PycharmProjects/???"]
os.chdir(mylibs[0])
for mylib in mylibs:
    if mylib not in sys.path:
        sys.path.insert(0, mylib)


import time
from hej_utils import Logger
from hej_utils import str2list, str2bool
from hej_utils import print_network, print_options
from hej_model import get_scheduler, load_net, save_net


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="unet3d_8s")
parser.add_argument("--model_n_blocks", type=int, default=1)
parser.add_argument("--model_n_filters", type=int, default=16)
parser.add_argument("--checkpoints_dir", type=str, default="/data1/tmps/tmpl_train")
parser.add_argument("--dataset_train", type=str, default="/data1/tmps/data_train")
parser.add_argument("--dataset_val", type=str, default="/data1/tmps/data_val")
parser.add_argument("--gpu_ids", type=str2list, default="0,")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_worker", type=int, default=1)
parser.add_argument("--resume_iters", type=int, default=-1)
parser.add_argument("--start_iters", type=int, default=1)
parser.add_argument("--train_iters", type=int, default=1)
parser.add_argument("--steps_on_g", type=int, default=1)
parser.add_argument("--steps_on_d", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.0002)
parser.add_argument("--lr_update_step", type=int, default=200)
parser.add_argument("--lr_update_gamma", type=float, default=0.1)
opt, _ = parser.parse_known_args([])
print_options(opt)


logger = Logger(opt.checkpoints_dir)


since = time.time()
data_iter = iter(data_loader)
for curr_iters in range(opt.start_iters, opt.start_iters + opt.train_iters):
    num = 0
    print("-" * 50)
    for inputs, labels_a, labels_b in data_loader:
        num = num % (opt.steps_on_g + opt.steps_on_d)
        if num < opt.steps_on_g:
            logger.add(name=None, ..)
            # by Adam
            pass
        else:
            logger.add(name=None, ..)
            # by SGD
            pass
        num += 1
    print(logger.save(curr_iters))
    save_net(model, curr_iters, opt.checkpoints_dir)
    print("Complete in {:.0f}m {:.0f}s".format(*divmod(time.time() - since, 60)))