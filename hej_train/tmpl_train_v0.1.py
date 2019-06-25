import os
import sys

mylibs = ["/home/hejian/PycharmProjects/pyhej-gans"]
os.chdir(mylibs[0])
for mylib in mylibs:
    if mylib not in sys.path:
        sys.path.insert(0, mylib)

import time
import torch
import argparse
import numpy as np
from hej_utils import Logger
from hej_utils import str2list, str2bool
from hej_utils import print_network, print_options
from hej_model import get_scheduler, load_net, save_net
from hej_model import define_F, unet_8s
from hej_model.loss import GANLoss
from hej_data import get_loader


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_train", type=bool, default=True)
    parser.add_argument("--resume_iters", type=int, default=-1)
    parser.add_argument("--checkpoints_dir", type=str, default="/data1/tmps/train_gans")
    parser.add_argument("--start_iters", type=int, default=1)
    parser.add_argument("--train_iters", type=int, default=1)
    parser.add_argument("--train_steps_on_g", type=int, default=1)
    parser.add_argument("--train_steps_on_d", type=int, default=1)
    parser.add_argument("--dataset_dir", type=str, default="/data1/tmps/dataset/train")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_worker", type=int, default=1)
    parser.add_argument("--input_nc", type=int, default=1)
    parser.add_argument("--model_g", type=str, default="unet_8s")
    parser.add_argument("--model_g_filters", type=int, default=32)
    parser.add_argument("--model_d", type=str, default="unet_8s")
    parser.add_argument("--model_d_filters", type=int, default=32)
    parser.add_argument("--use_dropout", type=str, default=1)
    parser.add_argument("--gpu_ids", type=str2list, default="0,")
    parser.add_argument("--lr_g", type=float, default=0.0002)
    parser.add_argument("--lr_g_beta1", type=float, default=0.5)
    parser.add_argument("--lr_d", type=float, default=0.0002)
    parser.add_argument("--lr_d_momentum", type=float, default=0.9)
    parser.add_argument("--scheduler_step_size", type=int, default=200)
    parser.add_argument("--scheduler_gamma", type=float, default=0.1)
    opt, _ = parser.parse_known_args(args)
    logger = Logger(opt.checkpoints_dir)
    print_options(opt)

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:{}".format(opt.gpu_ids[0]) if opt.gpu_ids else "cpu")

    if opt.model_g == "unet_8s":
        model_g = unet_8s.G(opt.model_n_blocks, opt.model_n_filters).to(device)
    else:
        raise NotImplementedError("model[{}] is not implemented".format(opt.model_name))

    if opt.model_d == "unet_8s":
        model_d = unet_8s.D(opt.model_n_blocks, opt.model_n_filters).to(device)
    else:
        raise NotImplementedError("model[{}] is not implemented".format(opt.model_name))

    if opt.resume_iters > -1:
        load_net(model_g, opt.resume_iters, "g", opt.checkpoints_dir, device)
        load_net(model_d, opt.resume_iters, "d", opt.checkpoints_dir, device)

    if len(opt.gpu_ids) > 1:
        model_g = torch.nn.DataParallel(model_g, device_ids=opt.gpu_ids)
        model_d = torch.nn.DataParallel(model_d, device_ids=opt.gpu_ids)
    print_network(model_g, "Net_G")
    print_network(model_d, "Net_D")

    model_f = define_F(opt.gpu_ids, use_bn=False, device=device)

    criterion_pix = torch.nn.L1Loss().to(device)
    criterion_fea = torch.nn.L1Loss().to(device)
    criterion_gan = GANLoss("vanilla", 1.0, 0.0).to(device)

    optimizer_g = torch.optim.Adam(model_g.parameters(), lr=opt.lr_g, betas=(0.5, 0.999))
    scheduler_g = get_scheduler(optimizer_g, opt.lr_g_step, opt.lr_g_gamma)

    optimizer_d = torch.optim.Adam(model_d.parameters(), lr=opt.lr_d, betas=(0.5, 0.999))
    scheduler_d = get_scheduler(optimizer_d, opt.lr_d_step, opt.lr_d_gamma)

    dataset = opt.dataset_dir
    data_loader = get_loader(dataset, opt.batch_size, True, opt.num_worker)
    data_iter = iter(data_loader)

    since = time.time()
    for curr_iters in range(opt.start_iters, opt.start_iters + opt.train_iters):
        print("-" * 50)
        scheduler_g.step()
        scheduler_d.step()

        for n in range(opt.steps_on_g):
            try:
                real_l, real_h = next(data_iter)
            except Exception:
                data_iter = iter(data_loader)
                real_l, real_h = next(data_iter)

            optimizer_g.zero_grad()
            fake_h = model_g(real_l)
            real_fea = model_f(real_h).detach()
            fake_fea = model_f(fake_h)
            pred_g_fake = model_d(fake_h)

            l_g_pix = criterion_pix(fake_h, real_h)
            l_g_fea = criterion_fea(real_fea, fake_fea)
            l_g_gan = criterion_gan(pred_g_fake, True)

            l_g_total = l_g_pix * 0.01 + l_g_fea * 1.0 + l_g_gan * 0.005
            l_g_total.backward()
            optimizer_g.step()

            logger.add(g_pix=l_g_pix.item(), g_fea=l_g_fea.item(), g_gan=l_g_gan.item(), g_total=l_g_total.item())
        for n in range(opt.steps_on_d):
            try:
                real_l, real_h = next(data_iter)
            except Exception:
                data_iter = iter(data_loader)
                real_l, real_h = next(data_iter)

            optimizer_d.zero_grad()
            fake_h = model_g(real_l)
            pred_d_real = model_d(real_h)
            pred_d_fake = model_d(fake_h.detach())

            l_d_real = criterion_gan(pred_d_real, True)
            l_d_feak = criterion_gan(pred_d_fake, False)

            l_d_total = l_d_real + l_d_feak
            l_d_total.backward()
            optimizer_d.step()

            logger.add(d_real=l_d_real.item(), d_feak=l_d_feak.item(), d_total=l_d_total.item())
        time_elapsed = time.time() - since
        print("Complete at {:.0f}H {:.0f}S".format(*divmod(time_elapsed, 3600)))
        save_net(model_g, curr_iters, "g", opt.checkpoints_dir)
        save_net(model_d, curr_iters, "d", opt.checkpoints_dir)
        message = (optimizer_g.param_groups[0]["lr"], optimizer_d.param_groups[0]["lr"])
        logger.log("!lr_g:{:.8f},lr_d:{:.8f}".format(*message))
        logger.save(curr_iters)
        print(message)
