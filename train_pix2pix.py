import os
import sys
import time
import torch
from torch.nn import init
from torch.backends import cudnn
from torch.optim import lr_scheduler
from models.options.base_junyanz import BaseOptions
from models.discriminator.base_junyanz import Discriminator
from models.generator.resnet_junyanz import Generator
from models.utils import make_dir, print_network
from models.utils import Logger
from models.data.dicom_ct import Dataset
from models.data import get_loader
from models.loss import GANLoss


def init_weights(net, init_type="normal"):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, 0.02)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            else:
                raise NotImplementedError("initialization method [{}] is not implemented".format(init_type))
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm") != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with [{}]".format(init_type))
    net.apply(init_func)


def get_scheduler(optimizer, opt):
    return lr_scheduler.StepLR(optimizer, step_size=opt.lr_update_step, gamma=opt.lr_update_gamma)


def load_net(net, iters, name, opt, device):
    file_name = os.path.join(opt.checkpoints_dir, "{:8d}_net_{}.pth".format(iters, name))
    net.load_state_dict(torch.load(file_name, map_location=device))
    net.to(device)


def save_net(net, iters, name, opt):
    file_name = os.path.join(opt.checkpoints_dir, "{:8d}_net_{}.pth".format(iters, name))
    if isinstance(net, torch.nn.DataParallel):
        torch.save(net.module.state_dict(), file_name)
    else:
        torch.save(net.state_dict(), file_name)


def set_requires_grad(net, requires_grad=False):
    for param in net.parameters():
        param.requires_grad = requires_grad


def main(args):
    opt = BaseOptions.parse(args)
    make_dir(opt.checkpoints_dir)
    BaseOptions.print_options(opt)

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:{}".format(opt.gpu_ids[0]) if opt.gpu_ids else "cpu")

    net_D = Discriminator(opt.input_nc, opt.conv_dim_d, opt.n_layers_d, opt.use_sigmoid).to(device)
    net_G = Generator(opt.input_nc, opt.conv_dim_g, opt.n_blocks_g, opt.use_bias).to(device)

    if opt.resume_iters:
        load_net(net_D, opt.resume_iters, "D", device)
        load_net(net_G, opt.resume_iters, "G", device)
    else:
        init_weights(net_D, opt.init_type)
        init_weights(net_G, opt.init_type)

    if len(opt.gpu_ids) > 1:
        net_D = torch.nn.DataParallel(net_D, device_ids=opt.gpu_ids)
        net_G = torch.nn.DataParallel(net_G, device_ids=opt.gpu_ids)

    print_network(net_D, "net_D")
    print_network(net_G, "net_G")

    optimizer_D = torch.optim.Adam(net_D.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_G = torch.optim.Adam(net_G.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizers = [optimizer_D, optimizer_G]

    schedulers = [get_scheduler(optimizer, opt) for optimizer in optimizers]

    criterionGAN = GANLoss(no_lsgan=True).to(device)
    criterionL1 = torch.nn.L1Loss()

    dataset = Dataset(opt.json_file, opt.aligned)
    print("#training images = {}".format(len(dataset)))
    data_loader = get_loader(dataset, opt.batch_size, True, opt.workers)

    data_time = 0.0
    total_time = 0.0
    data_iter = iter(data_loader)
    logger = Logger(opt.checkpoints_dir)
    for curr_iters in range(opt.start_iters, opt.start_iters + opt.train_iters):
        start_time = time.time()

        try:
            real_A, real_B = next(data_iter)
        except Exception:
            data_iter = iter(data_loader)
            real_A, real_B = next(data_iter)

        real_A = real_A.to(device)
        real_B = real_B.to(device)

        data_time += time.time() - start_time

        fake_B = net_G(real_A)

        # update D
        set_requires_grad(net_D, True)
        optimizer_D.zero_grad()

        pred_fake = net_D(fake_B.detach())
        loss_D_fake = criterionGAN(pred_fake, False)

        pred_real = net_D(real_B)
        loss_D_real = criterionGAN(pred_real, True)

        loss_D = loss_D_fake + loss_D_real
        loss_D.backward()
        optimizer_D.step()

        # update G
        set_requires_grad(net_D, False)
        optimizer_G.zero_grad()

        pred_fake = net_D(fake_B)
        loss_G_GAN = criterionGAN(pred_fake, True)

        loss_G_L1 = criterionL1(fake_B, real_B)

        loss_G = loss_G_GAN + loss_G_L1 * 100
        loss_G.backward()
        optimizer_G.step()

        total_time += time.time() - start_time

        logger.add(loss_D_fake=loss_D_fake.mean().item(), loss_D_real=loss_D_real.mean().item(),
                   loss_G_GAN=loss_G_GAN.mean().item(), loss_G_L1=loss_G_L1.mean().item())

        for scheduler in schedulers:
            scheduler.step()

        if curr_iters % opt.model_save == 0:
            print("saving the model: iters = {}, lr = {}".format(curr_iters, optimizers[0].param_groups[0]["lr"]))
            save_net(net_D, curr_iters, "D", opt)
            save_net(net_G, curr_iters, "G", opt)

        if curr_iters % opt.display_freq == 0:
            print("#iters[{}]: data time {}, total time {}".format(curr_iters, data_time, total_time))
            data_time = 0.0
            total_time = 0.0
            logger.save(curr_iters)


if __name__ == "__main__":
    main(sys.argv[1:])
