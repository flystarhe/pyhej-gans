import os
import torch
from . import networks
from .generator.resnet_junyanz import ResNet
from .discriminator.base_junyanz import NLayerDiscriminator


class Pix2PixModel(object):
    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device("cuda:{}".format(self.gpu_ids[0])) if self.gpu_ids else torch.device("cpu")
        self.save_dir = opt.save_dir
        self.loss_names = ["G_GAN", "G_L1", "D_real", "D_fake"]

        norm_layer = networks.get_norm_layer(opt.norm)

        self.model_names = ["G"]
        self.netG = ResNet(opt.input_nc, opt.output_nc, opt.ngf, opt.use_bias, norm_layer, opt.dropout, opt.n_blocks)

        if self.isTrain:
            self.model_names = ["G", "D"]
            self.netD = NLayerDiscriminator(opt.input_nc, opt.ndf, opt.n_layers_D, opt.use_bias, norm_layer, opt.lsgan)

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, real_A, real_B):
        self.real_A = real_A.to(self.device)
        self.real_B = real_B.to(self.device)

    # make models eval mode during test time
    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                net.eval()

    # used in test time, wrapping `forward()` in `no_grad()`
    def test(self):
        with torch.no_grad():
            self.forward()

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]["lr"]
        print("learning rate = {:.8f}".format(lr))

    # save models to the disk
    def save_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = "{}_net_{}.pth".format(epoch, name)
                save_path = os.path.join(self.save_dir, "models", save_filename)
                net = getattr(self, "net" + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    # load models from the disk
    def load_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = "{}_net_{}.pth".format(epoch, name)
                load_path = os.path.join(self.save_dir, "models", load_filename)
                print("loading the model from {}".format(load_path))
                state_dict = torch.load(load_path, map_location=self.device)
                net = getattr(self, "net" + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                net.load_state_dict(state_dict)

    # print network information
    def print_networks(self, verbose=False):
        print("---------- Networks initialized ----------")
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print("[Network {}] Total number of parameters: {:.3f} M".format(name, num_params / 1e6))
        print("----------------------------------------")

    # set requies_grad=Fasle to avoid computation
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self):
        self.fake_B = self.netG(self.real_A)

    def optimize_parameters(self):
        pass
