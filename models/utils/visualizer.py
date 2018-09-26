import visdom


class Visualizer(object):
    def __init__(self, opt):
        self.viz = visdom.Visdom()
        self.name = opt.checkpoints_dir
        self.ncols = opt.display_ncols
        self.plot_data = None
