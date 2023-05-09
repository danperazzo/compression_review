from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import codecs



class Func1DWrapper(torch.utils.data.Dataset):
    def __init__(self, range, fn, grad_fn=None,
                 sampling_density=100, train_every=10):

        coords = self.get_samples(range, sampling_density)
        self.fn_vals = fn(coords)
        self.train_idx = torch.arange(0, coords.shape[0], train_every).float()

        self.grid = coords
        self.grid.requires_grad_(True)
        self.range = range

    def get_samples(self, range, sampling_density):
        num = int(range[1] - range[0])*sampling_density
        coords = np.linspace(start=range[0], stop=range[1], num=num)
        coords.astype(np.float32)
        coords = torch.Tensor(coords).view(-1, 1)
        return coords

    def get_num_samples(self):
        return self.grid.shape[0]

    def __len__(self):
        return 1

    def __getitem__(self, idx):

        return  self.grid, self.fn_vals


def rect(coords, width=1):
    return torch.where(abs(coords) < width/2, 1.0/width, 0.0)


def gaussian(coords, sigma=1, center=0.5):
    return 1 / (sigma * math.sqrt(2*np.pi)) * torch.exp(-(coords-center)**2 / (2*sigma**2))


def sines1(coords):
    return 0.3 * torch.sin(2*np.pi*8*coords + np.pi/3) + 0.65 * torch.sin(2*np.pi*2*coords + np.pi)


def polynomial_1(coords):
    return .1*((coords+.2)*3)**5 - .2*((coords+.2)*3)**4 + .2*((coords+.2)*3)**3 - .4*((coords+.2)*3)**2 + .1*((coords+.2)*3)


def sinc(coords):
    coords[coords == 0] += 1
    return torch.div(torch.sin(20*coords), 20*coords)


def linear(coords):
    return 1.0 * coords


def xcosx(coords):
    return coords * torch.cos(coords)
