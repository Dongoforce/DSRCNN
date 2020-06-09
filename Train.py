import argparse
import os
import numpy as np
import math
import itertools
import sys
import glob
import copy

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.models import vgg19
from torch.utils.data import Dataset
from PIL import Image

import torch.nn as nn
import torch

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def th_flatten(a):
    return a.contiguous().view(a.nelement())


def th_repeat(a, repeats, axis=0):
    assert len(a.size()) == 1
    return th_flatten(torch.transpose(a.repeat(repeats, 1), 0, 1))


def np_repeat_2d(a, repeats):

    assert len(a.shape) == 2
    a = np.expand_dims(a, 0)
    a = np.tile(a, [repeats, 1, 1])
    return a


def th_batch_map_coordinates(input, coords, order=1):

    batch_size = input.size(0)
    input_height = input.size(1)
    input_width = input.size(2)

    n_coords = coords.size(1)

    # coords = torch.clamp(coords, 0, input_size - 1)

    coords = torch.cat((torch.clamp(coords.narrow(2, 0, 1), 0, input_height - 1),
                        torch.clamp(coords.narrow(2, 1, 1), 0, input_width - 1)), 2)

    assert (coords.size(1) == n_coords)

    coords_lt = coords.floor().long()
    coords_rb = coords.ceil().long()
    coords_lb = torch.stack([coords_lt[..., 0], coords_rb[..., 1]], 2)
    coords_rt = torch.stack([coords_rb[..., 0], coords_lt[..., 1]], 2)
    idx = th_repeat(torch.arange(0, batch_size), n_coords).long()
    idx = Variable(idx, requires_grad=False)
    if input.is_cuda:
        idx = idx.cuda()

    def _get_vals_by_coords(input, coords):
        indices = torch.stack([
            idx, th_flatten(coords[..., 0]), th_flatten(coords[..., 1])
        ], 1)
        inds = indices[:, 0] * input.size(1) * input.size(2) + indices[:, 1] * input.size(2) + indices[:, 2]
        vals = th_flatten(input).index_select(0, inds)
        vals = vals.view(batch_size, n_coords)
        return vals

    vals_lt = _get_vals_by_coords(input, coords_lt.detach())
    vals_rb = _get_vals_by_coords(input, coords_rb.detach())
    vals_lb = _get_vals_by_coords(input, coords_lb.detach())
    vals_rt = _get_vals_by_coords(input, coords_rt.detach())

    coords_offset_lt = coords - coords_lt.type(coords.data.type())
    vals_t = coords_offset_lt[..., 0] * (vals_rt - vals_lt) + vals_lt
    vals_b = coords_offset_lt[..., 0] * (vals_rb - vals_lb) + vals_lb
    mapped_vals = coords_offset_lt[..., 1] * (vals_b - vals_t) + vals_t
    return mapped_vals


def th_generate_grid(batch_size, input_height, input_width, dtype, cuda):
    grid = np.meshgrid(
        range(input_height), range(input_width), indexing='ij'
    )
    grid = np.stack(grid, axis=-1)
    grid = grid.reshape(-1, 2)

    grid = np_repeat_2d(grid, batch_size)
    grid = torch.from_numpy(grid).type(dtype)
    if cuda:
        grid = grid.cuda()
    return Variable(grid, requires_grad=False)


def th_batch_map_offsets(input, offsets, grid=None, order=1):

    batch_size = input.size(0)
    input_height = input.size(1)
    input_width = input.size(2)

    offsets = offsets.view(batch_size, -1, 2)
    if grid is None:
        grid = th_generate_grid(batch_size, input_height, input_width, offsets.data.type(), offsets.data.is_cuda)

    coords = offsets + grid

    mapped_vals = th_batch_map_coordinates(input, coords)
    return mapped_vals


class ConvOffset2D(nn.Conv2d):

    def __init__(self, filters, init_normal_stddev=0, **kwargs):

        self.filters = filters
        self._grid_param = None
        super(ConvOffset2D, self).__init__(self.filters, self.filters * 2, 3, padding=1, bias=False, **kwargs)
        self.weight.data.copy_(self._init_weights(self.weight, init_normal_stddev))

    def forward(self, x):
        x_shape = x.size()
        offsets = super(ConvOffset2D, self).forward(x)

        # offsets: (b*c, h, w, 2)
        offsets = self._to_bc_h_w_2(offsets, x_shape)

        # x: (b*c, h, w)
        x = self._to_bc_h_w(x, x_shape)

        # X_offset: (b*c, h, w)
        x_offset = th_batch_map_offsets(x, offsets, grid=self._get_grid(self, x))

        # x_offset: (b, h, w, c)
        x_offset = self._to_b_c_h_w(x_offset, x_shape)

        return x_offset

    @staticmethod
    def _get_grid(self, x):
        batch_size, input_height, input_width = x.size(0), x.size(1), x.size(2)
        dtype, cuda = x.data.type(), x.data.is_cuda
        if self._grid_param == (batch_size, input_height, input_width, dtype, cuda):
            return self._grid
        self._grid_param = (batch_size, input_height, input_width, dtype, cuda)
        self._grid = th_generate_grid(batch_size, input_height, input_width, dtype, cuda)
        return self._grid

    @staticmethod
    def _init_weights(weights, std):
        fan_out = weights.size(0)
        fan_in = weights.size(1) * weights.size(2) * weights.size(3)
        w = np.random.normal(0.0, std, (fan_out, fan_in))
        return torch.from_numpy(w.reshape(weights.size()))

    @staticmethod
    def _to_bc_h_w_2(x, x_shape):
        """(b, 2c, h, w) -> (b*c, h, w, 2)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]), 2)
        return x

    @staticmethod
    def _to_bc_h_w(x, x_shape):
        """(b, c, h, w) -> (b*c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))
        return x

    @staticmethod
    def _to_b_c_h_w(x, x_shape):
        """(b*c, h, w) -> (b, c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[1]), int(x_shape[2]), int(x_shape[3]))
        return x


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(
            *list(vgg19_model.features.children())[:35])

    def forward(self, img):
        return self.feature_extractor(img)


class ResidualBlock(nn.Module):

    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.conv_block = nn.Sequential(
            #ConvOffset2D(in_features),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
            nn.ReLU(),
            #ConvOffset2D(in_features),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class Net(nn.Module):
    def __init__(self, n_residual_blocks=16):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(), nn.BatchNorm2d(64, 0.8))
        res_blocks = []

        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))

        self.res_blocks = nn.Sequential(*res_blocks)
        # self.ofset = ConvOffset2D(64)
        self.conv2 = nn.Sequential(ConvOffset2D(64),
                                   nn.Conv2d(in_channels=64, out_channels=64, stride=1, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(64, 0.8))

        upsampling = []
        for out_features in range(2):
            upsampling += [
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        self.conv3 = nn.Sequential(ConvOffset2D(64), nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(64, 0.8))
        #self.actleakyrelu = nn.LeakyReLU()
        self.conv4 = nn.Sequential(nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4), nn.Tanh())

    def forward(self, x):
        x1 = self.conv1(x)
        #x = self.act1(x1)
        x = self.res_blocks(x1)
        x2 = self.conv2(x)
        x = torch.add(x1, x2)
        # x = self.conv1(x)
        # x = self.conv2(x)
        x = self.upsampling(x)
        x = self.conv3(x)
        #x = self.actleakyrelu(x)
        x = self.conv4(x)
        return x


class ImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        opt.height, opt.width = hr_shape
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((opt.height // 4, opt.height // 4), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((opt.height, opt.height), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.files = sorted(glob.glob(root + "/*.*"))

    def to_rgb(self, image):
        rgb_image = Image.new("RGB", image.size)
        rgb_image.paste(image)
        return rgb_image

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])

        if img.mode != "RGB":
            img = self.to_rgb(img)

        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    os.makedirs("images/training", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="begining of the cycle")
    parser.add_argument("--n_epochs", type=int, default=10, help="quantity of training cycle")
    parser.add_argument("--n_threads", type=int, default=4, help="num_workers")
    parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--dataset_name", type=str, default="B", help="name of the dataset")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--chanels", type=int, default=3, help="rgb(3) or grayscale(1)")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--images", type=str, default="", help="path to images")
    parser.add_argument("--sample_interval", type=int, default=700, help="interval between saving image samples")
    parser.add_argument("--alpha", type=int, default=5e-3, help="alpha for pixel loss")
    parser.add_argument("--beta", type=int, default=1e-2, help="beta for pixel loss")
    # parser.add_argument("--test_dir", type=str, default= '/storage/BiryukovE/db/classification_data/test_mix', help = "test folder")
    parser.add_argument("--train_dir", type=str, default='img_data/%s', help="train folder")
    parser.add_argument("--model_name", type=str, default="model.pth", help="name for saved model")
    # parser.add_argument("--deviation", type=float, default = 9.0, help = "deviation for yaw, pitch, roll")

    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)
    netcon = Net()
    extractor = FeatureExtractor()
    extractor.eval()

    dir = opt.train_dir
    # files = os.listdir(dir)
    hr_shape = (opt.height, opt.width)
    dataloader = DataLoader(
        ImageDataset("img_data/%s" % opt.dataset_name, hr_shape=hr_shape),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu

    )

    criterion_gen = torch.nn.MSELoss().to(device)
    criterion_content = torch.nn.L1Loss().to(device)
    criterion_pixel = torch.nn.MSELoss().to(device)

    netcon = netcon.to(device)
    extractor = extractor.to(device)

    if opt.epoch != 0:
        netcon.load_state_dict(torch.load("saved_models/Net_%d.pth"))

    optimizer = torch.optim.Adam(netcon.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    for epoch in range(opt.n_epochs):
        for i, imgs in enumerate(dataloader):
            imgs_lr = Variable(imgs["lr"].type(Tensor))
            imgs_hr = Variable(imgs["hr"].type(Tensor))
            print(imgs_lr)

            optimizer.zero_grad()
            gen_hr = netcon(imgs_lr)
            gen_features = extractor(gen_hr)
            real_features = extractor(imgs_hr)
            loss_gen = criterion_gen(gen_features, real_features)
            loss_content = criterion_content(gen_features, real_features.detach())

            loss_pixel = criterion_pixel(gen_hr, imgs_hr)

            loss = loss_content + opt.alpha * loss_gen + opt.beta * loss_pixel
            # loss = loss_content + opt.beta * loss_pixel
            # loss = loss_content
            loss.backward()
            optimizer.step()

            sys.stdout.write(
                "[Epoch %d/%d] [Batch %d/%d] [loss: %f]\n"
                % (epoch, opt.n_epochs, i, len(dataloader), loss.item())
            )

            batches_done = epoch * len(dataloader) + i

            if batches_done % opt.sample_interval == 0:
                imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
                gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
                imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)

                img_grid = torch.cat((imgs_lr, gen_hr), -1)
                save_image(img_grid, "images/%d.png" % batches_done)
    torch.save(optimizer.state_dict(), "saved_models/Net%d.pth" % opt.n_epochs)
