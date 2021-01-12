import numpy as np
import torch
import torch.distributions as td
import torch.nn.functional as F 
import torch.nn as nn


class ObservationEncoder(nn.Module):
    def __init__(self, depth=32, stride=2, shape=(3, 64, 64), activation=nn.ReLU):
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(shape[0], 1 * depth, 4, stride),
            activation(),
            nn.Conv2d(1 * depth, 2 * depth, 4, stride),
            activation(),
            nn.Conv2d(2 * depth, 4 * depth, 4, stride),
            activation(),
            nn.Conv2d(4 * depth, 8 * depth, 4, stride),
            activation(),
        )
        self.shape = shape
        self.stride = stride
        self.depth = depth

    def forward(self, obs):
        batch_shape = obs.shape[:-3]
        img_shape = obs.shape[-3:]
        embed = self.convolutions(obs.reshape(-1, *img_shape))
        embed = torch.reshape(embed, (*batch_shape, -1))
        return embed

    @property
    def embed_size(self):
        conv1_shape = conv_out_shape(self.shape[1:], 0, 4, self.stride)
        conv2_shape = conv_out_shape(conv1_shape, 0, 4, self.stride)
        conv3_shape = conv_out_shape(conv2_shape, 0, 4, self.stride)
        conv4_shape = conv_out_shape(conv3_shape, 0, 4, self.stride)
        embed_size = 8 * self.depth * np.prod(conv4_shape).item()
        return embed_size


class ObservationEncoder2(nn.Module):
    def __init__(self, depth=32, stride=1, shape=(1, 4, 200), activation=nn.ReLU):
        super().__init__()
        n_lanes = 4
        phase_dim = 60
        self.out_size = 128 + phase_dim

        self.conv1 = nn.Conv2d(1, 128, [1, 200], stride=1)
        self.fc1 = nn.Linear(128 * 4, 128)
        self.fc2 = nn.Linear(128 + phase_dim, 128 + phase_dim)

    def forward(self, obs, phase):  # [50, 1, 4, 200]
        img_shape = obs.shape[-3:]
        phase_shape = phase.shape[-1:]
        batch_shape = obs.shape[:-3]

        obs = obs.reshape(-1, *img_shape)
        phase = phase.reshape(-1, *phase_shape)

        x = F.relu(self.conv1(obs))  # [50, 128, 4, 1]
        x = x.view(x.size(0), -1)  # [50, 512]
        x = F.relu(self.fc1(x))  # [50, 128]

        # phase = phase.view(phase.size(0), -1).float()  # [50, 60]
        x = torch.cat((x, phase), 1)  # [50, 188]
        x = F.relu(self.fc2(x))  # [50, 188]

        embed = torch.reshape(x, (*batch_shape, -1))
        return embed

    @property
    def embed_size(self):
        return self.out_size


class ObservationDecoder(nn.Module):
    def __init__(self, depth=32, stride=2, activation=nn.ReLU, embed_size=1024, shape=(1, 4, 200)):
        super().__init__()
        self.depth = depth
        self.shape = shape

        c, h, w = shape
        conv1_kernel_size = 6
        conv2_kernel_size = 6
        conv3_kernel_size = 5
        conv4_kernel_size = 5
        padding = 0
        conv1_shape = conv_out_shape((h, w), padding, conv1_kernel_size, stride)
        conv1_pad = output_padding_shape((h, w), conv1_shape, padding, conv1_kernel_size, stride)
        conv2_shape = conv_out_shape(conv1_shape, padding, conv2_kernel_size, stride)
        conv2_pad = output_padding_shape(conv1_shape, conv2_shape, padding, conv2_kernel_size, stride)
        conv3_shape = conv_out_shape(conv2_shape, padding, conv3_kernel_size, stride)
        conv3_pad = output_padding_shape(conv2_shape, conv3_shape, padding, conv3_kernel_size, stride)
        conv4_shape = conv_out_shape(conv3_shape, padding, conv4_kernel_size, stride)
        conv4_pad = output_padding_shape(conv3_shape, conv4_shape, padding, conv4_kernel_size, stride)
        self.conv_shape = (32 * depth, *conv4_shape)
        self.linear = nn.Linear(embed_size, 32 * depth * np.prod(conv4_shape).item())
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32 * depth, 4 * depth, conv4_kernel_size, stride, output_padding=conv4_pad),
            activation(),
            nn.ConvTranspose2d(4 * depth, 2 * depth, conv3_kernel_size, stride, output_padding=conv3_pad),
            activation(),
            nn.ConvTranspose2d(2 * depth, 1 * depth, conv2_kernel_size, stride, output_padding=conv2_pad),
            activation(),
            nn.ConvTranspose2d(1 * depth, shape[0], conv1_kernel_size, stride, output_padding=conv1_pad),
        )

    def forward(self, x):
        """
        :param x: size(*batch_shape, embed_size)
        :return: obs_dist = size(*batch_shape, *self.shape)
        """
        batch_shape = x.shape[:-1]
        embed_size = x.shape[-1]
        squeezed_size = np.prod(batch_shape).item()
        x = x.reshape(squeezed_size, embed_size)
        x = self.linear(x)
        x = torch.reshape(x, (squeezed_size, *self.conv_shape))
        x = self.decoder(x)
        mean = torch.reshape(x, (*batch_shape, *self.shape))
        obs_dist = td.Independent(td.Normal(mean, 1), len(self.shape))
        return obs_dist


class ObservationDecoder2(nn.Module):
    def __init__(self, depth=32, stride=2, activation=nn.ReLU, embed_size=230, shape=(1, 4, 200), dist=False):
        super().__init__()
        n_lanes = 4
        fc1_out = 400
        fc2_out = 600
        deconv_in_channels = int(fc2_out / n_lanes)
        self.shape = shape
        self.dist = dist

        self.fc1 = nn.Linear(embed_size, fc1_out)
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.deconv1 = nn.ConvTranspose2d(deconv_in_channels, 1, [1, 200], stride=1)

    def forward(self, x):  # [50, 230]
        """
        :param x: size(*batch_shape, embed_size)
        :return: obs_dist = size(*batch_shape, *self.shape)
        """
        batch_shape = x.shape[:-1]
        embed_size = x.shape[-1]
        squeezed_size = np.prod(batch_shape).item()
        x = x.reshape(squeezed_size, embed_size)

        x = F.relu(self.fc1(x))  # [50, 400]
        x = F.relu(self.fc2(x))  # [50, 600]
        x = x.view(x.size(0), -1, 4, 1)  # [50, 150, 4, 1]
        x = self.deconv1(x)  # [50, 1, 4, 200]

        mean = torch.reshape(x, (*batch_shape, *self.shape))

        if self.dist:
            obs_dist = td.Independent(td.Normal(mean, 1), len(self.shape))
            return obs_dist
        else:
            return mean


def conv_out(h_in, padding, kernel_size, stride):
    return int((h_in + 2. * padding - (kernel_size - 1.) - 1.) / stride + 1.)


def output_padding(h_in, conv_out, padding, kernel_size, stride):
    return h_in - (conv_out - 1) * stride + 2 * padding - (kernel_size - 1) - 1


def conv_out_shape(h_in, padding, kernel_size, stride):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)


def output_padding_shape(h_in, conv_out, padding, kernel_size, stride):
    return tuple(output_padding(h_in[i], conv_out[i], padding, kernel_size, stride) for i in range(len(h_in)))
