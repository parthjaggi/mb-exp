import torch.nn.functional as F
from torch import nn
import torch


class ConvTransitionModel(nn.Module):
    def __init__(self):
        """
        Phase is one-hot. [1, 0] or [0, 1].
        """
        super().__init__()
        n_lanes = 4
        phase_dim = 120
        deconv_in_channels = int((128 + phase_dim) / n_lanes)

        self.conv1 = nn.Conv2d(1, 128, [1, 200], stride=1)
        self.fc1 = nn.Linear(128 * 4, 128)
        self.fc2 = nn.Linear(128 + phase_dim, 128 + phase_dim)
        self.deconv1 = nn.ConvTranspose2d(deconv_in_channels, 1, [1, 200], stride=1)

    def forward(self, obs, phase):  # [50, 1, 4, 200]
        x = F.relu(self.conv1(obs))  # [50, 128, 4, 1]
        x = x.view(x.size(0), -1)  # [50, 512]
        x = F.relu(self.fc1(x))  # [50, 128]

        phase = phase.view(phase.size(0), -1).float()  # [50, 60, 2]
        x = torch.cat((x, phase), 1)  # [50, 248]

        x = F.relu(self.fc2(x))  # [50, 248]
        x = x.view(x.size(0), -1, 4, 1)  # [50, 62, 4, 1]
        x = self.deconv1(x)  # [50, 1, 4, 200]
        return x


class ConvTransitionModel2(nn.Module):
    def __init__(self):
        """
        Phase is not one-hot. [0] or [1].
        """
        super().__init__()
        n_lanes = 4
        phase_dim = 60
        deconv_in_channels = int((128 + phase_dim) / n_lanes)

        self.conv1 = nn.Conv2d(1, 128, [1, 200], stride=1)
        self.fc1 = nn.Linear(128 * 4, 128)
        self.fc2 = nn.Linear(128 + phase_dim, 128 + phase_dim)
        self.deconv1 = nn.ConvTranspose2d(deconv_in_channels, 1, [1, 200], stride=1)

    def forward(self, obs, phase):  # [50, 1, 4, 200]
        x = F.relu(self.conv1(obs))  # [50, 128, 4, 1]
        x = x.view(x.size(0), -1)  # [50, 512]
        x = F.relu(self.fc1(x))  # [50, 128]

        # phase = phase.view(phase.size(0), -1).float()  # [50, 60]
        x = torch.cat((x, phase), 1)  # [50, 188]

        x = F.relu(self.fc2(x))  # [50, 188]
        x = x.view(x.size(0), -1, 4, 1)  # [50, 47, 4, 1]
        x = self.deconv1(x)  # [50, 1, 4, 200]
        return x


class ConvTransitionModel3(nn.Module):
    def __init__(self):
        """
        Same as ConvTransitionModel2, but takes in 2 channels of observation (pos + speed).
        """
        super().__init__()
        n_lanes = 4
        phase_dim = 60
        deconv_in_channels = int((128 + phase_dim) / n_lanes)

        self.conv1 = nn.Conv2d(2, 128, [1, 200], stride=1)
        self.fc1 = nn.Linear(128 * 4, 128)
        self.fc2 = nn.Linear(128 + phase_dim, 128 + phase_dim)
        self.deconv1 = nn.ConvTranspose2d(deconv_in_channels, 2, [1, 200], stride=1)

    def forward(self, obs, phase):  # [50, 1, 4, 200]
        x = F.relu(self.conv1(obs))  # [50, 128, 4, 1]
        x = x.view(x.size(0), -1)  # [50, 512]
        x = F.relu(self.fc1(x))  # [50, 128]

        # phase = phase.view(phase.size(0), -1).float()  # [50, 60]
        x = torch.cat((x, phase), 1)  # [50, 188]

        x = F.relu(self.fc2(x))  # [50, 188]
        x = x.view(x.size(0), -1, 4, 1)  # [50, 47, 4, 1]
        x = self.deconv1(x)  # [50, 1, 4, 200]
        return x


class ClassificationModel(nn.Module):
    def __init__(self):
        """
        Phase is not one-hot. [0] or [1].
        """
        super().__init__()
        n_lanes = 4
        phase_dim = 60
        deconv_in_channels = int((128 + phase_dim) / n_lanes)
        
        # [50, 200]
        self.fc1 = nn.Linear(200, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400 + phase_dim, 400)
        self.fc4 = nn.Linear(400, 200)
        self.sig = nn.Sigmoid()

    def forward(self, obs, phase):  # [50, 200]
        x = F.relu(self.fc1(obs))  # [50, 400]
        x = F.relu(self.fc2(x))  # [50, 400]
        
        x = torch.cat((x, phase), 1)  # [50, 460]
        
        x = F.relu(self.fc3(x))  # [50, 400]
        x = self.fc4(x)  # [50, 200]
        return self.sig(x)
        # return x


class VehicleTransitionModel(nn.Module):
    def __init__(self):
        super().__init__()
        input_shape = 5 + 60
        output_shape = 2
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, output_shape)

    def forward(self, x, phase):
        x = torch.cat((x, phase), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
