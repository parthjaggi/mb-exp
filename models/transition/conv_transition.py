import torch.nn.functional as F
from torch import nn
import torch


class ConvTransitionModel(nn.Module):
    """
    Phase is one-hot. [1, 0] or [0, 1].
    """
    def __init__(self):
        super().__init__()
        n_lanes = 4
        phase_dim = 120
        deconv_in_channels = int((128 + phase_dim) / n_lanes)

        self.conv1 = nn.Conv2d(1, 128, [1, 200], stride=1)
        self.fc1 = nn.Linear(128 * 4, 128)
        self.fc2 = nn.Linear(128 + phase_dim, 128 + phase_dim)
        self.deconv1 = nn.ConvTranspose2d(deconv_in_channels, 1, [1, 200], stride=1)

    def forward(self, sample):
        obs, phase = sample['x'], sample['phases']  # [50, 1, 4, 200]
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
    """
    Phase is not one-hot. [0] or [1].
    """
    def __init__(self):
        super().__init__()
        n_lanes = 4
        phase_dim = 60
        deconv_in_channels = int((128 + phase_dim) / n_lanes)

        self.conv1 = nn.Conv2d(1, 128, [1, 200], stride=1)
        self.fc1 = nn.Linear(128 * 4, 128)
        self.fc2 = nn.Linear(128 + phase_dim, 128 + phase_dim)
        self.deconv1 = nn.ConvTranspose2d(deconv_in_channels, 1, [1, 200], stride=1)

    def forward(self, sample):
        obs, phase, action = sample['x'], sample['phases'], sample['action']  # [50, 1, 4, 200]
        x = F.relu(self.conv1(obs))  # [50, 128, 4, 1]
        x = x.view(x.size(0), -1)  # [50, 512]
        x = F.relu(self.fc1(x))  # [50, 128]

        # phase = phase.view(phase.size(0), -1).float()  # [50, 60]
        x = torch.cat((x, phase), 1)  # [50, 188]

        x = F.relu(self.fc2(x))  # [50, 188]
        x = x.view(x.size(0), -1, 4, 1)  # [50, 47, 4, 1]
        x = self.deconv1(x)  # [50, 1, 4, 200]
        return x


class ConvTransitionModel2_2(nn.Module):
    """
    Phase is not one-hot. [0] or [1].
    """
    def __init__(self):
        super().__init__()
        n_lanes = 4
        phase_dim = 60
        deconv_in_channels = int((128 + phase_dim) / n_lanes)

        self.conv1 = nn.Conv2d(1, 128, [1, 200], stride=1)
        self.fc1 = nn.Linear(128 * 4, 128)
        self.fc2 = nn.Linear(128 + phase_dim + 1, 128 + phase_dim)
        self.deconv1 = nn.ConvTranspose2d(deconv_in_channels, 1, [1, 200], stride=1)

    def forward(self, sample):
        obs, phase, action = sample['x'], sample['phases'], sample['action']  # [50, 1, 4, 200]
        x = F.relu(self.conv1(obs))  # [50, 128, 4, 1]
        x = x.view(x.size(0), -1)  # [50, 512]
        x = F.relu(self.fc1(x))  # [50, 128]

        # phase = phase.view(phase.size(0), -1).float()  # [50, 60]
        x = torch.cat((x, phase, action), 1)  # [50, 189]

        x = F.relu(self.fc2(x))  # [50, 189]
        x = x.view(x.size(0), -1, 4, 1)  # [50, 47, 4, 1]
        x = self.deconv1(x)  # [50, 1, 4, 200]
        return x


class ConvTransitionModel3(nn.Module):
    """
    Same as ConvTransitionModel2, but takes in 2 channels of observation (pos + speed).
    """
    def __init__(self):
        super().__init__()
        n_lanes = 4
        phase_dim = 60
        deconv_in_channels = int((128 + phase_dim) / n_lanes)

        self.conv1 = nn.Conv2d(2, 128, [1, 200], stride=1)
        self.fc1 = nn.Linear(128 * 4, 128)
        self.fc2 = nn.Linear(128 + phase_dim, 128 + phase_dim)
        self.deconv1 = nn.ConvTranspose2d(deconv_in_channels, 2, [1, 200], stride=1)

    def forward(self, sample):
        obs, phase = sample['x'], sample['phases']  # [50, 1, 4, 200]
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
    """
    Phase is not one-hot. [0] or [1].
    """
    def __init__(self):
        super().__init__()
        n_lanes = 4
        phase_action_dim = 61
        
        # [50, 200]
        self.fc1 = nn.Linear(200, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400 + phase_action_dim, 400)
        self.fc4 = nn.Linear(400, 200)
        self.sig = nn.Sigmoid()

    def forward(self, sample):
        obs, phase, action = sample['x'], sample['phases'], sample['action']  # [50, 200]
        x = F.relu(self.fc1(obs))  # [50, 400]
        x = F.relu(self.fc2(x))  # [50, 400]
        
        x = torch.cat((x, phase, action), 1)  # [50, 460]
        
        x = F.relu(self.fc3(x))  # [50, 400]
        x = self.fc4(x)  # [50, 200]
        return self.sig(x)
        # return x


class ClassificationModel2(nn.Module):
    """
    Phase is not one-hot. [0] or [1].
    Adapted from ClassificationModel. Uses 'action2' which is phase specific rather than extend-change.
    """
    def __init__(self):
        super().__init__()
        n_lanes = 4
        phase_action_dim = 62
        
        # [50, 200]
        self.fc1 = nn.Linear(200, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400 + phase_action_dim, 400)
        self.fc4 = nn.Linear(400, 200)
        self.sig = nn.Sigmoid()

    def forward(self, sample):
        obs, phase, phase_action = sample['x'], sample['phases'], sample['phase_action']  # [50, 200]
        x = F.relu(self.fc1(obs))  # [50, 400]
        x = F.relu(self.fc2(x))  # [50, 400]
        
        x = torch.cat((x, phase, phase_action), 1)  # [50, 460]
        
        x = F.relu(self.fc3(x))  # [50, 400]
        x = self.fc4(x)  # [50, 200]
        return self.sig(x)
        # return x


class ClassificationModel3(nn.Module):
    """
    Phase is not one-hot. [0] or [1].
    Uses 'phase_action' which is phase specific rather than extend-change.
    Adapted from ClassificationModel2. Limited phase history of t timesteps. Initially 10.
    """
    def __init__(self, history_length=10):
        super().__init__()
        n_lanes = 4
        self.history_length = history_length
        phase_action_dim = history_length + 2
        
        # [50, 200]
        self.fc1 = nn.Linear(200, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400 + phase_action_dim, 400)
        self.fc4 = nn.Linear(400, 200)
        self.sig = nn.Sigmoid()

    def forward(self, sample):
        obs, phase, phase_action = sample['x'], sample['phases'], sample['phase_action']  # [50, 200]
        phase = phase[:, -self.history_length:]
        x = F.relu(self.fc1(obs))  # [50, 400]
        x = F.relu(self.fc2(x))  # [50, 400]
        
        x = torch.cat((x, phase, phase_action), 1)  # [50, 412]
        
        x = F.relu(self.fc3(x))  # [50, 400]
        x = self.fc4(x)  # [50, 200]
        return self.sig(x)
        # return x


class LatentFCTransitionModel(nn.Module):
    """
    Adapted from ClassificationModel.
    No sigmoid activated at the end.
    """
    def __init__(self):
        super().__init__()
        n_lanes = 4
        phase_action_dim = 61
        
        obs_shape = 20
        self.fc1 = nn.Linear(obs_shape, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400 + phase_action_dim, 400)
        self.fc4 = nn.Linear(400, obs_shape)
        # self.sig = nn.Sigmoid()

    def forward(self, sample):
        obs, phase, action = sample['x'], sample['phases'], sample['action']  # [50, 20]
        x = F.relu(self.fc1(obs))  # [50, 400]
        x = F.relu(self.fc2(x))  # [50, 400]
        
        x = torch.cat((x, phase, action), 1)  # [50, 461]
        
        x = F.relu(self.fc3(x))  # [50, 400]
        x = self.fc4(x)  # [50, 20]
        # return self.sig(x)
        return x


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
