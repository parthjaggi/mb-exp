import torch.nn.functional as F
from torch import nn
import torch
from utils import first, second


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

    def forward(self, sample):
        x, phase = sample['x'], sample['phases']
        x = torch.cat((x, phase), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class VehicleTransitionModelWrapper:
    """
    Wrapper for VehicleTransitionModel type models.
    Used when this model is to used inside rollouts.
    """
    DISCRETIZATION = LINK_LENGTH = 200
    HALF_WIDTH = LINK_LENGTH / (DISCRETIZATION * 2)

    def __init__(self, model):
        self.model = model

    def __call__(self, sample):
        sample = {'x': sample['x'].reshape(-1), 'v': sample['v'].reshape(-1), 'p': sample['next_phases']}
        samples, p = self.convert_to_vtm_format(sample)
        ls, ss = self.convert_vtm_output_to_dtse(samples, p)
        return ls, ss
    
    def convert_to_vtm_format(self, sample):
        """
        Creates samples in the format usable by the vehicle transition model.
        Samples have the following features: veh_pos, veh_spd, ds_veh_pos, ds_veh_spd, has_ds_veh.

        Args:
            x (torch.Tensor): Single lane state. Should be of shape: [DISCRETIZATION].
            v (torch.Tensor): Single lane speed state. Should be of shape: [DISCRETIZATION].
            p (torch.Tensor): Phase history. Should be of shape: [1, HISTORY_LEN].

        Returns:
            tuple: Tuple of 2 torch Tensors which has samples and phase information for vehicle transition model.
                samples is of shape (vehicle count, number of features). phase is of shape (vehicle count, HISTORY_LEN).
        """
        
        POS_IDXS, HISTORY_LEN = [0, 2], 60
        veh_infos = []

        x, v, p = sample['x'], sample['v'], sample['p']

        assert first(x.shape) == self.DISCRETIZATION
        assert first(v.shape) == self.DISCRETIZATION
        assert second(p.shape) == HISTORY_LEN

        position_idxs = first(torch.where(x > 0))

        for idx in torch.flip(position_idxs, [0]):
            position = idx + self.HALF_WIDTH
            speed = v[idx]
            veh_infos.append(VehicleInfo(position, speed))

        samples = []
        for idx, veh_info in enumerate(veh_infos):
            downstream_veh_idx = idx + 1
            if downstream_veh_idx < len(veh_infos):
                downstream_veh_info = veh_infos[downstream_veh_idx]
                samples.append([veh_info.pos, veh_info.vel, downstream_veh_info.pos, downstream_veh_info.vel, 1])
            else:
                samples.append([veh_info.pos, veh_info.vel, 0, 0, 0])

        samples = torch.Tensor(samples).to(x.device)

        if len(veh_infos):
            samples[:, POS_IDXS] = samples[:, POS_IDXS] / self.LINK_LENGTH
            p = p.expand([first(samples.shape), -1]).to(x.device)

        return samples, p

    def convert_vtm_output_to_dtse(self, samples, p):
        MEAN_OBS_VALUE = 0.0
        dtse_x = torch.zeros(self.DISCRETIZATION) - MEAN_OBS_VALUE
        dtse_v = torch.zeros(self.DISCRETIZATION)
        
        if len(samples):
            input_samples = {'x': samples, 'phases': p}
            veh_model_output = self.model(input_samples)

            for norm_pos, norm_spd in veh_model_output:
                if norm_pos < 0: continue
                position_idx = int(norm_pos * self.LINK_LENGTH / (2 * self.HALF_WIDTH))
                dtse_x[position_idx] += 1
                dtse_v[position_idx] = norm_spd
        return dtse_x, dtse_v

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


class VehicleInfo:
    def __init__(self, pos, vel):
        self.pos = pos
        self.vel = vel
