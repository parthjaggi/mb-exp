import time
from math import ceil
import collections
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from tools import load_dataset, Every, Once, get_test_train_episodes, save_checkpoint
from models.base_model import Model
from models.transition.conv_transition import *
from utils import EarlyStopping, reparameterize


class ConvModel(Model):
    def __init__(self, config):
        super().__init__(config)
        if config.type == 'conv':
            # self.dynamics = ConvTransitionModel2().cuda()
            self.dynamics = ConvTransitionModel2_2().cuda()  # uses corrected action
            self.get_dataset_sample = self.get_dataset_sample_no_speed
            self.criterion = F.mse_loss
        elif config.type == 'conv_speed':
            self.dynamics = ConvTransitionModel3().cuda()
            self.get_dataset_sample = self.get_dataset_sample_with_speed
            self.criterion = F.mse_loss
        elif config.type == 'class':
            # self.dynamics = ClassificationModel().cuda()
            # self.dynamics = ClassificationModel2().cuda()  # uses corrected phase action
            self.dynamics = ClassificationModel3().cuda()  # uses limited phase history
            self.get_dataset_sample = self.get_dataset_sample_for_classification
            self.criterion = torch.nn.BCELoss()
        elif config.type == 'latent_fc':
            self.dynamics = LatentFCTransitionModel().cuda()
            self.get_dataset_sample = self.get_dataset_sample_for_latent_fc
            self.criterion = F.mse_loss
        elif config.type == 'veh':
            self.dynamics = VehicleTransitionModel().cuda()
            self.get_dataset_sample = self.get_dataset_sample_for_veh_model
            self.criterion = F.mse_loss
        else:
            raise NotImplementedError

        self.optim = torch.optim.Adam(self.dynamics.parameters())
        self.earlystopping = EarlyStopping(patience=self._c.early_stop_patience)
        self.set_epoch_length()
        self.writer = SummaryWriter(log_dir=config.logdir, purge_step=0)

    def preprocess(self,):
        pass

    # def get_sample(self):
    #     return get_dataset_sample(self._dataset)
        # danijar style get sample
        # yield method
        # choose any episode
        # why yield episode and not sample.

        # while true
            # for files in directory:
                # if not in cache add to cache
            # for i in random set of cache:
                # length limitation?
                # yield i episode
        
        # while true
            # check for files in dir, add new files to cache
            # for i in train_steps number of episodes (sampled from episode cache):
                # yield a sample of given length
        pass

    def get_dataset_sample_no_speed(self, dataset):
        s = dataset if isinstance(dataset, dict) else next(dataset)
        sample = {}
        sample['phases'] = torch.Tensor(s['phases'][:,  0, 0, 3, :, 0].numpy()).cuda()
        sample['y'] = self.preprocess(torch.Tensor(s['x'][:, 1, :, :, :, 0].numpy()))
        sample['v'] = self.preprocess(torch.Tensor(s['x'][:, 0, :, :, :, 1].numpy())) + 0.5
        sample['x'] = self.preprocess(torch.Tensor(s['x'][:, 0, :, :, :, 0].numpy()))
        sample['action'] = torch.Tensor(s['corrected_action'][:, :1].numpy()).cuda()
        
        ## not needed for now.
        sample['reward'] = s['reward'].numpy()
        # sample['action'] = s['action'].numpy()
        return sample

    def get_dataset_sample_with_speed(self, dataset):
        s = dataset if isinstance(dataset, dict) else next(dataset)
        sample = {}
        sample['phases'] = self.preprocess(torch.Tensor(s['phases'][:,  0, 0, 3, :, 0].numpy()))
        sample['y'] = self.preprocess(torch.Tensor(s['x'][:, 1, :, :, :, :].numpy())).permute(0, 4, 2, 3, 1).squeeze(-1).contiguous()
        sample['v'] = self.preprocess(torch.Tensor(s['x'][:, 0, :, :, :, 1].numpy())) + 0.5
        sample['x'] = self.preprocess(torch.Tensor(s['x'][:, 0, :, :, :, :].numpy())).permute(0, 4, 2, 3, 1).squeeze(-1).contiguous()
        
        sample['x'][:, 1] = sample['x'][:, 1] + 0.5
        sample['y'][:, 1] = sample['y'][:, 1] + 0.5

        ## not needed for now.
        sample['reward'] = s['reward'].numpy()
        # sample['action'] = s['action'].numpy()
        return sample

    def get_dataset_sample_for_classification(self, dataset):
        s = dataset if isinstance(dataset, dict) else next(dataset)
        sample = {}
        sample['phases'] = torch.Tensor(s['phases'][:,  0, 0, 3, :, 0].numpy()).cuda()
        sample['y'] = torch.Tensor(s['x'][:, 1, :, :, :, 0].numpy()).cuda()
        sample['v'] = torch.Tensor(s['x'][:, 0, :, :, :, 1].numpy()).cuda()
        sample['x'] = torch.Tensor(s['x'][:, 0, :, :, :, 0].numpy()).cuda()
        sample['action'] = torch.Tensor(s['corrected_action'][:, :1].numpy()).cuda()
        sample['phase_action'] = torch.Tensor(s['corrected_p_action'][:, 0].numpy()).cuda()

        # classification model only works on the last lane
        sample['x'] = sample['x'][:, 0, -1]
        sample['y'] = sample['y'][:, 0, -1]
        
        ## not needed for now.
        sample['reward'] = s['reward'].numpy()
        return sample

    def get_dataset_sample_for_classification_kstep(self, dataset):
        # to see accuracy of k-step predictions
        # need formatted samples of higher batch_length
        s = dataset if isinstance(dataset, dict) else next(dataset)
        sample = {}
        sample['phases'] = torch.Tensor(s['phases'][:,  :, 0, 3, :, 0].numpy()).cuda()
        sample['x'] = torch.Tensor(s['x'][:, :, :, :, :, 0].numpy()).cuda()
        sample['action'] = torch.Tensor(s['corrected_action'][:, :].numpy()).cuda()
        sample['phase_action'] = torch.Tensor(s['corrected_p_action'][:, :].numpy()).cuda()

        # classification model only works on the last lane
        sample['x'] = sample['x'][:, :, 0, -1]

        sample['reward'] = s['reward'].numpy()
        return sample

    def get_dataset_sample_for_latent_fc(self, dataset):
        s = dataset if isinstance(dataset, dict) else next(dataset)
        sample = {}
        sample['phases'] = torch.Tensor(s['phases'][:,  0, 0, 3, :, 0].numpy()).cuda()
        sample['action'] = torch.Tensor(s['corrected_action'][:, :1].numpy()).cuda()
        sample['reward'] = s['reward'].numpy()

        mu = (torch.Tensor(s['mu'].numpy())).cuda()
        logvar = (torch.Tensor(s['logvar'].numpy())).cuda()
        latent = reparameterize(mu, logvar)

        sample['x'] = latent[:, 0]
        sample['y'] = latent[:, 1]
        return sample

    def get_dataset_sample_for_veh_model(self, dataset):
        s = dataset if isinstance(dataset, dict) else next(dataset)
        sample = {}
        # bs = self._c.batch_size
        sample['x'] = torch.Tensor(s['x'][:, 0, :].numpy()).cuda()
        sample['y'] = torch.Tensor(s['y'][:, 0, :].numpy()).cuda()
        sample['phases'] = torch.Tensor(s['phases'][:, 0, :].numpy()).cuda()

        sample['x'][:, [0, 2]] = sample['x'][:, [0, 2]] / 200  # detection range
        sample['x'][:, [1, 3]] = sample['x'][:, [1, 3]] / 35

        sample['y'][:, 0] = sample['y'][:, 0] / 200
        sample['y'][:, 1] = sample['y'][:, 1] / 35
        return sample

    def preprocess(self, x):
        x = x - 0.5
        return x.cuda()

    def set_epoch_length(self):
        """
        These many number of batches when sampled from the dataset would lead to 1 epoch.
        """
        num_episodes = len(self.train_eps)
        episode_length = 500
        batch_length = self._c.batch_length
        batch_size = self._c.batch_size
        self.epoch_length = ceil(num_episodes * (episode_length - (batch_length - 1)) / batch_size)

        test_num_episodes = len(self.test_eps)
        self.test_epoch_length = ceil(test_num_episodes * (episode_length - (batch_length - 1)) / batch_size)

    def batch_update_model(self):
        # calculate loss
        # loss.backward()
        # optim.step()

        sample = self.get_sample()
        loss = self._loss(sample)
        loss.backward()
        self.optim.step()

    def train(self):
        cur_best = None
        for epoch in range(self._c.epochs):
            self.train_dynamics(epoch)
            test_loss = self.test(epoch)
            # scheduler.step(test_loss)
            self.earlystopping.step(test_loss)
            self.writer.file_writer.flush()

            # checkpointing
            best_filename = self._c.logdir / 'best.tar'
            filename = self._c.logdir / f'checkpoint_{epoch}.tar'
            is_best = not cur_best or test_loss < cur_best
            if is_best:
                cur_best = test_loss
            
            if is_best or (epoch % 10 == 0):
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': self.dynamics.state_dict(),
                    'precision': test_loss,
                    'optimizer': self.optim.state_dict(),
                    'earlystopping': self.earlystopping.state_dict(),
                    # 'scheduler': scheduler.state_dict(),
                }
                save_checkpoint(checkpoint, is_best, filename, best_filename)

            if self.earlystopping.stop:
                print("End of Training because of early stopping at epoch {}".format(epoch))

    def train_dynamics(self, epoch):
        print('=======================> epoch:', epoch)
        self.dynamics.train()
        train_loss = 0
        t1 = time.time()
        for u in range(self.epoch_length):
            s = self.get_dataset_sample(self._dataset)
            self.optim.zero_grad()
            y_pred = self.dynamics(s)
            loss = self.criterion(y_pred, s['y'])
            loss.backward()
            train_loss += loss
            self.optim.step()
        
            if (u % int(self.epoch_length/min(self.epoch_length, 5)) == 0):
                t2 = time.time()
                print(u, round(t2-t1, 2), '{:.10f}'.format(loss.item() / self._c.batch_size))
        
        norm_train_loss = (train_loss / (self.epoch_length * self._c.batch_size)).item()
        self.writer.add_scalar('train/loss', norm_train_loss, epoch)
        print('====> Epoch: {} Average loss: {:.10f}'.format(epoch, norm_train_loss))
    
    def test(self, epoch):
        self.dynamics.eval()
        test_loss = 0
        for u in range(self.test_epoch_length):
            s = self.get_dataset_sample(self._test_dataset)
            y_pred = self.dynamics(s)
            test_loss += F.mse_loss(y_pred, s['y'])
        
        norm_test_loss = (test_loss / (self.test_epoch_length * self._c.batch_size)).item()
        self.writer.add_scalar('test/loss', norm_test_loss, epoch)
        print('====> Test set loss: {:.10f}'.format(norm_test_loss))
        print()
        return norm_test_loss

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def _loss(self):
        raise NotImplementedError

    def create_reconstructions(self):
        pass

