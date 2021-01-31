import time
from math import ceil
import collections
import torch
import torch.nn.functional as F

from tools import load_dataset, Every, Once, get_test_train_episodes, save_checkpoint
from models.base_model import Model
from models.transition.conv_transition import ConvTransitionModel, ConvTransitionModel2
from utils import EarlyStopping


class DreamerModel(Model):
    def __init__(self, config, preprocessing=False):
        super().__init__(config)
        self.dynamics = ConvTransitionModel2().cuda()
        self.optim = torch.optim.Adam(self.dynamics.parameters())
        self.earlystopping = EarlyStopping(patience=self._c.early_stop_patience)
        self.set_epoch_length()
        self.cutoff = None
        self.get_dataset_sample = self.get_dataset_sample2 if preprocessing else self.get_dataset_sample1

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

    def get_dataset_sample_old(self, dataset):
        s = next(dataset)
        sample = {}
        sample['phases'] = self.preprocess(torch.Tensor(s['phases'][:,  :, 0, 3, :, 0].numpy())).permute(1, 0, 2)
        # sample['y'] = self.preprocess(torch.Tensor(s['x'][:, 1:, 0, :, :, 0].numpy()))
        sample['v'] = self.preprocess(torch.Tensor(s['x'][:, :, :, :, :, 1].numpy())) + 0.5
        sample['x'] = self.preprocess(torch.Tensor(s['x'][:, :, :, :, :, 0].numpy())).permute(1, 0, 2, 3, 4)
        self.cutoff = 0.0

        ## not needed for now.
        # sample['reward'] = s['reward'].numpy()
        # sample['action'] = s['action'].numpy()
        return sample

    def get_dataset_sample1(self, dataset):
        # No preprocessing.
        s = dataset if isinstance(dataset, dict) else next(dataset)
        sample = {}
        sample['phases'] = torch.Tensor(s['phases'][:,  :, 0, 3, :, 0].numpy()).cuda().permute(1, 0, 2)
        sample['x'] = torch.Tensor(s['x'][:, :, :, :, :, 0].numpy()).cuda().permute(1, 0, 2, 3, 4)
        sample['action'] = torch.Tensor(s['corrected_action'][:, :].numpy()).cuda().permute(1, 0)
        sample['phase_action'] = torch.Tensor(s['corrected_p_action'][:, :].numpy()).cuda().permute(1, 0, 2)
        self.cutoff = 0.5

        # classification model only works on the last lane
        sample['x'] = sample['x'][:, :, 0, -1]

        # dreamer requires action indices be 1 step behind observations
        sample['x'] = sample['x'][1:]
        sample['phases'] = sample['phases'][1:]
        sample['action'] = sample['action'][:-1]
        sample['phase_action'] = sample['phase_action'][:-1]

        # sample['reward'] = s['reward'].numpy()
        return sample

    def get_dataset_sample2(self, dataset):
        # With preprocessing.
        s = dataset if isinstance(dataset, dict) else next(dataset)
        sample = {}
        sample['phases'] = self.preprocess(torch.Tensor(s['phases'][:,  :, 0, 3, :, 0].numpy())).permute(1, 0, 2)
        sample['x'] = self.preprocess(torch.Tensor(s['x'][:, :, :, :, :, 0].numpy())).permute(1, 0, 2, 3, 4)
        sample['action'] = torch.Tensor(s['corrected_action'][:, :].numpy()).cuda().permute(1, 0)
        sample['phase_action'] = torch.Tensor(s['corrected_p_action'][:, :].numpy()).cuda().permute(1, 0, 2)
        self.cutoff = 0.0

        # classification model only works on the last lane
        sample['x'] = sample['x'][:, :, 0, -1]

        # dreamer requires action indices be 1 step behind observations
        sample['x'] = sample['x'][1:]
        sample['phases'] = sample['phases'][1:]
        sample['action'] = sample['action'][:-1]
        sample['phase_action'] = sample['phase_action'][:-1]

        # sample['reward'] = s['reward'].numpy()
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
            test_loss = self.test()
            # scheduler.step(test_loss)
            self.earlystopping.step(test_loss)

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

    def train_dynamics(self, epoch):
        print('=======================> epoch:', epoch)
        train_loss = 0
        t1 = time.time()
        for u in range(self.epoch_length):
            s = self.get_dataset_sample(self._dataset)
            self.optim.zero_grad()
            y_pred = self.dynamics(s['x'], s['phases'])
            loss = F.mse_loss(y_pred, s['y'])
            loss.backward()
            train_loss += loss
            self.optim.step()
        
            if (u % int(self.epoch_length/min(self.epoch_length, 20)) == 0):
                t2 = time.time()
                print(u, round(t2-t1, 2), '{:.10f}'.format(loss.item() / self._c.batch_size))
        
        print('====> Epoch: {} Average loss: {:.10f}'.format(epoch, train_loss / (self.epoch_length * self._c.batch_size)))
    
    def test(self):
        self.dynamics.eval()
        test_loss = 0
        for u in range(self.test_epoch_length):
            s = self.get_dataset_sample(self._dataset)
            y_pred = self.dynamics(s['x'], s['phases'])
            test_loss += F.mse_loss(y_pred, s['y'])
        
        test_loss /= (self.test_epoch_length * self._c.batch_size)
        print('====> Test set loss: {:.10f}'.format(test_loss))
        print()
        return test_loss

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def _loss(self):
        raise NotImplementedError

    def create_reconstructions(self):
        pass

