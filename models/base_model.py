import collections

from tools import load_dataset, Every, Once


class Model:
    def __init__(self, config):
        datadir = config.datadir
        self._c = config
        self.should_eval = Every(config.eval_every)
        self.should_log = Every(config.log_every)
        self.should_image_log = Every(config.image_log_every)
        self.should_save_model = Every(config.save_model_every)

        train_eps, test_eps = self.get_train_test_split()
        self._dataset = iter(load_dataset(self._c, train_eps))
        # self._test_dataset = iter(load_dataset(self._c, test_eps))
        pass

    def get_train_test_split(self):
        # divide episodes into train and test. last 2 episodes are test.
        filenames = list(self._c.datadir.glob('*.npy'))
        train_eps = filenames[:-2]
        test_eps = filenames[-2:]
        return train_eps, test_eps

    def get_sample(self):
        return next(self._dataset)
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

    def batch_update_model(self):
        # calculate loss
        # loss.backward()
        # optim.step()

        sample = self.get_sample()
        loss = self._loss(sample)
        loss.backward()
        self.optim.step()

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def _loss(self):
        raise NotImplementedError

    def create_reconstructions(self):
        pass

