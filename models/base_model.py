import collections

from tools import load_dataset, Every, Once, get_test_train_episodes


class Model:
    def __init__(self, config):
        datadir = config.datadir
        self._c = config
        self.should_eval = Every(config.eval_every)
        self.should_log = Every(config.log_every)
        self.should_image_log = Every(config.image_log_every)
        self.should_save_model = Every(config.save_model_every)

        self.train_eps, self.test_eps = get_test_train_episodes(self._c)
        self._dataset = iter(load_dataset(self._c, self.train_eps))
        self._test_dataset = iter(load_dataset(self._c, self.test_eps))

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def _loss(self):
        raise NotImplementedError

    def create_reconstructions(self):
        pass

