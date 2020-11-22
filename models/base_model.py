from tools import load_dataset

class Model:
    def __init__(self, config, datadir):
        self._c = config
        self._dataset = iter(load_dataset(datadir, self._c))
        pass

    def get_sample(self):
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
        pass

    def _loss(self):
        raise NotImplementedError

    def create_replays(self):
        pass

