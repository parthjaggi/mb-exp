import yaml
import gym
import numpy as np
from wolf.utils.configuration.registry import R


class TrafficEnv:

    def __init__(self):
        # config_path = '/home/parth/repos/traffic-management/sow45_code3/wolf/ray/tests/traffic_env/test4_2/iql_global_reward_no_dueling_dtse.yaml'
        # config_path = '/home/parth/repos/traffic-management/sow45_code3/wolf/ray/tests/traffic_env/test4_2/iql_global_reward_no_dueling_trans_image.yaml'
        config_path = '/home/parth/repos/traffic-management/sow45_code3/wolf/ray/tests/traffic_env/test0_1/iql_global_reward_dtse.yaml'
        
        env_name, env_config = self.load_config(config_path)
        # self._size = (16, 200)  # for test4_2 dtse
        self._size = (4, 200)  # for test0_1 dtse
        self._env = self.get_env(env_name, env_config)
        self._node_id = next(iter(self._env._agents.keys()))

    def load_config(self, config_path):
        with open(config_path) as file:
            config = yaml.safe_load(file)

        experiments = config['ray']['run_experiments']['experiments']
        experiment = next(iter(experiments.values()))
        env_name = experiment['config']['env']
        env_config = experiment['config']['env_config']
        env_config['horizon'] = experiment['config']['horizon']
        self.gamma = experiment['config']['gamma']

        return env_name, env_config

    def get_env(self, env_name, env_config):
        create_env = R.env_factory(env_name)
        env = create_env(env_config)
        return env

    @property
    def observation_space(self):
        shape = self._size + (2,)
        space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        return gym.spaces.Dict({'image': space})

    @property
    def action_space(self):
        return next(iter(self._env._agents.values())).action_space()

    # def step(self, action):
    #   action = {self._node_id: action}
    #   time_step = self._env.step(action)
    #   obs = dict(time_step.observation)
    #   obs['image'] = self.render()
    #   reward = time_step.reward or 0
    #   done = time_step.last()
    #   info = {'discount': np.array(time_step.discount, np.float32)}
    #   return obs, reward, done, info

    def step(self, action):
        action = {self._node_id: action}
        obs, reward, done, info = self._env.step(action)
        obs = self.transform_obs(obs)
        done = done['__all__']
        reward = reward[self._node_id]
        info = {'discount': np.array(self.gamma, np.float32)}
        return obs, reward, done, info

    def transform_obs(self, obs):
        # obs = obs[self._node_id]['dtse']  # shape: (1, 16, 200, 2)
        obs = obs[self._node_id]  # shape: (1, 16, 200, 2)
        # obs = np.transpose(obs, (2, 0, 1))
        # obs['image'].shape: (64, 64, 3)
        return obs

    # def reset(self):
    #   time_step = self._env.reset()
    #   obs = dict(time_step.observation)
    #   obs['image'] = self.render()
    #   return obs

    def reset(self):
        obs = self._env.reset()
        self.obs = self.transform_obs(obs)
        return self.obs

    def render(self, *args, **kwargs):
        pass