import numpy as np

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from .proto_sac import ProtoSoftActorCritic

class MetaLearner(object):
    """
    sample tasks and perform the meta-training loop
    """
    def __init__(
            self,
            envs,
            nets,
            meta_batch_size=32,
            meta_epochs=1000,
            *args,
            **kwargs
    ):
        self.envs = envs
        self.meta_epochs = meta_epochs
        self.meta_batch_size = meta_batch_size
        max_size = 1000000
        self.buffers = [EnvReplayBuffer(max_size, e) for e in envs]
        task_id = self.sample_task()
        self.rl_algo = ProtoSoftActorCritic(self.envs[task_id], *nets, *args, **kwargs)

    def sample_task(self):
        return np.random.randint(len(self.envs))

    def num_tasks(self):
        return len(self.envs)

    def train(self):
        for i in range(self.meta_epochs):
            print('epoch', i)
            for _ in range(self.meta_batch_size):
                # sample a task, along with its env and replay buffer
                task_id = self.sample_task()
                self.rl_algo.env = self.envs[task_id]
                self.rl_algo.replay_buffer = self.buffers[task_id]
                if ptu.gpu_enabled():
                    self.rl_algo.cuda()
                self.rl_algo.train()
                print(self.rl_algo.replay_buffer.num_steps_can_sample())
            print('updating nets')
            self.rl_algo.update_nets()
            print('evaluating policy')
            self.rl_algo._try_to_eval(i)

