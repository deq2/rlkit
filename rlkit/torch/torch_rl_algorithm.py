import abc
from collections import OrderedDict
from typing import Iterable
import pickle

import numpy as np
from torch.autograd import Variable

import rlkit.core.eval_util
from rlkit.core.rl_algorithm import MetaRLAlgorithm
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
from rlkit.core import logger, eval_util


class MetaTorchRLAlgorithm(MetaRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(self, *args, render_eval_paths=False, plotter=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_statistics = None
        self.render_eval_paths = render_eval_paths
        self.plotter = plotter

    def get_batch(self):
        batch = self.replay_buffer.random_batch(self.task_idx, self.batch_size)
        return np_to_pytorch_batch(batch)

    @property
    @abc.abstractmethod
    def networks(self) -> Iterable[PyTorchModule]:
        pass

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def cuda(self):
        for net in self.networks:
            net.cuda()

    def obtain_samples(self, env):
        '''
        get rollouts of current policy in env
        '''
        return self.eval_sampler.obtain_samples()

    def evaluate(self, epoch):
        statistics = OrderedDict()
        # save stuff from training
        statistics.update(self.eval_statistics)
        self.eval_statistics = None
        print('evaluating on {} evaluation tasks'.format(len(self.eval_tasks)))
        for idx in range(len(self.eval_tasks)):
            self.task_idx = idx
            print('Task:', idx)
            # TODO how to handle eval over multiple tasks?
            self.eval_sampler.env.reset_task(idx)

            test_paths = self.obtain_samples(idx, epoch)
            # TODO incorporate into proper logging
            # save evaluation rollouts for vis
            with open("/mounts/output/proto-sac-point-mass-fb-16z-task{}-{}.pkl".format(idx, epoch), 'wb') as f:
                pickle.dump(test_paths, f, pickle.HIGHEST_PROTOCOL)

            statistics.update(eval_util.get_generic_path_information(
                test_paths, stat_prefix="Test_task{}".format(idx),
            ))
            statistics.update(eval_util.get_generic_path_information(
                self._exploration_paths, stat_prefix="Exploration_task{}".format(idx),
            ))
            if hasattr(self.env, "log_diagnostics"):
                self.env.log_diagnostics(test_paths)

            average_returns = rlkit.core.eval_util.get_average_returns(test_paths)
            statistics['AverageReturn_task{}'.format(idx)] = average_returns
        for key, value in statistics.items():
            logger.record_tabular(key, value)

        if self.render_eval_paths:
            self.env.render_paths(test_paths)

        if self.plotter:
            self.plotter.draw()

def _elem_or_tuple_to_variable(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(
            _elem_or_tuple_to_variable(e) for e in elem_or_tuple
        )
    return Variable(ptu.from_numpy(elem_or_tuple).float(), requires_grad=False)


def _filter_batch(np_batch):
    for k, v in np_batch.items():
        if v.dtype == np.bool:
            yield k, v.astype(int)
        else:
            yield k, v


def np_to_pytorch_batch(np_batch):
    return {
        k: _elem_or_tuple_to_variable(x)
        for k, x in _filter_batch(np_batch)
        if x.dtype != np.dtype('O')  # ignore object (e.g. dictionaries)
    }
