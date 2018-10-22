from collections import OrderedDict
import numpy as np
import pickle

import torch
import torch.optim as optim
from torch import nn as nn
from torch.autograd import Variable

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import np_ify, torch_ify
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import MetaTorchRLAlgorithm
from rlkit.torch.sac.policies import MakeDeterministic, ProtoExplorationPolicy


GOAL_VEC = [-1, 1]

class ProtoSoftActorCritic(MetaTorchRLAlgorithm):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            nets,

            class_lr=1e-1,
            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            context_lr=1e-3,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,
            eval_deterministic=True,
            **kwargs
    ):
        self.task_enc, self.policy, self.qf, self.vf = nets
        super().__init__(
            env=env,
            policy=self.policy,
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            **kwargs
        )
        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.target_vf = self.vf.copy()
        self.class_criterion = nn.BCEWithLogitsLoss()
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.eval_statistics = None

        self.class_optimizer = optim.SGD(
                self.task_enc.parameters(),
                lr=class_lr,
        )

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf_optimizer = optimizer_class(
            self.qf.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
        )
        self.context_optimizer = optimizer_class(
            self.task_enc.parameters(),
            lr=context_lr,
        )

    def make_dataset(self, batch, idx):
        obs = batch['observations']
        rewards = batch['rewards'] / self.reward_scale
        targets = np.full_like(rewards, idx)
        l = len(obs) // 2
        train = np.concatenate([obs[:l], rewards[:l], targets[:l]], axis=1)
        test = np.concatenate([obs[l:], rewards[l:], targets[l:]], axis=1)
        return train, test

    def train_task_classifier(self, train_flag=True):
        # train classifier to convergence to distinguish between tasks,
        # using data collected by exploration policy
        training_data = []
        test_data = []
        for idx in self.train_tasks:
            nsamp = min(2000, self.replay_buffer.num_steps_can_sample(idx))
            train, test = self.make_dataset(self.replay_buffer.random_batch(idx, nsamp), idx)
            training_data.append(train)
            test_data.append(test)
        training_data = np.concatenate(training_data)
        test_data = np.concatenate(test_data)

        if train_flag:
            batch_size = 256
            for epoch in range(10):
                for it in range(len(training_data // batch_size)):
                    indices = np.random.choice(len(training_data), batch_size)
                    train = training_data[indices]
                    obs = torch_ify(train[:, :2])
                    rewards = torch_ify(train[:, 2:3])
                    targets = torch_ify(train[:, -1:])
                    preds = self.task_enc(obs, rewards)
                    class_loss = self.class_criterion(preds, targets)
                    class_loss.backward()
                    self.class_optimizer.step()
                    self.class_optimizer.zero_grad()
                print('Loss:', class_loss)
        # evaluate
        obs = torch_ify(test_data[:, :2])
        rewards = torch_ify(test_data[:, 2:3])
        targets = test_data[:, -1:]
        preds = (np_ify(self.task_enc(obs, rewards).detach()) > 0).astype(np.int)
        errors = np.sum(np.abs(np_ify(preds) - targets)) / len(preds)
        print('Classification error:', errors)

    def make_exploration_policy(self, policy):
        return ProtoExplorationPolicy(policy)

    def make_eval_policy(self, policy, deterministic=True):
        if deterministic:
            eval_policy = MakeDeterministic(policy)
        else:
            eval_policy = self.policy
        return eval_policy

    def obtain_samples(self, idx, epoch):
        '''
        this is more involved than usual because we have to sample rollouts, compute z, then sample new rollouts conditioned on z
        '''
        # TODO for now set task encoder to zero, should be sampled
        # TODO: collect context tuples from replay buffer to match training stats
        batch = self.get_batch()
        rewards = batch['rewards']
        obs = batch['observations']
        goals = batch['goals']
        # goals = goal.
        # Evaluate task classifier on sampled tuples
        # Task encoding is classification prob of a single tuple
<<<<<<< HEAD
        z = np_ify(torch.mean(self.task_enc(obs, rewards / self.reward_scale, goals)))
=======
        z = np_ify(torch.mean(self.task_enc(obs, rewards / self.reward_scale)))
>>>>>>> parent of bdadf3d... make task encoding multi-dimensional
        print('task encoding', z)
        self.eval_sampler.policy.set_eval_z(z)

        test_paths = self.eval_sampler.obtain_samples(explore=False)
        return test_paths

    def perform_meta_update(self):
        # assume gradients have been accumulated for each parameter, apply update
        self.qf_optimizer.step()
        self.vf_optimizer.step()
        self.policy_optimizer.step()
        self.context_optimizer.step()
        self._update_target_network()

        self.qf_optimizer.zero_grad()
        self.vf_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
        self.context_optimizer.zero_grad()

    def _do_training(self):

        # sample from replay buffer to compute training update
        batch = self.get_batch()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        goals = batch['goals']
        # print('goals')
        # print(goals)

        # NOTE: right now policy is updated on the same rollouts used
        # for the task encoding z
        z = torch.mean(self.task_enc(obs, rewards / self.reward_scale))
        batch_z = z.repeat(obs.shape[0])[..., None]
        q_pred = self.qf(obs, actions, batch_z)
        v_pred = self.vf(obs, batch_z)
        # make sure policy accounts for squashing functions like tanh correctly!
        in_ = torch.cat([obs, batch_z.detach()], dim=1)
        policy_outputs = self.policy(in_, return_log_prob=True)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # qf loss and gradients
        target_v_values = self.target_vf(next_obs, batch_z)
        q_target = rewards + (1. - terminals) * self.discount * target_v_values
        qf_loss = self.qf_criterion(q_pred, q_target.detach())
        qf_loss.backward(retain_graph=True)

        # vf loss and gradients
        q_new_actions = self.qf(obs, new_actions, batch_z)
        v_target = q_new_actions - log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        vf_loss.backward(retain_graph=True)

        # policy loss and gradients
        log_policy_target = q_new_actions - v_pred
        policy_loss = (
            log_pi * (log_pi - log_policy_target).detach()
        ).mean()
        policy_loss.backward(retain_graph=True)
        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value**2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))

        # update policy's task encoding for data collection
        self.policy.set_eval_z(np_ify(z))


    @property
    def networks(self):
        return [
            self.policy,
            self.qf,
            self.vf,
            self.target_vf,
        ]

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(
            qf=self.qf,
            policy=self.policy,
            vf=self.vf,
            target_vf=self.target_vf,
        )
        return snapshot