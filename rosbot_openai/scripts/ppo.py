#!/usr/bin/env python
import numpy as np
import torch

from networks import ActorNetwork, CriticNetwork, FeedForwardNN
from torch.optim import Adam
import torch.nn as nn
import rospy
import ppo_env
import gym
from torch.distributions import MultivariateNormal




class PPO:


    def __init__(self, env):
        self._init_hyperparameters()

        self.env = env
        self.obs_dim=29
        self.act_dim=2

        self.actor = ActorNetwork(self.obs_dim)
        self.critic = CriticNetwork(self.obs_dim)



        self.cov_var = torch.full(size=(self.act_dim,),fill_value=0.5)

        self.cov_mat = torch.diag(self.cov_var)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr_critic)

    def _init_hyperparameters(self):
        self.timesteps_per_batch = 20000
        self.max_timesteps_per_episode = 1500
        self.gamma = 0.99
        self.n_updates_per_iteration = 80
        self.clip = 0.2
        self.lr_actor = 0.0003
        self.lr_critic = 0.001


    def get_action(self, obs):
        mean = self.actor(obs)
        dist = MultivariateNormal(mean, self.cov_mat)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()

    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        t = 0
        total_reward=0

        while t<self.timesteps_per_batch:
            ep_rews = []

            obs = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                t+=1
                batch_obs.append(obs)
                action, log_prob = self.get_action(obs)
                action[0] = np.clip(action[0], 0.0, 1.0)
                action[1] = np.clip(action[1], -1.0, 1.0)
                obs, rew, done, _ = self.env.step(action)

                total_reward+=rew
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            batch_lens.append(ep_t+1)
            batch_rews.append(ep_rews)

        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        file = open("/home/marvin/ros_workspace/src/rosbot_openai/logs/ep_rewards.txt", "a")
        file.write(str(int(total_reward)) + ",")
        file.close()

        batch_rtgs = self.compute_rtgs(batch_rews)

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def compute_rtgs(self, batch_rews):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []  # Iterate through each episode backwards to maintain same order
        # in batch_rtgs
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0  # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)  # Convert the rewards-to-go into a tensor

        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs

    def learn(self, total_timesteps):
        t_so_far = 0
        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()


            V, _ = self.evaluate(batch_obs, batch_acts)

            A_k = batch_rtgs - V.detach()

            A_k = (A_k -A_k.mean())/(A_k.std()+1e-10)

            for _ in range(self.n_updates_per_iteration):
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                ratios = torch.exp(curr_log_probs-batch_log_probs)

                surr1 = ratios*A_k
                surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip)*A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V,batch_rtgs)

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

            t_so_far+=np.sum(batch_lens)

    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        return V, log_probs


if __name__ == '__main__':
    rospy.init_node('husarion_ppo', anonymous=True, log_level=rospy.DEBUG)
    task_and_robot_environment_name = 'Husarion_Walldodge-v1'
    env = gym.make(task_and_robot_environment_name)
    model = PPO(env)
    model.learn(400000)