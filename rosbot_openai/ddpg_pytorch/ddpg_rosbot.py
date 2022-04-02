#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import rospy
import rosbot_env

from ddpg_agent import Agent

if __name__ == '__main__':
    rospy.init_node('husarion_wall_ddpg',
                    anonymous=True, log_level=rospy.DEBUG)


    # Create the Gym environment
    # Init OpenAI_ROS Husarion ENV
    task_and_robot_environment_name = 'Husarion_Walldodge-v1'
    env = gym.make(task_and_robot_environment_name)
    rospy.loginfo("Gym environment done")
    nb_actions = 2
    env.seed(10)
    agent = Agent(state_size=24, action_size=2, random_seed=10)


    def ddpg(n_episodes=2000, max_t=700):
        scores_deque = deque(maxlen=100)
        scores = []
        max_score = -np.Inf
        for i_episode in range(1, n_episodes + 1):
            state = env.reset()
            agent.reset()
            score = 0
            for t in range(max_t):
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            scores_deque.append(score)
            scores.append(score)
            print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), score))
            if i_episode % 100 == 0:
                torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
                torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        return scores


    scores = ddpg()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()