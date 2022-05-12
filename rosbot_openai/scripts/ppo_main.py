#!/usr/bin/env python
import gym
import numpy as np
from ppo import Agent
import ppo_env
import matplotlib.pyplot as plt
import rospy

if __name__ == '__main__':
    rospy.init_node('husarion_ppo',
                    anonymous=True, log_level=rospy.DEBUG)
    task_and_robot_environment_name = 'Husarion_Walldodge-v1'
    env = gym.make(task_and_robot_environment_name)
    rospy.loginfo("Gym environment done")
    n_actions = 2
    n_states = 30 #24 Laserscans, past_lin, past_ang, (desired pos in polarcoordinates 2), yaw, diff_angle
    batch_size=64
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=n_actions, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs, input_dims=n_states)

    score_history = []

    N = 20
    learn_iters = 0
    n_games = 300
    n_steps = 0


    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps+=1
            score +=reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps%N==0:
                agent.learn()
                learn_iters +=1
            observation = observation_

        score_history.append(score)
