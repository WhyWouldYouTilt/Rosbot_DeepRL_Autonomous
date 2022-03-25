#!/usr/bin/env python
import rospy
import gym
import numpy as np
import tensorflow as tf
from ddpg import *
import walldodge_actor_env

exploration_decay_start_step = 50000
state_dim = 42
action_dim = 2
action_linear_max = 0.25 # m/s
action_angular_max = 0.25  # rad/s
is_training = True


def main():
    rospy.init_node('husarion_wall_ddpg',
                    anonymous=True, log_level=rospy.DEBUG)

    # Create the Gym environment
    # Init OpenAI_ROS Husarion ENV
    task_and_robot_environment_name = 'Husarion_Walldodge-v1'
    env = gym.make(task_and_robot_environment_name)
    rospy.loginfo("Gym environment done")
    agent = DDPG(env, state_dim, action_dim)
    if is_training:
        print('Training mode')
        avg_reward_his = []
        total_reward = 0
        var = 1.

        while True:
            state = env.reset()
            one_round_step = 0

            while True:
                a = agent.action(state)
                a[0] = np.clip(np.random.normal(a[0], var), 0., 1.0)
                a[1] = np.clip(np.random.normal(a[1], var), -0.5, 0.5)

                state_, r, done, info = env.step(a)
                time_step = agent.perceive(state, a, r, state_, done)



                if time_step > 0:
                    total_reward += r

                if time_step % 10000 == 0 and time_step > 0:
                    print('---------------------------------------------------')
                    avg_reward = total_reward / 10000
                    print('Average_reward = ', avg_reward)
                    avg_reward_his.append(round(avg_reward, 2))
                    print('Average Reward:',avg_reward_his)

                    file = open("/home/marvin/ros_workspace/src/rosbot_openai/logs/Rewards.txt", "a")
                    file.write("Average Reward in " + str(int(time_step)) + " Steps : " + str(int(avg_reward))+"\n")
                    file.write("Total Reward in " + str(int(time_step)) + " Steps: " + str(int(total_reward))+"\n")
                    file.close()
                    total_reward = 0

                """if time_step % 5 == 0 and time_step > exploration_decay_start_step:
                    var *= 0.9999"""


                state = state_
                one_round_step += 1



                if done: #or one_round_step >= 1000:
                    #print('Step: %3i' % one_round_step, '| Var: %.2f' % var, '| Time step: %i' % time_step, '|')
                    break

    else:
        print('Testing mode')
        while True:
            state = env.reset()
            one_round_step = 0

            while True:
                a = agent.action(state)
                a[0] = np.clip(a[0], 0., 1.)
                a[1] = np.clip(a[1], -0.5, 0.5)
                state_, r, done, info = env.step(a)
                state = state_
                one_round_step += 1



                if done:
                    print('Step: %3i' % one_round_step, '| Collision!!!')
                    break


if __name__ == '__main__':
     main()