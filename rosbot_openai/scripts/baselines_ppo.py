#!/usr/bin/env python
import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2, PPO1
import baselines_env
import rospy

def main():
    rospy.init_node('husarion_wall_ddpg',
                    anonymous=True, log_level=rospy.DEBUG)
    total_temp = 0
    succ_temp = 0

    # Create the Gym environment
    # Init OpenAI_ROS Husarion ENV
    task_and_robot_environment_name = 'Husarion_Walldodge-v1'
    env = gym.make(task_and_robot_environment_name)
    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("ppo_baselines")




if __name__ == '__main__':
     main()
