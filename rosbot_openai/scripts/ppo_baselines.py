#!/usr/bin/env python
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import rospy
#import husarion_environment
import ppo_baselines_env
def main():
    rospy.init_node('husarion_wall_ppo',
                    anonymous=True, log_level=rospy.DEBUG)
    # Create the Gym environment
    # Init OpenAI_ROS Husarion ENV
    env_name = 'Husarion_Walldodge-v1'
    env = make_vec_env(env_name, n_envs=1)

    model = PPO("MlpPolicy", env, verbose=1) #tensorboard_log="/home/marvin/ros_workspace/src/rosbot_openai/ppo_discrete_models/logs")
    model.learn(total_timesteps=100000)
    model.save("/home/marvin/ros_workspace/src/rosbot_openai/ppo_baselines_models/rosbot_100k_steps")
    model.load("/home/marvin/ros_workspace/src/rosbot_openai/ppo_baselines_models/rosbot_100k_steps")
    #evaluate_policy(model=model, env=env, n_eval_episodes=10)
    model.learn(total_timesteps=100000)
    model.save("/home/marvin/ros_workspace/src/rosbot_openai/ppo_baselines_models/rosbot_200k_steps")




































if __name__ == '__main__':
     main()