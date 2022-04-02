#!/usr/bin/env python
import argparse

import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.callbacks import ModelIntervalCheckpoint, FileLogger
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import wandb
import rospy
import walldodge_task

if __name__ == '__main__':
    rospy.init_node('husarion_wall_qlearn',
                    anonymous=True, log_level=rospy.DEBUG)


    # Create the Gym environment
    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = 'Husarion_Walldodge-v0'
    env = gym.make(task_and_robot_environment_name)
    rospy.loginfo("Gym environment done")
    rospy.loginfo("OBSERVATION DIMENSIONS: %s",env.observation_space.shape)
    rospy.loginfo("OBSERVATIONs: %s", env.observation_space)
    rospy.loginfo("ACTIONS : %s", env.action_space)

    # np.random.seed(123)
    # env.seed(123)
    nb_actions = env.action_space.n

    # Next, we build a very simple model.
    model = Sequential()
    #Dimensions of Flatten are (1, Length of Observations) Observations are the odometry (posx, posy, yaw)(3), the desired pos (posx,posy)(2) and the laser readings (30)
    model.add(Flatten(input_shape=(1, 25)))  # + env.observation_space.shape))
    #model.add(Flatten(input_shape=nb_obs))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())



    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, enable_dueling_network=True, dueling_type='avg',
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])


    def build_callbacks(task_and_robot_environment_name):
        checkpoint_weights_filename = 'dqn_' + task_and_robot_environment_name + '_weights_{step}.h5f'
        log_filename = '/home/marvin/ros_workspace/src/rosbot_openai/logs/dqn2_{}_log.json'.format(task_and_robot_environment_name)
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=10)]
        callbacks += [FileLogger(log_filename, interval=10)]
        return callbacks


    # callbacks = build_callbacks(task_and_robot_environment_name)


    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.

    #dqn.fit(env, callbacks=build_callbacks(task_and_robot_environment_name), nb_steps=10000, nb_max_episode_steps=300, visualize=False, verbose=2)

    # After training is done, we save the final weights.
    # dqn.save_weights('~/ros_workspace/src/rosbot_openai/models/dqn_{}_weights.h5f'.format(task_and_robot_environment_name), overwrite=True)
    #dqn.save_weights('/home/marvin/ros_workspace/src/rosbot_openai/models/dqn2_10000_Steps.h5f', overwrite=True)
    #print("TRAINING ONE DONE")
    dqn.load_weights('/home/marvin/ros_workspace/src/rosbot_openai/models/dqn2_10000_Steps.h5f')
    
    dqn.fit(env, nb_steps=20000, nb_max_episode_steps=300, visualize=False, verbose=2)
    
    dqn.save_weights('/home/marvin/ros_workspace/src/rosbot_openai/models/dqn2_30000_Steps.h5f', overwrite=True)
    print("TRAINING TWO DONE")
    dqn.load_weights('/home/marvin/ros_workspace/src/rosbot_openai/models/dqn2_30000_Steps.h5f')
    
    dqn.fit(env, nb_steps=20000, nb_max_episode_steps=300, visualize=False, verbose=2)
    
    dqn.save_weights('/home/marvin/ros_workspace/src/rosbot_openai/models/dqn2_50000_Steps.h5f', overwrite=True)
    
    print("TRAINING THREE DONE")

    dqn.load_weights('/home/marvin/ros_workspace/src/rosbot_openai/models/dqn2_50000_Steps.h5f')

    dqn.fit(env, nb_steps=20000, nb_max_episode_steps=300, visualize=False, verbose=2)

    dqn.save_weights('/home/marvin/ros_workspace/src/rosbot_openai/models/dqn2_70000_Steps.h5f', overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    #dqn.test(env, nb_episodes=5, visualize=False)'''
