#!/usr/bin/env python
import keras.layers
import numpy as np
import gym
import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, BatchNormalization
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

# import wandb
import rospy
import walldodge_actor_env
from keras import backend as K

if __name__ == '__main__':
    rospy.init_node('husarion_wall_ddpg',
                    anonymous=True, log_level=rospy.DEBUG)

    # Create the Gym environment
    # Init OpenAI_ROS Husarion ENV
    task_and_robot_environment_name = 'Husarion_Walldodge-v1'
    env = gym.make(task_and_robot_environment_name)
    rospy.loginfo("Gym environment done")
    nb_actions = 2

    # Next, we build a very simple model.
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    inputs = Input(shape=(1, 40))
    a = Flatten()(inputs)
    a = Dense(512)(a)
    a = Activation("relu")(a)
    a = Dense(512)(a)
    a = Activation("relu")(a)
    a = Dense(512)(a)
    a = Activation("relu")(a)
    output1 = Dense(1, activation="sigmoid")(a)
    output2 = Dense(1, activation="tanh", kernel_initializer=last_init)(a)
    a = Concatenate()([output1, output2])
    actor = Model(inputs=inputs, outputs=a)
    print(actor.summary())
    keras.utils.plot_model(actor, "/home/marvin/Bilder/TestNetz2.png", show_shapes=True)

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1, 40), name='observation_input')
    flattened_observation = Flatten()(observation_input)

    x = Dense(512)(flattened_observation)
    x = Activation("relu")(x)
    x = Concatenate()([x, action_input])
    x = Dense(512)(x)
    x = Activation("relu")(x)
    x = Dense(512)(x)
    x = Activation("relu")(x)
    x = Dense(1)(x)
    x = Activation("linear")(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    print(critic.summary())

    memory = SequentialMemory(limit=100000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                      random_process=random_process, gamma=.99, target_model_update=1e-3)
    agent.compile(Adam(lr=0.001, clipnorm=1.), metrics=['mse'])

    agent.fit(env, nb_steps=20000, visualize=False, verbose=2)

    agent.save_weights('/home/marvin/ros_workspace/src/rosbot_openai/models/ddpg_20000_Steps.h5f', overwrite=True)
