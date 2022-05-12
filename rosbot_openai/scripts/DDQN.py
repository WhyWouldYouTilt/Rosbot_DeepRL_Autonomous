#!/usr/bin/env python
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import rospy
import husarion_environment

ENV_NAME = 'Husarion_Walldodge-v1'


def main():
    total_temp = 0
    succ_temp = 0
    coll_temp = 0
    rospy.init_node('husarion_wall_ddqn_learn',
                    anonymous=True, log_level=rospy.DEBUG)

    # Create the Gym environment
    # Init OpenAI_ROS ENV
    env = gym.make(ENV_NAME)
    nb_actions = env.action_space.n

    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions, activation='linear'))

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    # enable the dueling network
    # you can specify the dueling_type to one of {'avg','max','naive'}
    ddqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                   enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)
    ddqn.compile(Adam(lr=1e-3), metrics=['mae'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    #50K Steps
    ddqn.fit(env, nb_steps=50000, visualize=False, verbose=2)
    ddqn.save_weights('/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/ddqn_weights_50000_steps.h5f', overwrite=True)

    runs_total = env.total_runs - total_temp
    runs_succ = env.successful_runs - succ_temp
    runs_collisions = env.collisions - coll_temp
    file = open("/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/logs/total_runs.txt", "a")
    file.write(str(runs_total) + ",")
    file.close()
    file = open("/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/logs/succ_runs.txt", "a")
    file.write(str(runs_succ) + ",")
    file.close()
    file = open("/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/logs/collisions.txt", "a")
    file.write(str(runs_collisions) + ",")
    file.close()
    total_temp = env.total_runs
    succ_temp = env.successful_runs
    coll_temp = env.collisions
    #100k Steps
    ddqn.load_weights('/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/ddqn_weights_50000_steps.h5f')
    ddqn.fit(env, nb_steps=50000, visualize=False, verbose=2)
    ddqn.save_weights('/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/ddqn_weights_100000_steps.h5f',
                     overwrite=True)

    runs_total = env.total_runs - total_temp
    runs_succ = env.successful_runs - succ_temp
    runs_collisions = env.collisions - coll_temp
    file = open("/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/logs/total_runs.txt", "a")
    file.write(str(runs_total) + ",")
    file.close()
    file = open("/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/logs/succ_runs.txt", "a")
    file.write(str(runs_succ) + ",")
    file.close()
    file = open("/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/logs/collisions.txt", "a")
    file.write(str(runs_collisions) + ",")
    file.close()
    total_temp = env.total_runs
    succ_temp = env.successful_runs
    coll_temp = env.collisions
    #150k Steps
    ddqn.load_weights('/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/ddqn_weights_100000_steps.h5f')
    ddqn.fit(env, nb_steps=50000, visualize=False, verbose=2)
    ddqn.save_weights('/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/ddqn_weights_150000_steps.h5f',
                     overwrite=True)

    runs_total = env.total_runs - total_temp
    runs_succ = env.successful_runs - succ_temp
    runs_collisions = env.collisions - coll_temp
    file = open("/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/logs/total_runs.txt", "a")
    file.write(str(runs_total) + ",")
    file.close()
    file = open("/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/logs/succ_runs.txt", "a")
    file.write(str(runs_succ) + ",")
    file.close()
    file = open("/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/logs/collisions.txt", "a")
    file.write(str(runs_collisions) + ",")
    file.close()
    total_temp = env.total_runs
    succ_temp = env.successful_runs
    coll_temp = env.collisions
    #200k Steps
    ddqn.load_weights('/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/ddqn_weights_150000_steps.h5f')
    ddqn.fit(env, nb_steps=50000, visualize=False, verbose=2)
    ddqn.save_weights('/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/ddqn_weights_200000_steps.h5f',
                     overwrite=True)

    runs_total = env.total_runs - total_temp
    runs_succ = env.successful_runs - succ_temp
    runs_collisions = env.collisions - coll_temp
    file = open("/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/logs/total_runs.txt", "a")
    file.write(str(runs_total) + ",")
    file.close()
    file = open("/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/logs/succ_runs.txt", "a")
    file.write(str(runs_succ) + ",")
    file.close()
    file = open("/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/logs/collisions.txt", "a")
    file.write(str(runs_collisions) + ",")
    file.close()
    total_temp = env.total_runs
    succ_temp = env.successful_runs
    coll_temp = env.collisions
    # 250k Steps
    ddqn.load_weights('/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/ddqn_weights_200000_steps.h5f')
    ddqn.fit(env, nb_steps=50000, visualize=False, verbose=2)
    ddqn.save_weights('/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/ddqn_weights_250000_steps.h5f',
                      overwrite=True)

    runs_total = env.total_runs - total_temp
    runs_succ = env.successful_runs - succ_temp
    runs_collisions = env.collisions - coll_temp
    file = open("/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/logs/total_runs.txt", "a")
    file.write(str(runs_total) + ",")
    file.close()
    file = open("/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/logs/succ_runs.txt", "a")
    file.write(str(runs_succ) + ",")
    file.close()
    file = open("/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/logs/collisions.txt", "a")
    file.write(str(runs_collisions) + ",")
    file.close()
    total_temp = env.total_runs
    succ_temp = env.successful_runs
    coll_temp = env.collisions
    # 300k Steps
    ddqn.load_weights('/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/ddqn_weights_250000_steps.h5f')
    ddqn.fit(env, nb_steps=50000, visualize=False, verbose=2)
    ddqn.save_weights('/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/ddqn_weights_300000_steps.h5f',
                      overwrite=True)

    runs_total = env.total_runs - total_temp
    runs_succ = env.successful_runs - succ_temp
    runs_collisions = env.collisions - coll_temp
    file = open("/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/logs/total_runs.txt", "a")
    file.write(str(runs_total) + ",")
    file.close()
    file = open("/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/logs/succ_runs.txt", "a")
    file.write(str(runs_succ) + ",")
    file.close()
    file = open("/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/logs/collisions.txt", "a")
    file.write(str(runs_collisions) + ",")
    file.close()
    total_temp = env.total_runs
    succ_temp = env.successful_runs
    coll_temp = env.collisions

    # 3500k Steps
    ddqn.load_weights('/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/ddqn_weights_300000_steps.h5f')
    ddqn.fit(env, nb_steps=50000, visualize=False, verbose=2)
    ddqn.save_weights('/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/ddqn_weights_3500000_steps.h5f',
                      overwrite=True)

    runs_total = env.total_runs - total_temp
    runs_succ = env.successful_runs - succ_temp
    runs_collisions = env.collisions - coll_temp
    file = open("/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/logs/total_runs.txt", "a")
    file.write(str(runs_total) + ",")
    file.close()
    file = open("/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/logs/succ_runs.txt", "a")
    file.write(str(runs_succ) + ",")
    file.close()
    file = open("/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/logs/collisions.txt", "a")
    file.write(str(runs_collisions) + ",")
    file.close()
    total_temp = env.total_runs
    succ_temp = env.successful_runs
    coll_temp = env.collisions

    # 400k Steps
    ddqn.load_weights('/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/ddqn_weights_350000_steps.h5f')
    ddqn.fit(env, nb_steps=50000, visualize=False, verbose=2)
    ddqn.save_weights('/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/ddqn_weights_400000_steps.h5f',
                      overwrite=True)

    runs_total = env.total_runs - total_temp
    runs_succ = env.successful_runs - succ_temp
    runs_collisions = env.collisions - coll_temp
    file = open("/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/logs/total_runs.txt", "a")
    file.write(str(runs_total) + ",")
    file.close()
    file = open("/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/logs/succ_runs.txt", "a")
    file.write(str(runs_succ) + ",")
    file.close()
    file = open("/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/logs/collisions.txt", "a")
    file.write(str(runs_collisions) + ",")
    file.close()
    total_temp = env.total_runs
    succ_temp = env.successful_runs
    coll_temp = env.collisions

    # 450k Steps
    ddqn.load_weights('/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/ddqn_weights_400000_steps.h5f')
    ddqn.fit(env, nb_steps=50000, visualize=False, verbose=2)
    ddqn.save_weights('/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/ddqn_weights_450000_steps.h5f',
                      overwrite=True)

    runs_total = env.total_runs - total_temp
    runs_succ = env.successful_runs - succ_temp
    runs_collisions = env.collisions - coll_temp
    file = open("/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/logs/total_runs.txt", "a")
    file.write(str(runs_total) + ",")
    file.close()
    file = open("/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/logs/succ_runs.txt", "a")
    file.write(str(runs_succ) + ",")
    file.close()
    file = open("/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/logs/collisions.txt", "a")
    file.write(str(runs_collisions) + ",")
    file.close()
    total_temp = env.total_runs
    succ_temp = env.successful_runs
    coll_temp = env.collisions

    # 500k Steps
    ddqn.load_weights('/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/ddqn_weights_450000_steps.h5f')
    ddqn.fit(env, nb_steps=50000, visualize=False, verbose=2)
    ddqn.save_weights('/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/ddqn_weights_500000_steps.h5f',
                      overwrite=True)

    runs_total = env.total_runs - total_temp
    runs_succ = env.successful_runs - succ_temp
    runs_collisions = env.collisions - coll_temp
    file = open("/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/logs/total_runs.txt", "a")
    file.write(str(runs_total) + ",")
    file.close()
    file = open("/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/logs/succ_runs.txt", "a")
    file.write(str(runs_succ) + ",")
    file.close()
    file = open("/home/marvin/ros_workspace/src/rosbot_openai/ddqn_models/logs/collisions.txt", "a")
    file.write(str(runs_collisions) + ",")
    file.close()
    total_temp = env.total_runs
    succ_temp = env.successful_runs
    coll_temp = env.collisions


















if __name__ == '__main__':
    main()