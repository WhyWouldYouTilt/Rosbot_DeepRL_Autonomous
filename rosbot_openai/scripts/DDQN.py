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
import ddqn_environment

task_and_robot_environment_name = 'Husarion_Walldodge-v2'


def main():
    rospy.init_node('husarion_wall_ddqn_earn',
                    anonymous=True, log_level=rospy.DEBUG)

    # Create the Gym environment
    # Init OpenAI_ROS ENV
    env = gym.make(task_and_robot_environment_name)
    nb_actions = 11


    model = Sequential()
    model.add(Flatten(input_shape=(1,22)))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions, activation='linear'))
    print(model.summary())

    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()

    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])



    dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)

    dqn.save_weights('/home/marvin/ros_workspace/src/rosbot_openai/models/ddqn_50000_Steps.h5f', overwrite=True)
    runs_total = env.total_runs
    runs_succ = env.successful_runs
    file = open("/home/marvin/ros_workspace/src/rosbot_openai/logs/runs.txt", "a")
    file.write("TOTAL RUNS:" + str(runs_total) + "\n")
    file.write("SUCCESSFUL RUNS:" + str(runs_succ) + "\n")
    file.write("FINISHED FIRST 50000 TRAININGSSTEPS" + "\n")
    file.close()

    dqn.load_weights('/home/marvin/ros_workspace/src/rosbot_openai/models/ddqn_50000_Steps.h5f')
    dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)
    dqn.save_weights('/home/marvin/ros_workspace/src/rosbot_openai/models/ddqn_100000_Steps.h5f', overwrite=True)
    runs_total = env.total_runs
    runs_succ = env.successful_runs
    file = open("/home/marvin/ros_workspace/src/rosbot_openai/logs/runs.txt", "a")
    file.write("TOTAL RUNS:" + str(runs_total) + "\n")
    file.write("SUCCESSFUL RUNS:" + str(runs_succ) + "\n")
    file.write("FINISHED FIRST 100000 TRAININGSSTEPS" + "\n")
    file.close()
    dqn.load_weights('/home/marvin/ros_workspace/src/rosbot_openai/models/ddqn_100000_Steps.h5f')
    dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)
    dqn.save_weights('/home/marvin/ros_workspace/src/rosbot_openai/models/ddqn_150000_Steps.h5f', overwrite=True)
    runs_total = env.total_runs
    runs_succ = env.successful_runs
    file = open("/home/marvin/ros_workspace/src/rosbot_openai/logs/runs.txt", "a")
    file.write("TOTAL RUNS:" + str(runs_total) + "\n")
    file.write("SUCCESSFUL RUNS:" + str(runs_succ) + "\n")
    file.write("FINISHED FIRST 150000 TRAININGSSTEPS" + "\n")
    file.close()
    dqn.load_weights('/home/marvin/ros_workspace/src/rosbot_openai/models/ddqn_150000_Steps.h5f')
    dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)
    dqn.save_weights('/home/marvin/ros_workspace/src/rosbot_openai/models/ddqn_200000_Steps.h5f', overwrite=True)
    runs_total = env.total_runs
    runs_succ = env.successful_runs
    file = open("/home/marvin/ros_workspace/src/rosbot_openai/logs/runs.txt", "a")
    file.write("TOTAL RUNS:" + str(runs_total) + "\n")
    file.write("SUCCESSFUL RUNS:" + str(runs_succ) + "\n")
    file.write("FINISHED FIRST 200000 TRAININGSSTEPS" + "\n")
    file.close()
    dqn.load_weights('/home/marvin/ros_workspace/src/rosbot_openai/models/ddqn_200000_Steps.h5f')
    dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)
    dqn.save_weights('/home/marvin/ros_workspace/src/rosbot_openai/models/ddqn_250000_Steps.h5f', overwrite=True)
    runs_total = env.total_runs
    runs_succ = env.successful_runs
    file = open("/home/marvin/ros_workspace/src/rosbot_openai/logs/runs.txt", "a")
    file.write("TOTAL RUNS:" + str(runs_total) + "\n")
    file.write("SUCCESSFUL RUNS:" + str(runs_succ) + "\n")
    file.write("FINISHED FIRST 500000 TRAININGSSTEPS" + "\n")
    file.close()
    dqn.load_weights('/home/marvin/ros_workspace/src/rosbot_openai/models/ddqn_250000_Steps.h5f')
    dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)
    dqn.save_weights('/home/marvin/ros_workspace/src/rosbot_openai/models/ddqn_300000_Steps.h5f', overwrite=True)
    runs_total = env.total_runs
    runs_succ = env.successful_runs
    file = open("/home/marvin/ros_workspace/src/rosbot_openai/logs/runs.txt", "a")
    file.write("TOTAL RUNS:" + str(runs_total) + "\n")
    file.write("SUCCESSFUL RUNS:" + str(runs_succ) + "\n")
    file.write("FINISHED FIRST 300000 TRAININGSSTEPS" + "\n")
    file.close()
    dqn.load_weights('/home/marvin/ros_workspace/src/rosbot_openai/models/ddqn_300000_Steps.h5f')
    dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)
    dqn.save_weights('/home/marvin/ros_workspace/src/rosbot_openai/models/ddqn_350000_Steps.h5f', overwrite=True)
    runs_total = env.total_runs
    runs_succ = env.successful_runs
    file = open("/home/marvin/ros_workspace/src/rosbot_openai/logs/runs.txt", "a")
    file.write("TOTAL RUNS:" + str(runs_total) + "\n")
    file.write("SUCCESSFUL RUNS:" + str(runs_succ) + "\n")
    file.write("FINISHED FIRST 350000 TRAININGSSTEPS" + "\n")
    file.close()



















if __name__ == '__main__':
    main()