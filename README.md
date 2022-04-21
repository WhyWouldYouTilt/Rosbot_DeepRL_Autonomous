# Rosbot_autonomous
 Deep RL for autonomous driving with Rosbot

The Project uses the openai_ros Package to connect Gazebo and Deep Reinforcement Learning:
http://wiki.ros.org/openai_ros

The Environment for DRL is in scripts/walldodge_actor_env.py
The DDPG Algorithm is implemented with the actor_network_bn.py, critiy_network.py and the replay_buffer.py

The Project is based on the Paper: "Virtual-to-real Deep Reinforcement Learning: Continuous Control of Mobile Robots for Mapless Navigation"
https://arxiv.org/pdf/1703.00420.pdf

and needs Python 2.7 and TF-GPU






Required Packages:

depth_image_to_laserscan (RGBD to Laserscan converter)
hector_slam (For trajectory plotting)
openai_ros (Connection between Husarion Rosbot and Gazebo Simulation)
rosbot_description (Files for the Rosbot)
