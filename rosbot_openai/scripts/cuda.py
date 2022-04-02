'''import tensorflow as tf

print(tf.test.is_gpu_available())

x = 2.0
y = 3.0
timestep = 2000

file = open("/home/marvin/ros_workspace/src/rosbot_openai/logs/Rewards.txt", "a")

file.write("REACHED POSITION: "+ str(float(x))+" "+str(float(y))+" At Timestep: "+str(int(timestep))+"\n")
file.write("REACHED POSITION: "+ str(float(x))+" "+str(float(y))+" At Timestep: "+str(int(timestep))+"\n")
file.write("REACHED POSITION: "+ str(float(x))+" "+str(float(y))+" At Timestep: "+str(int(timestep))+"\n")

file.close()'''

import random
rand = random.randint(0,5)
print(rand)