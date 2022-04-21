import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x = []
    for i in range(0,72):
        x=x+[10*i]
    print(len(x))

    #y = [7,1,86,45,21,16,25,125,56,42,65,50,21,14,21,20,40,24,11,22,14,33,30,23,16,34,36,35,37,34,38,34,36,40,37,37,38,28,32,26,7,32,30,38,35,44,46,40,40,43,43,42,41,34,32,36,41,40,34,36,29,43,43,44,44,37,32,41,37,37,38,47]
    #print(len(y))
    z = [0,0,0,4,3,0,0,0,0,4,3,3,5,2,3,2,4,7,4,8,4,13,8,5,3,3,7,9,6,9,16,15,17,20,15,18,17,13,19,6,1,12,10,19,15,11,23,12,17,22,13,18,14,15,13,18,18,22,13,13,13,19,8,17,17,16,13,22,19,19,19,26]
    print(len(z))
    plt.figure()
    #plt.plot(x, y, label='total runs')
    plt.plot(x,z, label='Successful runs')
    plt.xlabel('Training Steps in 1000')
    plt.ylabel('Runs')
    plt.legend()
    plt.title('Training runs')
    plt.grid(True)
    #Comment in and CHANGE the path, otherwise plot will be overwritten
    plt.savefig("/home/marvin/ros_workspace/src/rosbot_openai/weights/ddpg_new/succ_runs_700k.png")
    plt.show()