import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x = []
    for i in range(0,75):
        x=x+[10*i]
    #x = [0,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,420,440,460,480,500,520,540,560,580,600,620,640,660,680,700,720,740,760]
    print(len(x))

    y = [126,71,51,72,38,18,13,4,32,38,60,49,21,17,15,22,57,24,7,11,19,39,50,52,41,27,31,30,22,2,15,10,11,10,12,10,9,4,5,7,1,8,5,5,0,7,11,10,10,11,26,20,10,14,4,1,11,16,8,16,6,2,12,3,1,4,8,7,7,2,1,6,11,16,16]
    print(len(y))
    z = [0,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,3,3,0,0,3,0,0,0,0,1,1,0,0,3,0,0,0,0,0,1,0,2,0,0,2,2,0,0,2,1,2,3,1,0,1,0,2,1,0,0,2,1,2,1,1,3,0,0,0]
    print(len(z))
    plt.figure()
    plt.plot(x, y, label='total runs')
    plt.plot(x,z, label='successful runs')
    plt.xlabel('Training Steps in 1000')
    plt.ylabel('Anzahl Runs')
    plt.legend()
    plt.title('Training Runs')
    plt.grid(True)
    #Comment in and CHANGE the path, otherwise plot will be overwritten
    #plt.savefig("/home/marvin/ros_workspace/src/rosbot_openai/weights/ddpg_180_600k_steps/total_runs2.png")
    plt.show()