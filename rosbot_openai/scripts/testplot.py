import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x = []
    for i in range(0,106):
        x=x+[10*i]
    print(len(x))

    y = [-167645,-100619,-47043,-179125,-131914,-123380,-193866,-173582,-140936,-119896,-22617,-6029,-8908,-22312,-29317,-36388,-37803,-19628,-28814,-22055,737,-4621,6709,47286,54037,15929,59236,37964,37376,36385,89124,34557,70562,41780,140792,142290,94637,79938,63608,130530,24706,12693,39363,46522,50321,71989,86112,123288,100911,87846,78721,66657,85872,89368,106209,113901,138007,76264,82517,95702,100739,55398,44815,91322,124535,112934,110896,114086,125996,123670,102163,117021,122398,175613,168129,155017,132666,171228,147294,79687,122442,148493,78775,33497,98041,58904,81388,83299,119803,78194,95670,120495,111860,138907,114408,101731,26107,37165,53394,55879,63645,65078,79453,104047,83945,77169]
    print(len(y))
    #z = []
    plt.figure()
    plt.plot(x, y, label='Total reward')
    #plt.plot(x,z, label='Successful runs')
    plt.xlabel('Training Steps in 1000')
    plt.ylabel('Reward')
    plt.legend()
    plt.title('Simple world rewards')
    plt.grid(True)
    #Comment in and CHANGE the path, otherwise plot will be overwritten
    plt.savefig("/home/marvin/ros_workspace/src/rosbot_openai/weights/ddpg_continuous/simple_world/simple_world_rewards_1Mil.png")
    plt.show()