import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x = []
    for i in range(0,51):
        x=x+[10*i]
    print(len(x))


    y = [-23662,9710,37363,49369,79768,82345,94500,103327,81208,61518,81875,79671,81140,73782,87694,99094,97793,111765,114019,106905,110162,120682,102578,102297,127381,121166,97038,113033,122122,100841,107397,118459,106744,122243,106488,104262,109857,102642,109523,119835,121127,128699,116952,126221,116394,120935,125521,106386,121794,109080,120569,]
    print(len(y))
    """succ_percent = [1.0,1.0,13.0,23.0,28.0,30.0,38.0,36.0,21.0,33.0,27.0,39.0,31.0,33.0,37.0,40.0,39.0,43.0,40.0,47.0,45.0,56.0,38.0,44.0,49.0,57.0,37.0,43.0,56.0,39.0,47.0,49.0,42.0,44.0,56.0,44.0,55.0,38.0,46.0,51.0,49.0,58.0,51.0,57.0,50.0,48.0,62.0,43.0,50.0,48.0,49.0]
    for i in range(0, len(y)):
        succ_percent[i]=round((succ_percent[i]/y[i])*100,2)"""

    #print("succ: %s", succ_percent)
    #z = []
    plt.figure()
    plt.plot(x, y, label='Total reward')
    #plt.plot(x,z, label='Successful runs')
    plt.xlabel('Training Steps in 1000')
    plt.ylabel('Total reward')
    plt.legend()
    plt.title('Curriculum Learning over 4 Tasks')
    plt.grid(True)
    #Comment in and CHANGE the path, otherwise plot will be overwritten
    plt.savefig("/home/marvin/ros_workspace/src/rosbot_openai/weights/ppo_curriculum/500k_steps/total_reward_500k.png")
    plt.show()


    """#Uncomment for Curriculum Learning plots
    task_1 = [4.0,1.0,0.0,22.0,7.0,20.0,19.0,13.0,0.0,40.0,4.0,23.0,14.0,17.0,12.0,12.0,7.0,21.0,0.0,37.0,17.0,19.0,9.0,21.0,6.0,25.0,7.0,10.0,28.0,6.0,18.0,18.0,0.0,0.0,48.0,12.0,23.0,1.0,16.0,14.0,20.0,13.0,15.0,20.0,12.0,0.0,36.0,2.0,23.0,18.0,19.0]
    task_2 = [2.0,0.0,10.0,8.0,23.0,5.0,26.0,20.0,22.0,4.0,24.0,13.0,9.0,17.0,14.0,1.0,20.0,17.0,28.0,2.0,15.0,29.0,4.0,0.0,35.0,21.0,9.0,20.0,17.0,6.0,15.0,18.0,16.0,23.0,2.0,0.0,12.0,35.0,7.0,23.0,8.0,30.0,9.0,16.0,13.0,21.0,18.0,15.0,14.0,16.0,19.0]
    task_3 = [5.0,1.0,13.0,1.0,15.0,26.0,6.0,28.0,23.0,1.0,25.0,4.0,14.0,0.0,12.0,34.0,14.0,16.0,24.0,8.0,19.0,18.0,12.0,20.0,9.0,0.0,30.0,10.0,20.0,14.0,16.0,16.0,19.0,21.0,11.0,13.0,22.0,3.0,15.0,12.0,20.0,4.0,25.0,16.0,18.0,20.0,14.0,14.0,15.0,12.0,17.0]
    task_4 = [59.0,49.0,17.0,0.0,2.0,3.0,3.0,0.0,3.0,1.0,0.0,18.0,14.0,14.0,13.0,13.0,14.0,19.0,6.0,26.0,11.0,0.0,33.0,17.0,11.0,22.0,10.0,20.0,0.0,29.0,14.0,12.0,20.0,19.0,2.0,30.0,5.0,14.0,18.0,11.0,19.0,21.0,10.0,14.0,19.0,19.0,1.0,27.0,14.0,18.0,13.0]
    print(len(task_4))
    plt.figure()
    plt.plot(x, task_1, label='Task 1')
    plt.plot(x, task_2, label='Task 2')
    plt.plot(x, task_3, label='Task 3')
    plt.plot(x, task_4, label='Task 4')
    plt.xlabel('Training Steps in 1000')
    plt.ylabel('Taskcount')
    plt.legend()
    plt.title('Curriculum Learning Tasks')
    plt.grid(True)
    # Comment in and CHANGE the path, otherwise plot will be overwritten
    plt.savefig("/home/marvin/ros_workspace/src/rosbot_openai/weights/ppo_curriculum/500k_steps/Task_Verteilung_500k.png")
    plt.show()"""

