#!/usr/bin/env python2.7
from operator import neg

import random
from sklearn import preprocessing
import math
import rospy
import numpy
from gym import spaces
from openai_ros.robot_envs import husarion_env
from gym.envs.registration import register
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point, Pose
#from tf.transformations import euler_from_quaternion
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
from nav_msgs.msg import Odometry

from interactive_markers.interactive_marker_server import *
from visualization_msgs.msg import *

# The path is __init__.py of openai_ros, where we import the SumitXlMazeEnv directly
timestep_limit_per_episode = 5000  # Can be any Value
diagonal_dis = 10

register(
    id='Husarion_Walldodge-v1',
    entry_point='husarion_environment:HusarionWalldodgeEnv',
    max_episode_steps=timestep_limit_per_episode,
)


class HusarionWalldodgeEnv(husarion_env.HusarionEnv):
    def __init__(self):
        """
        This Task Env is designed for having the husarion in the husarion world
        closed room with columns.
        It will learn how to move around a wall to a desired point without crashing.
        """

        # Only variable needed to be set here
        self.successful_runs = 0
        self.total_runs = 0
        self.collisions =0
        self.reached_count=0
        self.yaw = None
        self.rel_theta = None
        self.diff_angle = None
        self.prev_diff_angle = None
        number_actions = 3
        self.action_space = spaces.Discrete(number_actions)

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.odom_pub = rospy.Publisher('odom', Odometry, queue_size=10)


        self.max_laser_value = 5.0
        self.min_laser_value = 0.2
        self.workspace_x_max = 10.0
        self.workspace_x_min = -10.0
        self.workspace_y_max = 10.0
        self.workspace_y_min = -10.0

        # Obs_shape for 28 Lasescan
        high_laser = numpy.full(28, self.max_laser_value)
        low_laser = numpy.full(28, self.min_laser_value)
        # Obs_shape for previous actions
        """high_act = numpy.array([1.0, 1.0])
        low_act = numpy.array([0.0, -1.0])"""
        # Obs_shape for desired position in polarcoordinates (distance and angle)
        high_des_pos = numpy.array([self.workspace_x_max+self.workspace_x_max, 360])
        low_des_pos = numpy.array([0, 0])

        high = numpy.concatenate([high_laser, high_des_pos])
        low = numpy.concatenate([low_laser, low_des_pos])
        self.observation_space = spaces.Box(low, high)

        # Get Desired Point to start with
        self.desired_position = Point()
        self.desired_position.x = -4.0  # rospy.get_param("/husarion/desired_pose/x")
        self.desired_position.y = -4.0  # rospy.get_param("/husarion/desired_pose/y")

        # Get Robot Position and yaw
        self.position = Pose()
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)

        # Rewardfunction
        self.end_episode_points = 200.0
        self.closer_to_point_param = 10.0
        self.collision_reward = -200.0

        self.cumulated_steps = 0.0


        # Here we will add any init functions prior to starting the MyRobotEnv
        super(HusarionWalldodgeEnv, self).__init__()

    def get_past_angular_action(self):
        return self.past_angular

    def get_past_linear_action(self):
        return self.past_linear


    # Get the odom position
    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        # We get the orientation of the cube in RPY
        q_x, q_y, q_z, q_w = orientation.x, orientation.y, orientation.z, orientation.w
        yaw = round(math.degrees(math.atan2(2 * (q_x * q_y + q_w * q_z), 1 - 2 * (q_y * q_y + q_z * q_z))))

        if yaw >= 0:
            yaw = yaw
        else:
            yaw = yaw + 360

        rel_dis_x = round(self.desired_position.x - self.position.x, 1)
        rel_dis_y = round(self.desired_position.y - self.position.y, 1)


        # Calculate the angle between robot and target
        if rel_dis_x > 0 and rel_dis_y > 0:
            theta = math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x > 0 and rel_dis_y < 0:
            theta = 2 * math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y < 0:
            theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y > 0:
            theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x == 0 and rel_dis_y > 0:
            theta = 1 / 2 * math.pi
        elif rel_dis_x == 0 and rel_dis_y < 0:
            theta = 3 / 2 * math.pi
        elif rel_dis_y == 0 and rel_dis_x > 0:
            theta = 0
        else:
            theta = math.pi
        rel_theta = round(math.degrees(theta), 2)


        diff_angle = abs(rel_theta - yaw)

        if diff_angle <= 180:
            diff_angle = round(diff_angle, 2)
        else:
            diff_angle = round(360 - diff_angle, 2)


        self.rel_theta = rel_theta
        self.yaw = yaw
        self.diff_angle = diff_angle


    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        init_pose = Odometry()
        init_pose.pose.pose.position.x = 0.0
        init_pose.pose.pose.position.y = 0.0

        self.odom_pub.publish(init_pose)
        #init_pose = self.get_odom()
        self.total_runs += 1




        return True

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes
        self.cumulated_reward = 0.0

        self.index = 0

        odometry = self.get_odom()
        self.previous_distance_from_des_point = self.get_distance_from_desired_point(odometry.pose.pose.position,
                                                                                     self.desired_position)
        #self.prev_diff_angle = self.diff_angle

    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the SumitXl
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """

        if action == 0:  # Forward
            linear_vel = 0.15
            angular_vel = 0.0
        elif action == 1:  # Left
            linear_vel = 0.0
            angular_vel = -0.18
        elif action == 2:
            linear_vel = 0.0
            angular_vel = 0.18

        self.past_linear = linear_vel
        self.past_angular = angular_vel

        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel
        vel_cmd.angular.z = angular_vel
        self.pub_cmd_vel.publish(vel_cmd)



    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        HusarionEnv API DOCS
        :return:
        """
        # rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data

        laser_scan = self.get_laser_scan()
        discretized_laser_scan = self.discretize_scan_210(laser_scan)
        max_scan = max(discretized_laser_scan)
        for i in range(len(discretized_laser_scan)):
            discretized_laser_scan[i] /= max_scan


        odometry = self.get_odom()
        self.getOdometry(odometry)
        yaw = self.yaw
        rel_theta = self.rel_theta


        current_x = self.get_current_pos_x()
        current_y = self.get_current_pos_y()
        current_distance = math.sqrt((self.desired_position.x-current_x)*(self.desired_position.x-current_x)+(self.desired_position.y-current_y)*(self.desired_position.y-current_y))
        # Get the linear and angular velocity from past action
        #past_lin = self.get_past_linear_action()
        #past_ang = self.get_past_angular_action()
        observations = discretized_laser_scan + [current_distance/diagonal_dis] + [rel_theta/360]
        """print("Distance: %s", current_distance/diagonal_dis)
        print("Distance: %s", rel_theta / 360)"""
        print("OBS: %s", observations)
        return observations

    def get_current_pos_x(self):
        odometry = self.get_odom()
        x_position = odometry.pose.pose.position.x
        return x_position

    def get_current_pos_y(self):
        odometry = self.get_odom()
        y_position = odometry.pose.pose.position.y
        return y_position



    def _is_done(self, observations):
        """
        We consider that the episode has finished when:
        1) Husarion is too close to an object
        2) Husarion has reached the desired position
        """

        # We fetch data through the observations
        # Its all the array except from the last four elements, which are XY odom and XY des_pos
        # laser_readings = observations[:-5]
        laser_readings = observations[:-2]

        current_position = Point()
        current_position.x = self.get_current_pos_x()
        current_position.y = self.get_current_pos_y()
        current_position.z = 0.0

        desired_position = Point()
        desired_position.x = self.desired_position.x
        desired_position.y = self.desired_position.y
        desired_position.z = 0.0


        # too_close_to_object = self.check_husarion_has_crashed(laser_readings)
        too_close_to_object = self.check_husarion_has_crashed(laser_readings)
        reached_des_pos = self.check_reached_desired_position(current_position,
                                                              desired_position)

        # is_done = too_close_to_object or not (inside_workspace) or reached_des_pos
        is_done = too_close_to_object  or reached_des_pos

        """#Training for the difficult_world:
        if reached_des_pos and self.reached_count >= 3:
            rand_number = random.randint(0, 7)
            file = open("/home/marvin/ros_workspace/src/rosbot_openai/logs/Reached.txt", "a")
            file.write("REACHED POSITION: " + str(float(self.desired_position.x)) + " " + str(
                float(self.desired_position.y)) + "\n")
            file.close()
            if rand_number == 0:
                self.update_desired_pos(-3.0, -1.0)
            elif rand_number == 1:
                self.update_desired_pos(-2.0, -3.0)
            elif rand_number == 2:
                self.update_desired_pos(1.0, -3.0)
            elif rand_number == 3:
                self.update_desired_pos(2.0, -4.0)
            elif rand_number == 4:
                self.update_desired_pos(3.0, 3.0)
            elif rand_number == 5:
                self.update_desired_pos(-3.0, 4.0)
            elif rand_number == 6:
                self.update_desired_pos(4.0, 1.15)
            elif rand_number == 7:
                self.update_desired_pos(-4.0, -4.0)
            self.reached_count = 0"""



        """#Uncomment for testing purposes. You can plan your route here:
        if reached_des_pos and self.reached_count==1:
            self.update_desired_pos(-7.0,-4.0)
        elif reached_des_pos and self.reached_count==2:
            self.update_desired_pos(-4.0,4.0)
        elif reached_des_pos and self.reached_count==3:
            self.update_desired_pos(0.0,-6.0)
        elif reached_des_pos and self.reached_count==4:
            self.update_desired_pos(5.0,3.0)"""


        #Uncomment for training for the simple world
        if reached_des_pos and self.reached_count>=3:
            rand_number = random.randint(0, 8)
            file = open("/home/marvin/ros_workspace/src/rosbot_openai/logs/Reached.txt", "a")
            file.write("REACHED POSITION: "+ str(float(self.desired_position.x))+" "+str(float(self.desired_position.y))+"\n")
            file.close()
            if rand_number == 0:
                self.update_desired_pos(-3.0, -3.0)
            elif rand_number == 1:
                self.update_desired_pos(-4.0, 0.0)
            elif rand_number == 2:
                self.update_desired_pos(-4.0, 4.0)
            elif rand_number == 3:
                self.update_desired_pos(0.0, -3.0)
            elif rand_number == 4:
                self.update_desired_pos(0.0, 4.0)
            elif rand_number == 5:
                self.update_desired_pos(4.0, -4.0)
            elif rand_number == 6:
                self.update_desired_pos(4.0, 1.0)
            elif rand_number == 7:
                self.update_desired_pos(-4.0, -2.0)
            elif rand_number == 8:
                self.update_desired_pos(-4.0, 2.0)
            self.reached_count=0


        return is_done

    def _compute_reward(self, observations, done):
        """
        We will reward the following behaviours:
        1) The distance to the desired point has increase from last step
        2) The robot has reached the desired point

        We will penalise the following behaviours:
        1) Ending the episode without reaching the desired pos. That means it has crashed
        or it has gone outside the workspace

        """

        current_position = Point()
        current_position.x = self.get_current_pos_x()
        current_position.y = self.get_current_pos_y()
        current_position.z = 0.0

        print("Current POS: %s", current_position)

        desired_position = Point()
        desired_position.x = self.desired_position.x
        desired_position.y = self.desired_position.y
        desired_position.z = 0.0
        print("Desired POS: %s", desired_position)

        distance_from_des_point = self.get_distance_from_desired_point(current_position, desired_position)

        print("Distance: %s", distance_from_des_point)

        distance_difference = self.previous_distance_from_des_point - distance_from_des_point

        if not done:
            # If there has been a decrease in the distance to the desired point, we reward it
            reward = self.closer_to_point_param * distance_difference
        else:
            reached_des_pos = self.check_reached_desired_position(current_position,
                                                                  desired_position)
            if reached_des_pos:
                reward = self.end_episode_points
                self.successful_runs += 1
                # rospy.logwarn("GOT TO DESIRED POINT ; DONE, reward=" + str(reward))
            else:
                reward = self.collision_reward
                self.collisions +=1
                #rospy.logerr("SOMETHING WENT WRONG ; DONE, reward=" + str(reward))
        self.previous_distance_from_des_point = distance_from_des_point
        #self.prev_diff_angle = diff_angle

        rospy.logwarn("reward=" + str(reward))
        self.cumulated_reward += reward
        #rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))
        return reward

    # Internal TaskEnv Methods
    def update_desired_pos(self, x, y):
        """
        With this method you can change the desired position that you want
        Usarion to be that initialy is set through rosparams loaded through
        a yaml file possibly.
        :new_position: Type Point, because we only value the position.
        """
        self.desired_position.x = x
        self.desired_position.y = y

        file = open("/home/marvin/ros_workspace/src/rosbot_openai/logs/Change.txt", "a")

        file.write(
            "Changed Position to: " + str(float(self.desired_position.x)) + " " + str(float(self.desired_position.y))+"\n")
        file.close()


    def discretize_scan_210(self, data):
        discretized_ranges = []
        for i in range(0, 280, 20):
            temp_list = []
            for j in range(0, 20):
                temp_list.append(data.ranges[i + j])
            item = min(temp_list)
            if item == float('Inf') or numpy.isinf(item):
                discretized_ranges.append(self.max_laser_value)
            else:
                if item > self.max_laser_value:
                    discretized_ranges.append(round(self.max_laser_value, 1))
                else:
                    discretized_ranges.append(round(item, 1))
        for i in range(440, 720, 20):
            temp_list = []
            for j in range(0, 20):
                temp_list.append(data.ranges[i + j])
            item = min(temp_list)
            if item == float('Inf') or numpy.isinf(item):
                discretized_ranges.append(self.max_laser_value)
            else:
                if item > self.max_laser_value:
                    discretized_ranges.append(round(self.max_laser_value, 1))
                else:
                    discretized_ranges.append(round(item, 1))

        return discretized_ranges

    def get_distance_from_desired_point(self, current_position, desired_position):
        """
        Calculates the distance from the current position to the desired point
        :param current_position:
        :param desired_position:
        :return:
        """
        distance = self.get_distance_from_point(current_position,
                                                desired_position)

        return distance

    def get_distance_from_point(self, pstart, p_end):
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :return:
        """
        a = numpy.array((pstart.x, pstart.y, pstart.z))
        b = numpy.array((p_end.x, p_end.y, p_end.z))

        distance = numpy.linalg.norm(a - b)

        return distance

    def check_husarion_has_crashed(self, laser_readings):
        """
        Based on the laser readings we check if any laser readingdistance is below
        the minimum distance acceptable.
        """
        husarion_has_crashed = False

        for laser_distance in laser_readings:
            # rospy.logwarn("laser_distance==>"+str(laser_distance))
            if laser_distance == 0.04:
                husarion_has_crashed = True
                break


        return husarion_has_crashed


    def check_reached_desired_position(self, current_position, desired_position, epsilon=0.4):
        """
        It return True if the current position is similar to the desired poistion
        """

        is_in_desired_pos = False

        x_pos_plus = desired_position.x + epsilon
        x_pos_minus = desired_position.x - epsilon
        y_pos_plus = desired_position.y + epsilon
        y_pos_minus = desired_position.y - epsilon

        x_current = current_position.x
        y_current = current_position.y

        x_pos_are_close = (x_current <= x_pos_plus) and (x_current > x_pos_minus)
        y_pos_are_close = (y_current <= y_pos_plus) and (y_current > y_pos_minus)

        is_in_desired_pos = x_pos_are_close and y_pos_are_close

        if is_in_desired_pos:
            self.reached_count+=1

        return is_in_desired_pos
