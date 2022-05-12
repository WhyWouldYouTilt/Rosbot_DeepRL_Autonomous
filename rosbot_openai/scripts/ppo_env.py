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
timestep_limit_per_episode = 2000  # Can be any Value
diagonal_dis = 10

register(
    id='Husarion_Walldodge-v1',
    entry_point='ppo_env:HusarionWalldodgeEnv',
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
        number_actions = 2#rospy.get_param('/husarion/n_actions')
        self.action_space = spaces.Discrete(number_actions)

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.odom_pub = rospy.Publisher('odom', Odometry, queue_size=10)

        # Actions and Observations
        self.init_linear_forward_speed = rospy.get_param('/husarion/init_linear_forward_speed')
        self.init_linear_turn_speed = rospy.get_param('/husarion/init_linear_turn_speed')
        self.linear_forward_speed = rospy.get_param('/husarion/linear_forward_speed')
        self.linear_turn_speed = rospy.get_param('/husarion/linear_turn_speed')
        self.angular_speed = rospy.get_param('/husarion/angular_speed')

        self.max_laser_value = 5.0
        self.min_laser_value = 0.2#0.04

        self.work_space_x_max = 10.0  # rospy.get_param("/husarion/work_space/x_max")
        self.work_space_x_min = -10.0  # rospy.get_param("/husarion/work_space/x_min")
        self.work_space_y_max = 10.0  # rospy.get_param("/husarion/work_space/y_max")
        self.work_space_y_min = -10.0  # rospy.get_param("/husarion/work_space/y_min")


        # Get Desired Point to start with
        self.desired_position = Pose()
        self.desired_position.position.x = 3.0 # rospy.get_param("/husarion/desired_pose/x")
        self.desired_position.position.y = 3.0  # rospy.get_param("/husarion/desired_pose/y")

        # Get Robot Position and yaw
        self.position = Pose()
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)

        self.precision = rospy.get_param('/husarion/precision')
        self.precision_epsilon = 1.0 / (10.0 * self.precision)
        self.move_base_precision = rospy.get_param('/husarion/move_base_precision')

        # Rewards
        self.alive_reward = 0.0  # rospy.get_param("/husarion/alive_reward")
        self.end_episode_points = 200.0  # rospy.get_param("/husarion/end_episode_points")
        self.closer_to_point_param = 10.0  # rospy.get_param("/husarion/closer_to_point_param")
        self.collision_reward = -200.0
        self.angle_reward = 0.1

        self.cumulated_steps = 0.0

        self.laser_filtered_pub = rospy.Publisher('/rosbot/laser/scan_filtered', LaserScan, queue_size=1)

        self.past_linear = 0.0
        self.past_angular = 0.0

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

        rel_dis_x = round(self.desired_position.position.x - self.position.x, 1)
        rel_dis_y = round(self.desired_position.position.y - self.position.y, 1)


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
        """init_pose = Odometry()
        init_pose.pose.pose.position.x = 3.0
        init_pose.pose.pose.position.y = 3.0
        print("INIT POSE: %s", init_pose)

        self.odom_pub.publish(init_pose)"""
        self.move_base(self.init_linear_forward_speed,
                       self.init_linear_turn_speed,
                       epsilon=self.move_base_precision,
                       update_rate=10)
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
        self.prev_diff_angle = self.diff_angle

    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the SumitXl
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """

        linear_vel = action[0]/2
        ang_vel = action[1]
        #print("action: %s", action)

        self.past_linear = linear_vel
        self.past_angular = ang_vel

        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel
        vel_cmd.angular.z = ang_vel
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
        yaw = self.yaw
        rel_theta = self.rel_theta
        diff_angle = self.diff_angle

        laser_scan = self.get_laser_scan()
        discretized_laser_scan = self.discretize_scan_210(laser_scan)
        """max_scan = max(discretized_laser_scan)
        for i in range(len(discretized_laser_scan)):
            discretized_laser_scan[i]/=max_scan"""

        current_x = self.get_current_pos_x()
        current_y = self.get_current_pos_y()
        current_distance = math.sqrt((self.desired_position.position.x-current_x)*(self.desired_position.position.x-current_x)+(self.desired_position.position.y-current_y)*(self.desired_position.position.y-current_y))

        yaw = round((yaw / 360), 1)
        diff_angle = round((diff_angle/180),2)
        # Get the linear and angular velocity from past action
        past_lin = self.get_past_linear_action()
        past_ang = self.get_past_angular_action()
        observations = discretized_laser_scan + [past_lin] + [past_ang] + [current_distance/diagonal_dis] + [rel_theta/360] + [yaw] + [diff_angle]
        """print("Distance: %s", current_distance/diagonal_dis)
        print("Distance: %s", rel_theta / 360)"""
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
        1) Husarion has moved ouside the workspace defined.
        2) Husarion is too close to an object
        3) Husarion has reached the desired position
        """

        # We fetch data through the observations
        # Its all the array except from the last four elements, which are XY odom and XY des_pos
        # laser_readings = observations[:-5]
        laser_readings = observations[:-6]

        current_position = Point()
        current_position.x = self.get_current_pos_x()
        current_position.y = self.get_current_pos_y()
        current_position.z = 0.0

        desired_position = Pose()
        desired_position.position.x = self.desired_position.position.x
        desired_position.position.y = self.desired_position.position.y
        desired_position.position.z = 0.0

        # rospy.logwarn("is DONE? laser_readings=" + str(laser_readings_180))
        # rospy.logwarn("is DONE? current_position=" + str(current_position))
        # rospy.logwarn("is DONE? desired_position=" + str(desired_position))

        # too_close_to_object = self.check_husarion_has_crashed(laser_readings)
        too_close_to_object_180 = self.check_husarion_has_crashed(laser_readings)
        inside_workspace = self.check_inside_workspace(current_position)
        reached_des_pos = self.check_reached_desired_position(current_position,
                                                              desired_position)

        # is_done = too_close_to_object or not (inside_workspace) or reached_des_pos
        is_done = too_close_to_object_180 or not (inside_workspace) or reached_des_pos
        if too_close_to_object_180:
            self.collisions+=1

        """#Training for the difficult_world:
        if reached_des_pos and self.reached_count >= 3:
            rand_number = random.randint(0, 7)
            file = open("/home/marvin/ros_workspace/src/rosbot_openai/logs/Reached.txt", "a")
            file.write("REACHED POSITION: " + str(float(self.desired_position.position.x)) + " " + str(
                float(self.desired_position.position.y)) + "\n")
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
            self.update_desired_pos(-4.0,-2.0)
        elif reached_des_pos and self.reached_count==2:
            self.update_desired_pos(-4.0,4.0)
        elif reached_des_pos and self.reached_count==3:
            self.update_desired_pos(4.0,4.0)
        elif reached_des_pos and self.reached_count==4:
            self.update_desired_pos(4.0,-4.0)"""

        if reached_des_pos:
            file = open("/home/marvin/ros_workspace/src/rosbot_openai/logs/Reached.txt", "a")
            file.write("REACHED POSITION: " + str(float(self.desired_position.position.x)) + " " + str(
                float(self.desired_position.position.y)) + "\n")
            file.close()
            self.successful_runs+=1

        """if is_done:
            rand_x = random.randint(-6,6)
            rand_y = random.randint(-6,6)
            self.update_desired_pos(float(rand_x),float(rand_y))"""
        #Uncomment for training for the simple world
        """if is_done:
            rand_number = random.randint(0, 8)
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
            self.reached_count=0"""
        """#Training for walls.launch, second curriculum
        if reached_des_pos and self.reached_count >= 2:
            rand_number = random.randint(0, 4)
            if rand_number == 0:
                self.update_desired_pos(4.0,0.0)
            if rand_number == 1:
                self.update_desired_pos(-6.0,0.0)
            if rand_number == 2:
                self.update_desired_pos(4.0,3.0)
            if rand_number == 3:
                self.update_desired_pos(4.0,-3.0)
            if rand_number == 4:
                self.update_desired_pos(-4.0,0.0)
            self.reached_count = 0"""

        #Training for 7x7.world
        if is_done:
            rand_number = random.randint(0, 13)
            if rand_number == 0:
                self.update_desired_pos(-4.0, -4.0)
            elif rand_number == 1:
                self.update_desired_pos(-4.0, 4.0)
            elif rand_number == 2:
                self.update_desired_pos(-5.0, -5.0)
            elif rand_number == 3:
                self.update_desired_pos(-5.0, 5.0)
            elif rand_number == 4:
                self.update_desired_pos(5.0, 4.0)
            elif rand_number == 5:
                self.update_desired_pos(5.0, 5.0)
            elif rand_number == 6:
                self.update_desired_pos(-6.0, -6.0)
            elif rand_number == 7:
                self.update_desired_pos(6.0, 6.0)
            elif rand_number == 8:
                self.update_desired_pos(-6.0, 6.0)
            elif rand_number == 9:
                self.update_desired_pos(6.0, -6.0)
            elif rand_number == 10:
                self.update_desired_pos(4.0, -5.0)
            elif rand_number == 11:
                self.update_desired_pos(3.0, -5.0)
            elif rand_number == 12:
                self.update_desired_pos(3.0, 5.0)
            elif rand_number == 13:
                self.update_desired_pos(-3.0, 4.0)
            elif rand_number == 10:
                self.update_desired_pos(5.0, -2.0)
            elif rand_number == 11:
                self.update_desired_pos(5.0, 2.0)
            elif rand_number == 12:
                self.update_desired_pos(0.0, 5.0)
            elif rand_number == 13:
                self.update_desired_pos(0.0, -5.0)



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
        laser_readings = observations[:-6]
        laser_min = min(laser_readings)
        current_position = Point()
        current_position.x = self.get_current_pos_x()
        current_position.y = self.get_current_pos_y()
        current_position.z = 0.0

        desired_position = Pose()
        desired_position.position.x = self.desired_position.position.x
        desired_position.position.y = self.desired_position.position.y
        desired_position.position.z = 0.0

        distance_from_des_point = self.get_distance_from_desired_point(current_position, desired_position)

        distance_difference = round((self.previous_distance_from_des_point - distance_from_des_point),4)

        diff_angle = observations[-1]*180
        diff_angle_diff = round((self.prev_diff_angle - diff_angle),2)
        #print("DIFF: %s", diff_angle_diff)
        if not done:
            # If there has been a decrease in the distance to the desired point, we reward it
            # If there has been a decrease in the angle to the desired point, we reward it
            reward = (self.closer_to_point_param * distance_difference) + (self.angle_reward*diff_angle_diff)
        else:

            reached_des_pos = self.check_reached_desired_position(current_position,
                                                                  desired_position)

            if reached_des_pos:
                reward = self.end_episode_points
                #self.successful_runs += 1
                # rospy.logwarn("GOT TO DESIRED POINT ; DONE, reward=" + str(reward))
            else:
                reward = self.collision_reward
                #self.collisions +=1
                #rospy.logerr("SOMETHING WENT WRONG ; DONE, reward=" + str(reward))
        self.previous_distance_from_des_point = distance_from_des_point
        self.prev_diff_angle = diff_angle

        #rospy.logwarn("reward=" + str(reward))
        self.cumulated_reward += reward
        #rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))
        reward = round(reward,2)
        return reward

    # Internal TaskEnv Methods
    def update_desired_pos(self, x, y):
        """
        With this method you can change the desired position that you want
        Usarion to be that initialy is set through rosparams loaded through
        a yaml file possibly.
        :new_position: Type Point, because we only value the position.
        """
        self.desired_position.position.x = x
        self.desired_position.position.y = y

        """file = open("/home/marvin/ros_workspace/src/rosbot_openai/logs/Change.txt", "a")

        file.write(
            "Changed Position to: " + str(float(self.desired_position.position.x)) + " " + str(float(self.desired_position.position.y))+"\n")
        file.close()"""

    def discretize_scan_observation(self, data):
        """
         Evtl hier Fehler. Immer gleiche Scans an gleichen Gradzahlen verwenden. Abtasten in z.B. 10 Grad abstaenden
        """

        discretized_ranges = []



        for i in range(len(data.ranges)):
            if (i%20==0):
                item = data.ranges[i]
                if item==float('Inf') or numpy.isinf(item):
                    discretized_ranges.append(self.max_laser_value)
                elif numpy.isnan(item):
                    discretized_ranges.append(self.min_laser_value)
                else:
                    if item > self.max_laser_value:
                        discretized_ranges.append(round(self.max_laser_value, 1))
                    elif item < self.min_laser_value:
                        discretized_ranges.append(round(self.min_laser_value, 1))
                    else:
                        discretized_ranges.append(round(item, 1))

        return discretized_ranges

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

    """def get_orientation_euler(self):
        # We convert from quaternions to euler
        orientation_list = [self.odom.pose.pose.orientation.x,
                            self.odom.pose.pose.orientation.y,
                            self.odom.pose.pose.orientation.z,
                            self.odom.pose.pose.orientation.w]

        roll, pitch, yaw = euler_from_quaternion(orientation_list)
        return roll, pitch, yaw"""

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
        b = numpy.array((p_end.position.x, p_end.position.y, p_end.position.z))

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
            if laser_distance <= self.min_laser_value:
                husarion_has_crashed = True
                # rospy.logwarn("HAS CRASHED==>"+str(laser_distance)+", min="+str(self.min_laser_value))
                break

            """elif laser_distance < self.min_laser_value:
                rospy.logerr("Value of laser shouldnt be lower than min==>" + str(laser_distance) + ", min=" + str(
                    self.min_laser_value))
            elif laser_distance > self.max_laser_value:
                rospy.logerr("Value of laser shouldnt be higher than max==>" + str(laser_distance) + ", max=" + str(
                    self.min_laser_value))"""

        return husarion_has_crashed

    def check_inside_workspace(self, current_position):
        """
        We check that the current position is inside the given workspace.
        """
        is_inside = False

        '''rospy.logwarn("##### INSIDE WORK SPACE? #######")
        rospy.logwarn("XYZ current_position"+str(current_position))
        rospy.logwarn("work_space_x_max"+str(self.work_space_x_max)+",work_space_x_min="+str(self.work_space_x_min))
        rospy.logwarn("work_space_y_max"+str(self.work_space_y_max)+",work_space_y_min="+str(self.work_space_y_min))
        rospy.logwarn("############")'''

        if current_position.x > self.work_space_x_min and current_position.x <= self.work_space_x_max:
            if current_position.y > self.work_space_y_min and current_position.y <= self.work_space_y_max:
                is_inside = True

        return is_inside

    def check_reached_desired_position(self, current_position, desired_position, epsilon=0.4):
        """
        It return True if the current position is similar to the desired poistion
        """

        is_in_desired_pos = False

        x_pos_plus = desired_position.position.x + epsilon
        x_pos_minus = desired_position.position.x - epsilon
        y_pos_plus = desired_position.position.y + epsilon
        y_pos_minus = desired_position.position.y - epsilon

        x_current = current_position.x
        y_current = current_position.y

        x_pos_are_close = (x_current <= x_pos_plus) and (x_current > x_pos_minus)
        y_pos_are_close = (y_current <= y_pos_plus) and (y_current > y_pos_minus)

        is_in_desired_pos = x_pos_are_close and y_pos_are_close

        if is_in_desired_pos:
            self.reached_count+=1






        '''rospy.logdebug("###### IS DESIRED POS ? ######")
        rospy.logdebug("epsilon==>"+str(epsilon))
        rospy.logdebug("current_position"+str(current_position))
        rospy.logdebug("x_pos_plus"+str(x_pos_plus)+",x_pos_minus="+str(x_pos_minus))
        rospy.logdebug("y_pos_plus"+str(y_pos_plus)+",y_pos_minus="+str(y_pos_minus))
        rospy.logdebug("x_pos_are_close"+str(x_pos_are_close))
        rospy.logdebug("y_pos_are_close"+str(y_pos_are_close))
        rospy.logdebug("is_in_desired_pos"+str(is_in_desired_pos))
        rospy.logdebug("############")'''

        return is_in_desired_pos

    def publish_filtered_laser_scan(self, laser_original_data, new_filtered_laser_range):

        length_range = len(laser_original_data.ranges)
        length_intensities = len(laser_original_data.intensities)

        laser_filtered_object = LaserScan()

        h = Header()
        h.stamp = rospy.Time.now()  # Note you need to call rospy.init_node() before this will work
        h.frame_id = "chassis"

        laser_filtered_object.header = h
        laser_filtered_object.angle_min = laser_original_data.angle_min
        laser_filtered_object.angle_max = laser_original_data.angle_max
        laser_filtered_object.angle_increment = laser_original_data.angle_increment
        laser_filtered_object.time_increment = laser_original_data.time_increment
        laser_filtered_object.scan_time = laser_original_data.scan_time
        laser_filtered_object.range_min = laser_original_data.range_min
        laser_filtered_object.range_max = laser_original_data.range_max

        laser_filtered_object.ranges = []
        laser_filtered_object.intensities = []
        for item in new_filtered_laser_range:
            laser_filtered_object.ranges.append(item)
            laser_filtered_object.intensities.append(item)

        self.laser_filtered_pub.publish(laser_filtered_object)
