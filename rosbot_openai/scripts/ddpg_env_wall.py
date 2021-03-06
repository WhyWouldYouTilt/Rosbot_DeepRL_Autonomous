#!/usr/bin/env python2.7
from operator import neg

import random
import math
import rospy
import numpy
from gym import spaces
from openai_ros.robot_envs import husarion_env
from gym.envs.registration import register
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point, Pose
from tf.transformations import euler_from_quaternion
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
from nav_msgs.msg import Odometry

# The path is __init__.py of openai_ros, where we import the SumitXlMazeEnv directly
timestep_limit_per_episode = 10000  # Can be any Value
diagonal_dis = math.sqrt(2) * (3.6 + 3.8)

register(
    id='Husarion_Walldodge-v1',
    entry_point='walldodge_actor_env:HusarionWalldodgeEnv',
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
        self.yaw = None
        self.rel_theta = None
        self.diff_angle = None
        number_actions = rospy.get_param('/husarion/n_actions')
        self.action_space = spaces.Discrete(number_actions)

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10)

        # Actions and Observations
        self.init_linear_forward_speed = rospy.get_param('/husarion/init_linear_forward_speed')
        self.init_linear_turn_speed = rospy.get_param('/husarion/init_linear_turn_speed')

        self.linear_forward_speed = rospy.get_param('/husarion/linear_forward_speed')
        self.linear_turn_speed = rospy.get_param('/husarion/linear_turn_speed')
        self.angular_speed = rospy.get_param('/husarion/angular_speed')

        self.new_ranges = rospy.get_param('/husarion/new_ranges')
        self.max_laser_value = rospy.get_param('/husarion/max_laser_value')
        self.min_laser_value = rospy.get_param('/husarion/min_laser_value')

        self.work_space_x_max = 4.5  # rospy.get_param("/husarion/work_space/x_max")
        self.work_space_x_min = -4.5  # rospy.get_param("/husarion/work_space/x_min")
        self.work_space_y_max = 4.5  # rospy.get_param("/husarion/work_space/y_max")
        self.work_space_y_min = -4.5  # rospy.get_param("/husarion/work_space/y_min")

        # Get Desired Point to Get
        self.desired_position = Pose()
        self.desired_position.position.x = 3.0  # rospy.get_param("/husarion/desired_pose/x")
        self.desired_position.position.y = -1.0  # rospy.get_param("/husarion/desired_pose/y")

        # Get Robot Position and yaw
        self.position = Pose()
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)

        self.precision = rospy.get_param('/husarion/precision')
        self.precision_epsilon = 1.0 / (10.0 * self.precision)

        self.move_base_precision = rospy.get_param('/husarion/move_base_precision')

        # Rewards
        self.closer_to_point_reward = rospy.get_param("/husarion/closer_to_point_reward")
        self.alive_reward = 0.0  # rospy.get_param("/husarion/alive_reward")
        self.end_episode_points = 100.0  # rospy.get_param("/husarion/end_episode_points")
        self.closer_to_point_param = 500.0  # rospy.get_param("/husarion/closer_to_point_param")

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
        self.move_base(self.init_linear_forward_speed,
                       self.init_linear_turn_speed,
                       epsilon=self.move_base_precision,
                       update_rate=10)

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

    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the SumitXl
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """
        linear_vel = action[0]
        ang_vel = action[1]
        self.past_linear = linear_vel
        self.past_angular = ang_vel
        # self.move_base(linear_vel, ang_vel, epsilon=self.move_base_precision, update_rate=10)
        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel / 4
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
        print("DIFF ANGLE IS: %s", diff_angle)

        laser_scan = self.get_laser_scan()
        discretized_laser_scan = self.discretize_scan_observation(laser_scan,
                                                                  self.new_ranges
                                                                  )
        current_distance = math.hypot(self.desired_position.position.x - self.position.x,
                                      self.desired_position.position.y - self.position.y)
        odometry = self.get_odom()
        x_position = odometry.pose.pose.position.x
        y_position = odometry.pose.pose.position.y

        odometry_array = [round(x_position, 2),
                          round(y_position, 2),
                          round((yaw / 360), 1)]
        desired_position = [round(self.desired_position.position.x, 2),
                            round(self.desired_position.position.y, 2)]

        # Get the linear and angular velocity from past action
        past_lin = self.get_past_linear_action()
        past_ang = self.get_past_angular_action()

        observations = discretized_laser_scan + odometry_array + desired_position + [rel_theta / 360] + [
            diff_angle / 180] + [current_distance / diagonal_dis] + [past_lin] + [past_ang]

        return observations

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
        laser_readings_180 = observations[:-10]

        current_position = Point()
        current_position.x = observations[-10]
        current_position.y = observations[-9]
        current_position.z = 0.0

        desired_position = Pose()
        desired_position.position.x = observations[-7]
        desired_position.position.y = observations[-6]
        desired_position.position.z = 0.0

        # rospy.logwarn("is DONE? laser_readings=" + str(laser_readings_180))
        # rospy.logwarn("is DONE? current_position=" + str(current_position))
        # rospy.logwarn("is DONE? desired_position=" + str(desired_position))

        # too_close_to_object = self.check_husarion_has_crashed(laser_readings)
        too_close_to_object_180 = self.check_husarion_has_crashed(laser_readings_180)
        inside_workspace = self.check_inside_workspace(current_position)
        reached_des_pos = self.check_reached_desired_position(current_position,
                                                              desired_position,
                                                              self.precision_epsilon)

        # is_done = too_close_to_object or not (inside_workspace) or reached_des_pos
        is_done = too_close_to_object_180 or not (inside_workspace) or reached_des_pos

        if reached_des_pos:
            rand_number = random.randint(0, 5)
            file = open("/home/marvin/ros_workspace/src/rosbot_openai/logs/Reached.txt", "a")
            file.write("REACHED POSITION: "+ str(float(self.desired_position.position.x))+" "+str(float(self.desired_position.position.y))+"\n")
            file.close()
            if rand_number == 0:
                self.update_desired_pos(1.0, 3.0)
            elif rand_number == 1:
                self.update_desired_pos(-3.0, 3.0)
            elif rand_number == 2:
                self.update_desired_pos(-3.0, -3.0)
            elif rand_number == 3:
                self.update_desired_pos(0.0, -2.0)
            elif rand_number == 4:
                self.update_desired_pos(3.0, 3.0)
            elif rand_number == 5:
                self.update_desired_pos(4.0, -3.0)

        '''rospy.logwarn("####################")
        rospy.logwarn("too_close_to_object=" + str(too_close_to_object))
        rospy.logwarn("inside_workspace=" + str(inside_workspace))
        rospy.logwarn("reached_des_pos=" + str(reached_des_pos))
        rospy.logwarn("is_done=" + str(is_done))
        rospy.logwarn("######## END DONE ##")'''

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

        # laser_readings = observations[:-5]
        laser_readings_180 = observations[:-7]

        current_position = Point()
        current_position.x = observations[-10]
        current_position.y = observations[-9]
        current_position.z = 0.0

        desired_position = Pose()
        desired_position.position.x = observations[-7]
        desired_position.position.y = observations[-6]
        desired_position.position.z = 0.0

        distance_from_des_point = self.get_distance_from_desired_point(current_position, desired_position)

        distance_difference = distance_from_des_point - self.previous_distance_from_des_point

        '''rospy.logwarn("current_position=" + str(current_position))
        rospy.logwarn("desired_point=" + str(desired_position))

        rospy.logwarn("total_distance_from_des_point=" + str(self.previous_distance_from_des_point))
        rospy.logwarn("distance_from_des_point=" + str(distance_from_des_point))
        rospy.logwarn("distance_difference=" + str(distance_difference))'''

        if not done:
            # If there has been a decrease in the distance to the desired point, we reward it
            if distance_difference < 0.0:
                rospy.logwarn("DECREASE IN DISTANCE GOOD")
                # reward = self.closer_to_point_reward
                reward = self.closer_to_point_param * (neg(distance_difference))
            else:
                reward = self.alive_reward
        else:

            reached_des_pos = self.check_reached_desired_position(current_position,
                                                                  desired_position,
                                                                  self.precision_epsilon)

            if reached_des_pos:
                reward = self.end_episode_points
                # rospy.logwarn("GOT TO DESIRED POINT ; DONE, reward=" + str(reward))
            else:
                reward = -1 * self.end_episode_points
                rospy.logerr("SOMETHING WENT WRONG ; DONE, reward=" + str(reward))

        self.previous_distance_from_des_point = distance_from_des_point

        rospy.logwarn("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
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
        self.desired_position.position.x = x
        self.desired_position.position.y = y

        file = open("/home/marvin/ros_workspace/src/rosbot_openai/logs/Change.txt", "a")

        file.write(
            "Changed Position to: " + str(float(self.desired_position.position.x)) + " " + str(float(self.desired_position.position.y))+"\n")
        file.close()

    def discretize_scan_observation(self, data, new_ranges):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """

        discretized_ranges = []
        mod = len(data.ranges) / new_ranges

        filtered_range = []

        '''rospy.logdebug("data=" + str(data))
        rospy.logdebug("new_ranges=" + str(new_ranges))
        rospy.logdebug("mod=" + str(mod))'''

        nan_value = (self.min_laser_value + self.min_laser_value) / 2.0

        for i, item in enumerate(data.ranges):
            if (i % mod == 0):
                if item == float('Inf') or numpy.isinf(item):
                    # rospy.logerr("Infinite Value=" + str(item)+"Assigning Max value")
                    discretized_ranges.append(self.max_laser_value)
                elif numpy.isnan(item):
                    # rospy.logerr("Nan Value=" + str(item)+"Assigning MIN value")
                    discretized_ranges.append(self.min_laser_value)
                else:
                    # We clamp the laser readings
                    if item > self.max_laser_value:
                        # rospy.logwarn("Item Bigger Than MAX, CLAMPING=>" + str(item)+", MAX="+str(self.max_laser_value))
                        discretized_ranges.append(round(self.max_laser_value, 1))
                    elif item < self.min_laser_value:
                        # rospy.logwarn("Item smaller Than MIN, CLAMPING=>" + str(item)+", MIN="+str(self.min_laser_value))
                        discretized_ranges.append(round(self.min_laser_value, 1))
                    else:
                        # rospy.logwarn("Normal Item, no processing=>" + str(item))
                        discretized_ranges.append(round(item, 1))
                # We add last value appended
                filtered_range.append(discretized_ranges[-1])
            else:
                # We add value zero
                filtered_range.append(0.0)

        # rospy.logwarn(">>>>>>>>>>>>>>>>>>>>>>discretized_ranges=>" + str(discretized_ranges))

        self.publish_filtered_laser_scan(laser_original_data=data,
                                         new_filtered_laser_range=filtered_range)

        return discretized_ranges

    def get_orientation_euler(self):
        # We convert from quaternions to euler
        orientation_list = [self.odom.pose.pose.orientation.x,
                            self.odom.pose.pose.orientation.y,
                            self.odom.pose.pose.orientation.z,
                            self.odom.pose.pose.orientation.w]

        roll, pitch, yaw = euler_from_quaternion(orientation_list)
        return roll, pitch, yaw

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
            if laser_distance == self.min_laser_value:
                husarion_has_crashed = True
                # rospy.logwarn("HAS CRASHED==>"+str(laser_distance)+", min="+str(self.min_laser_value))
                break

            elif laser_distance < self.min_laser_value:
                rospy.logerr("Value of laser shouldnt be lower than min==>" + str(laser_distance) + ", min=" + str(
                    self.min_laser_value))
            elif laser_distance > self.max_laser_value:
                rospy.logerr("Value of laser shouldnt be higher than max==>" + str(laser_distance) + ", max=" + str(
                    self.min_laser_value))

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

    def check_reached_desired_position(self, current_position, desired_position, epsilon=0.2):
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
