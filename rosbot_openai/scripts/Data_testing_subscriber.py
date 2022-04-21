#!/usr/bin/env python
import rospy
import numpy
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import BatteryState
from openai_ros.robot_envs import husarion_env
#import tensorflow as tf


def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)
    TEST_ONE_TIME = True
    # Creating the different subscribers

    # Laserscan
    rospy.Subscriber('/scan_rgbd', LaserScan, laser_scan_callback)

    # RGB-D Image
    #rospy.Subscriber('/camera/depth/image_raw', Image, camera_depth_image_raw_callback)

    # Pointcloud
    #rospy.Subscriber('/camera/depth/points', PointCloud2, camera_depth_pointcloud_callback)

    # rospy.Subscriber('/base_scan', LaserScan, degree_callback)

    # BatteryState
    # rospy.Subscriber('/battery', BatteryState, battery_callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


def battery_callback(data):
    rospy.loginfo("Battery State: %s", data.present)


def degree_callback(data):
    rospy.loginfo("LASERDATA 180 DEGREES: %s", data.angle_min)


def laser_scan_callback(data):
    discretized_ranges = []

    for i in range(0, 640):
        if (i % 20 == 0):
            item = data.ranges[i]
            if item == float('Inf') or numpy.isinf(item):
                discretized_ranges.append(5.5)
            elif numpy.isnan(item):
                discretized_ranges.append(0.6)
            else:
                if item > 5.5:
                    discretized_ranges.append(round(5.5, 1))
                elif item < 0.3:
                    discretized_ranges.append(round(0.3, 1))
                else:
                    discretized_ranges.append(round(item, 1))

    print("SCAN: %s", discretized_ranges)


def camera_depth_image_raw_callback(data):
    rospy.loginfo("RGB IMAGEDATA: %s", data.width)


def camera_depth_pointcloud_callback(data):
    rospy.loginfo("POINTCLOUD SHAPE: %s", data)
    # rospy.loginfo("POINTCLOUD DATA: %s", data.width)


if __name__ == '__main__':
    listener()
