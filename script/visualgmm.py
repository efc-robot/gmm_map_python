#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import signal
import sys
import rospy
import copy

from visualization_msgs.msg import Marker,MarkerArray
from geometry_msgs.msg import Point,Vector3
from geometry_msgs.msg import PoseWithCovarianceStamped
from submap.msg import gmm, gmmlist

pubVehiclePosition = None
count_cb = 0

def forcequit(signum, frame):
    print '123'
    print 'stop fusion'
    sys.exit()

def new_SPHERE_Marker(frame_id, position, scale, id):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.id = id  # enumerate subsequent markers here
    marker.action = Marker.ADD  # can be ADD, REMOVE, or MODIFY
    marker.ns = "vehicle_model"
    marker.type = Marker.SPHERE
    
    marker.pose.position = position
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    marker.scale = scale

    marker.color.r = 1.0
    marker.color.g = 0.50
    marker.color.b = 0.0
    marker.color.a = 1.0 
    return marker

def subgmm_list_cb(data):
    global pubVehiclePosition, count_cb
    # print(count_cb)
    count_cb += 1
    markerarray = MarkerArray()
    for index in range(len(data.data)):
        cu_gmm = data.data[index]
        cu_mix_num = cu_gmm.mix_num
        for jndex in range(cu_mix_num):
            marker = new_SPHERE_Marker("map",Point(cu_gmm.x[jndex],cu_gmm.y[jndex],cu_gmm.z[jndex]),
            Vector3(10*cu_gmm.x_var[jndex],10*cu_gmm.y_var[jndex],10*cu_gmm.z_var[jndex]),
            jndex + index*1000 )
            markerarray.markers.append(marker)
    # print(markerarray)
    pubVehiclePosition.publish(markerarray)


def subgmm_cb(data):
    global pubVehiclePosition, count_cb
    # print(count_cb)
    count_cb += 1
    markerarray = MarkerArray()
    cu_gmm = data
    cu_mix_num = cu_gmm.mix_num
    for jndex in range(cu_mix_num):
        marker = new_SPHERE_Marker("map",Point(cu_gmm.x[jndex],cu_gmm.y[jndex],cu_gmm.z[jndex]),
        Vector3(10*cu_gmm.x_var[jndex],10*cu_gmm.y_var[jndex],10*cu_gmm.z_var[jndex]),
        jndex )
        markerarray.markers.append(marker)
    # print(markerarray)
    pubVehiclePosition.publish(markerarray)
    



def main():
    global pubVehiclePosition
    signal.signal(signal.SIGINT, forcequit)                                
    signal.signal(signal.SIGTERM, forcequit)
    rospy.init_node('visual_gmm', anonymous=True)
    rate = rospy.Rate(5)
    pubVehiclePosition = rospy.Publisher('vehicle_robot_position', MarkerArray, queue_size=10)
    sub_list_subgmm = rospy.Subscriber('subgmm_list_tmp',gmmlist, subgmm_list_cb,queue_size=100)
    sub_subgmm = rospy.Subscriber('gmm_after_trans',gmm, subgmm_cb,queue_size=100)


    # markerarray = MarkerArray(

    # for i in range(100):
    #     rate.sleep()
    #     print("OK" + str(i) )

    rospy.spin()

if __name__ == "__main__":
    main()