#!/usr/bin/env python
#coding=utf-8
import rospy
import threading
import numpy as np
# import struct
from submap.msg import gmm, gmmlist
from geometry_msgs.msg import TransformStamped, PoseStamped, Pose, PoseArray, Transform
from tf_conversions import posemath
import tf
import sys, getopt
import random
import pickle
import math
import time
# from scipy.spatial.transform import Rotation 
from autolab_core import RigidTransform

pub=None

def callback(data):
    global pub
    # print("------------------------------------3")
    gmm_result=gmm()
    gmm_result.header=data.header
    gmm_result.header.frame_id="/map"
    
    for i in range(0,data.mix_num):
        rotation_quaternion = np.asarray([data.pose.orientation.w,data.pose.orientation.x,data.pose.orientation.y,data.pose.orientation.z])
        T=np.asarray([data.pose.position.x,data.pose.position.y,data.pose.position.z])
        T_qua2rota = RigidTransform(rotation_quaternion, T)
        T=T.reshape(T.shape[0],1)
        # print(T)
        R=T_qua2rota.rotation
        # print(R)
        # R_inv=np.linalg.inv(R)
        # R_T_inv=np.linalg.inv(np.transpose(R))
        p_camera=np.asarray([data.x[i],data.y[i],data.z[i]])
        p_camera=p_camera.reshape(p_camera.shape[0],1)
        p_result=np.dot(R,p_camera)+T
        gmm_result.x.append(p_result[0])
        gmm_result.y.append(p_result[1])
        gmm_result.z.append(p_result[2])
        # print("------------------------------------4")

        covar=np.identity(3)
        covar[0,0]=data.x_var[i]
        covar[1,1]=data.y_var[i]
        covar[2,2]=data.z_var[i]
        covar_result=np.dot(R,covar)
        covar_result=np.dot(covar_result,np.transpose(R))
        gmm_result.x_var.append(covar_result[0,0])
        gmm_result.y_var.append(covar_result[1,1])
        gmm_result.z_var.append(covar_result[2,2])
        # print("------------------------------------5")

    gmm_result.pose.position.x=0
    gmm_result.pose.position.y=0
    gmm_result.pose.position.z=0
    gmm_result.pose.orientation.x=0
    gmm_result.pose.orientation.y=0
    gmm_result.pose.orientation.z=0
    gmm_result.pose.orientation.w=1

    gmm_result.mix_num=data.mix_num
    pub.publish(gmm_result)







def main(argv):
    # print("------------------------------------1")
    global pub
    rospy.init_node('talker', anonymous=True)
    pub = rospy.Publisher('gmm_after_trans', gmm, queue_size=10)
    rospy.Subscriber("subgmm", gmm, callback)
    # print("------------------------------------2")
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        rospy.spin()
        rate.sleep()
        
if __name__ == '__main__':
    main(sys.argv[1:])