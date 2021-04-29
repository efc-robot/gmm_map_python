#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.mixture import GaussianMixture
from std_msgs.msg import Header, String
from geometry_msgs.msg import Pose, PoseStamped, Transform, TransformStamped, Quaternion
from generate_descriptor import Descriptor
from gmm_map_python.msg import gmm
from autolab_core import RigidTransform

class GMMFrame: #GMM建完后操作都在GMMSubmap类中进行
    def __init__(self):
        self.mix_num =0
        self.dim=3
        self.weights =np.zeros((0,1))
        self.means =np.zeros((0,3))
        self.covariances = np.zeros((0,3))
        #不要frameid和pose，两者存在submap类中，在生成message和转换的时候再输入
        # self.frame_id=""
        # self.pose=Pose()
        # print("----------------------GMMinit")

    def Frame_to_SklearnMixture(self):
        clf = GaussianMixture(n_components=self.mix_num, covariance_type='diag')
        #为均值方差赋值
        clf.fit(np.random.rand(self.mix_num, 3))  # Now it thinks it is trained
        #下面这句话有更简单表达,把size(3,1)变成size（3，）只需reshape(3)即可
        for i in range(clf.weights_.shape[0]):    
            clf.weights_[i] = self.weights[i]   # mixture weights (n_components,) 
        clf.weights_ = clf.weights_/sum(clf.weights_)
        # print(type(clf.weights_),clf.weights_.shape)
        clf.means_ = self.means          # mixture means (n_components, 2) 
        clf.covariances_ = self.covariances  # mixture cov (n_components, 2, 2
        return clf

    def GMMsup(self,GMMmodel): #可用作gmm地图的增量更新，也可用作gmm对象的深拷贝
        # print(type(GMMmodel))
        if isinstance(GMMmodel,GMMFrame):
            # print("GMMmodel sup start!")
            self.mix_num = self.mix_num+GMMmodel.mix_num
            self.dim=3
            self.weights = np.concatenate((self.weights,GMMmodel.weights.reshape(GMMmodel.weights.shape[0],1)),axis=0)
            self.means = np.concatenate((self.means,GMMmodel.means),axis=0)
            self.covariances = np.concatenate((self.covariances,GMMmodel.covariances),axis=0)
        else: #isinstance(GMMmodel,sklearn.mixture):
            # print("sklearn.mixture sup start!")
            self.mix_num = self.mix_num+GMMmodel.n_components
            self.dim=3
            self.weights = np.concatenate((self.weights,GMMmodel.weights_.reshape(GMMmodel.weights_.shape[0],1)),axis=0)
            # print(self.weights.shape)
            self.means = np.concatenate((self.means,GMMmodel.means_),axis=0)
            self.covariances = np.concatenate((self.covariances,GMMmodel.covariances_),axis=0)
        # print("----------------------GMMsup")
        #同理，此处不需要id和pose
        # self.frame_id=
        # self.pose= 
    
    def GMMmsg(self, frame_input,submap_pose):#不需要
        gmmmsg=gmm() 
        gmmmsg.mix_num=self.mix_num
        gmmmsg.header=Header()
        gmmmsg.header.frame_id=frame_input
        gmmmsg.pose=submap_pose 
        for i in range(0,self.mix_num):
            gmmmsg.prior.append(self.weights[i])
            gmmmsg.x.append(self.means[i][0])
            gmmmsg.y.append(self.means[i][1])
            gmmmsg.z.append(self.means[i][2])
            gmmmsg.x_var.append(self.covariances[i][0])
            gmmmsg.y_var.append(self.covariances[i][1])
            gmmmsg.z_var.append(self.covariances[i][2])
        return gmmmsg


    def GMM2odom(self,submap_pose):
        # print(self.covariances.shape)
        # gmm_after_trans=GMMFrame()
        # gmm_after_trans.frame_id=frame_input
        # gmm_after_trans.mix_num=self.mix_num
        for i in range(0,self.mix_num):
            rotation_quaternion = np.asarray([submap_pose.transform.rotation.w,submap_pose.transform.rotation.x,submap_pose.transform.rotation.y,submap_pose.transform.rotation.z])
            T=np.asarray([submap_pose.transform.translation.x,submap_pose.transform.translation.y,submap_pose.transform.translation.z])
            T_qua2rota = RigidTransform(rotation_quaternion, T)
            T=T.reshape(T.shape[0],1)
            R=T_qua2rota.rotation
            p_camera=self.means[i]
  
            p_camera=p_camera.reshape(p_camera.shape[0],1)
            p_result=np.dot(R,p_camera)+T
            self.means[i]=p_result.reshape(1,3)
            # gmm_after_trans.means = np.concatenate((gmm_after_trans.means,p_result.reshape(1,3)),axis=0)
            
            covar=np.identity(3)
            # print(covar.shape,self.covariances)
            covar[0,0]=self.covariances[i][0]
            covar[1,1]=self.covariances[i][1]
            covar[2,2]=self.covariances[i][2]
            covar_result=np.dot(R,covar)
            covar_result=np.dot(covar_result,np.transpose(R))
            covar_tmp=np.array([covar_result[0,0],covar_result[1,1],covar_result[2,2]])
            
            self.covariances[i]=covar_tmp
            # gmm_after_trans.covariances=covar_tmp
        #GMMFrame里不存pose，建msg的函数再传入
        # return gmm_after_trans

        
    def GMMreg(self):
        #放在Register类中了
        pass


    def GMMcog(self,Des): #场景识别
        #生成和当前GMMFrame一样的sklearnGMM
        clf=self.Frame_to_SklearnMixture()
        #重采样
        [pc,aaa]=clf.sample(4096)
        # print("success!!")
        #生成描述子 Des是生成描述子的类 des是描述子
        Des.point_cloud = pc
        # print( Des.point_cloud)
        Des.filter()
        des = Des.generate_descriptor()
        # normalize
        des= des / (np.linalg.norm(des))
        return Des, des, pc
        
    def GMMnav(self):#似乎单写一个py文件用main函数来控制更合适
        pass

    def GMMmer(self):# HGMM生成
        pass