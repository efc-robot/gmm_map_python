#! /usr/bin/env python
# -*- coding: utf-8 -*-

from std_msgs.msg import Header, String
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from geometry_msgs.msg import Pose, PoseStamped, Transform, TransformStamped, Quaternion
from nav_msgs.msg import Path
import tf
import tf2_ros
from tf2_sensor_msgs import do_transform_cloud
import rospy
import time
import numpy as np
from autolab_core import RigidTransform
import time
from sklearn.mixture import GaussianMixture
import threading
import random
import pickle
# import octomap
# from octomap_msgs.msg import Octomap
from generate_descriptor import Descriptor
from map_registration import Registrar
from TFGraph import TFGraph
import gtsam
import copy
import pygraph

COVAR_STR = "1.000000 0.000000 0.000000 0.000000 0.000000 0.000000   1.000000 0.000000 0.000000 0.000000 0.000000   1.000000 0.000000 0.000000 0.000000   1.000000 0.000000 0.000000   1.000000 0.000000   1.000000"


def strip_leading_slash(s):
    return s[1:] if s.startswith("/") else s

def trans2pose(trans):
    pose = Pose()
    pose.orientation = trans.rotation
    pose.position = trans.translation
    return pose


def pose2trans(pose):
    trans = Transform()
    trans.rotation = pose.orientation
    trans.translation =pose.position
    return trans

def trans2Rigidtrans(trans, from_frame, to_frame):
    rotation_quaternion = np.asarray([trans.rotation.w, trans.rotation.x, trans.rotation.y, trans.rotation.z])
    T_trans = np.asarray([trans.translation.x, trans.translation.y, trans.translation.z])
    T_qua2rota = RigidTransform(rotation_quaternion, T_trans, from_frame = from_frame, to_frame=to_frame)
    return T_qua2rota


def pose2Rigidtrans(pose, from_frame, to_frame):
    trans = pose2trans(pose)
    T_qua2rota = trans2Rigidtrans(trans, from_frame, to_frame)
    return T_qua2rota


def transstamp2Rigidtrans(trans):
    from_frame = trans.child_frame_id
    # to_frame = trans.child_frame_id
    to_frame = trans.header.frame_id
    ## 注意,tranform 中的 child_id 和 frame_id 代表. child_id 在 frame_id 中的位置. 在 Rigidtrans 中,from 代表 需要倍转换的坐标系,to 代表要转换到的坐标系. 也就是: from 对应 child_id. to 对应 frame_id
    T_qua2rota = trans2Rigidtrans(trans.transform, from_frame, to_frame)
    return T_qua2rota

def Rigidtrans2transstamped(Rigidtrans):
    trans = TransformStamped()
    trans.header.frame_id = Rigidtrans.to_frame
    trans.child_frame_id = Rigidtrans.from_frame
    ## 注意,tranform 中的 child_id 和 frame_id 代表. child_id 在 frame_id 中的位置. 在 Rigidtrans 中,from 代表 需要倍转换的坐标系,to 代表要转换到的坐标系. 也就是: from 对应 child_id. to 对应 frame_id
    trans.transform = pose2trans(Rigidtrans.pose_msg)
    return trans

def np2pointcloud2(points, frame_id, timestamp = None): #转换三维点云
    outputheader = Header()
    outputheader.frame_id = frame_id
    if None == timestamp:
        outputheader.stamp = rospy.Time(0)
    else:
        outputheader.stamp = timestamp
    return point_cloud2.create_cloud_xyz32(outputheader,points)

def poseID2robotID(poseID):
    submap_index = int(poseID%1e8)
    robot_id = int(poseID/1e8)
    return robot_id, submap_index

def robotID2poseID(robot_id , submap_index):
    return int(submap_index + 1e8*robot_id)

def voxel_filter(point_cloud, leaf_size,filter_mode):
    filtered_points = []
    # 作业3
    # 屏蔽开始
    #step1 计算边界点
    x_max, y_max, z_max = np.amax(point_cloud,axis=0)      #计算 x,y,z三个维度的最值
    x_min, y_min, z_min = np.amin(point_cloud, axis=0)
    #step2 确定体素的尺寸
    size_r = leaf_size
    #step3 计算每个 volex的维度
    Dx = (x_max - x_min)/size_r
    Dy = (y_max - y_min)/size_r
    Dz = (z_max - z_min)/size_r
    #step4 计算每个点在volex grid内每一个维度的值
    h = list()
    for i in range(len(point_cloud)):
        hx = np.floor((point_cloud[i][0] - x_min)/size_r)
        hy = np.floor((point_cloud[i][1] - y_min)/size_r)
        hz = np.floor((point_cloud[i][2] - z_min)/size_r)
        h.append(hx + hy*Dx + hz*Dx*Dy)
    #step5 对h值进行排序
    h = np.array(h)
    h_indice  = np.argsort(h)   #提取索引
    h_sorted = h[h_indice]      #升序
    count = 0 #用于维度的累计
    #将h值相同的点放入到同一个grid中，并进行筛选
    for i  in range(len(h_sorted)-1):      #0-19999个数据点
        if h_sorted[i] == h_sorted[i+1]:   #当前的点与后面的相同，放在同一个volex grid中
            continue
        else:
            if(filter_mode == "centroid"):    #均值滤波
                point_idx = h_indice[count: i+1]
                filtered_points.append(np.mean(point_cloud[point_idx],axis=0))   #取同一个grid的均值
                count = i
            elif(filter_mode == "random"):  #随机滤波
                point_idx = h_indice[count: i+1]
                random_points =  random.choice(point_cloud[point_idx])
                filtered_points.append(random_points)
                count = i

    # 屏蔽结束

    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points


class ConstrOptimizer:
    def __init__(self, robot_id, all_pose_map_lists, all_self_constr_lists, inter_constr_list, tf_graph, newsubmap_builder, g2ofname):
        self.robot_id = robot_id
        self.tf_graph = tf_graph
        connected_robots = self.tf_graph.get_connected_robots(self.robot_id)
        self.pose_map = []
        self.constr = []
        for cons in inter_constr_list:
            if (cons.from_robot in connected_robots) and (cons.to_robot in connected_robots):
                self.constr.append(cons)
        for robot_id in connected_robots:
            self.pose_map += all_pose_map_lists["robot{}".format(robot_id)]
            self.constr += all_self_constr_lists["robot{}".format(robot_id)]
        self.g2ofname = g2ofname
        self.newsubmap_builder = newsubmap_builder


    def vector6(self, x, y, z, a, b, c):
        """Create 6d double numpy array."""
        return np.array([x, y, z, a, b, c], dtype=np.float)

    def Posemap2g2oVertex(self, g2ofname, pose_array, add=False):
        if add:
            openmethod = "a"
        else:
            openmethod = "w"
        with open(g2ofname, openmethod) as g2ofile:
            for VERTEXpose in pose_array:
                if VERTEXpose.robot_id != self.robot_id:
                    pose_local = pose2Rigidtrans(VERTEXpose.submap_pose, 'submap_base_link_{}'.format(VERTEXpose.submap_index), 'robot{}/map'.format(VERTEXpose.robot_id))
                    # print VERTEXpose.robot_id
                    # print self.robot_id
                    # print pose_local
                    # print self.tf_graph.get_tf(VERTEXpose.robot_id, self.robot_id)
                    pose_global = self.tf_graph.get_tf(VERTEXpose.robot_id, self.robot_id) * pose_local
                    pose_global = trans2pose(Rigidtrans2transstamped(pose_global).transform)
                else:
                    pose_global = VERTEXpose.submap_pose
                vertexID = robotID2poseID(VERTEXpose.robot_id, VERTEXpose.submap_index)
                # print vertexID
                print >> g2ofile, "VERTEX_SE3:QUAT {pointID} {tx} {ty} {tz} {rx} {ry} {rz} {rw}".format(pointID = vertexID,
                    tx = pose_global.position.x, ty = pose_global.position.y, tz = pose_global.position.z, 
                    rx = pose_global.orientation.x, ry = pose_global.orientation.y, rz = pose_global.orientation.z, rw = pose_global.orientation.w)

    def AddConstr2g2oEdge(self, g2ofname, constr):
        with open(g2ofname, "a") as g2ofile:
            for EDGEtrans in constr:
                id1 = robotID2poseID(EDGEtrans.to_robot, EDGEtrans.to_index)
                id2 = robotID2poseID(EDGEtrans.from_robot,EDGEtrans.from_index)
                # print '{} {}'.format(id1, id2)
                print >> g2ofile, "EDGE_SE3:QUAT {id1} {id2} {tx} {ty} {tz} {rx} {ry} {rz} {rw}  {COVAR_STR}".format(id1=id1,id2=id2, tx =EDGEtrans.constraint.transform.translation.x , ty =EDGEtrans.constraint.transform.translation.y , tz =EDGEtrans.constraint.transform.translation.z, rx =EDGEtrans.constraint.transform.rotation.x, ry =EDGEtrans.constraint.transform.rotation.y, rz =EDGEtrans.constraint.transform.rotation.z, rw =EDGEtrans.constraint.transform.rotation.w, COVAR_STR=COVAR_STR)

    def setPoseWithSubmapID(self, robotID, submapID, resultpose):
        for cursubmap in self.pose_map:
            map_robotID = cursubmap.robot_id
            map_submapID = cursubmap.submap_index
            if map_robotID == robotID and map_submapID == submapID:
                tmpRigit = RigidTransform(resultpose.rotation().quaternion(), resultpose.translation().vector(), from_frame = 'submap_base_link_{}'.format(submapID), to_frame = 'robot{}/map'.format(self.robot_id))
                if robotID != self.robot_id:
                    tmpRigit = self.tf_graph.get_tf(self.robot_id, robotID) * tmpRigit
                cursubmap.submap_pose =  trans2pose(Rigidtrans2transstamped(tmpRigit).transform)

    def setNowPosWithSubmapID(self, robotID, submapID, resultpose):
        cursubmap = self.newsubmap_builder
        map_robotID = cursubmap.robot_id
        map_submapID = cursubmap.submap_index
        if map_robotID == robotID and map_submapID == submapID:
            tmpRigit = RigidTransform(resultpose.rotation().quaternion(), resultpose.translation().vector())
            cursubmap.submap_pose =  trans2pose(Rigidtrans2transstamped(tmpRigit).transform)



    def gtsamOpt2Pose(self, g2ofname):
        graphWithPrior, initial = gtsam.readG2o(g2ofname, True)
        priorModel = gtsam.noiseModel_Diagonal.Variances(self.vector6(1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4) )
        print("Adding prior to g2o file ")

        firstKey = initial.keys().at(0)
        graphWithPrior.add(gtsam.PriorFactorPose3(firstKey, gtsam.Pose3(), priorModel))

        params = gtsam.GaussNewtonParams()
        params.setVerbosity("Termination")  # this will show info about stopping conds
        optimizer = gtsam.GaussNewtonOptimizer(graphWithPrior, initial, params)
        result = optimizer.optimize()
        print("Optimization complete")
        print("initial error = ", graphWithPrior.error(initial))
        print("final error = ", graphWithPrior.error(result))

        resultPoses = gtsam.allPose3s(result)
        resultPoseslen = resultPoses.keys().size()
        for keyindex in range(resultPoseslen):
            key = resultPoses.keys().at(keyindex)
            resultpose = resultPoses.atPose3(key)
            tmprobotID, tmpsubmapID = poseID2robotID(key)
            self.setPoseWithSubmapID(tmprobotID, tmpsubmapID, resultpose)
            self.setNowPosWithSubmapID(tmprobotID, tmpsubmapID, resultpose)
            tmp = 1


    def constructproblem(self):
        # print "w"
        self.Posemap2g2oVertex(self.g2ofname, self.pose_map)
        # print "a"
        self.Posemap2g2oVertex(self.g2ofname, [self.newsubmap_builder], add=True)
        self.AddConstr2g2oEdge(self.g2ofname, self.constr)
        self.gtsamOpt2Pose(self.g2ofname)
        
        tmp = 1




    



class ConstraintTransform:
    def __init__(self,from_robot, from_index, to_robot, to_index, trans_constr ):
        self.from_robot = from_robot
        self.from_index = from_index
        self.to_robot = to_robot
        self.to_index = to_index
        self.constraint = trans_constr

    def show(self):
        print 'to  :{} {}'.format(self.to_robot, self.to_index) + ' from:{} {}'.format(self.from_robot, self.from_index)
        # print 'from:{} {}'.format(self.from_robot, self.from_index)
        # print 'tf  :{}'.format(self.constraint)

class InsubmapProcess:
    def __init__(self, submap_index, robot_id, init_pose_odom, init_pose_map, Descriptor, add_time=None ):
        self.submap_index = submap_index
        self.robot_id = robot_id
        self.submap_pose_odom = init_pose_odom #相比于odom坐标系的位姿(介入odom),一旦输入就不会改变
        self.submap_pose = init_pose_map #相比于map坐标系的位姿势,一般是从上一个submap递推得到,优化也优化这个
        self.add_time = add_time
        self.submap_point_clouds = np.zeros((0,3)) #TODO LOCK #讲道理访问这个东西的时候需要加锁
        self.clf = GaussianMixture(n_components=100, covariance_type='diag')
        self.GMMmodel = None
        self.Octomap = None
        self.freezed = False
        self.descriptor = None
        self.Descriptor = Descriptor
        # load weights
        #checkpoint = torch.load("model.ckpt")
        #saved_state_dict = checkpoint['state_dict']
        #self.Descriptor.model.load_state_dict(saved_state_dict)

    def insert_point(self, in_point_cloud, ifcheck = True):
        print(in_point_cloud.header.frame_id)
        # assert( 'odom' == in_point_cloud.header.frame_id)
        if (ifcheck):
            assert( 'submap_base_link_{}'.format(self.submap_index) == in_point_cloud.header.frame_id)

        gen_cloud_point = point_cloud2.read_points_list(in_point_cloud, field_names=("x", "y", "z"), skip_nans=True)
        gen_np_cloud = np.array(gen_cloud_point)
        
        self.submap_point_clouds = np.concatenate( (self.submap_point_clouds,gen_np_cloud), axis=0 ) #TODO LOCK
        self.filter_point() #每次添加了新点云都进行一次滤波

    def filter_point(self):
        #要删除重复的点云
        self.submap_point_clouds = voxel_filter(self.submap_point_clouds,0.05,"centroid")
        return self.submap_point_clouds

    def model_gmm(self): #用GMM完成地图构建,第一个版本暂时不研究地图,只考虑轨迹合并
        # TODO 直接用kmeans结果建
        self.GMMmodel = self.clf.fit(self.submap_point_clouds)
        return self.GMMmodel

    def pointmap2odom(self):
        showpoints = np2pointcloud2(self.submap_point_clouds,'submap_base_link_{}'.format(self.submap_index) )
        
    # 调试点云的时候的可视化    
        transform_submap_odom = TransformStamped()
        transform_submap_odom.transform = pose2trans(self.submap_pose_odom)
        transform_submap_odom.child_frame_id = 'submap_base_link_{}'.format(self.submap_index)
        transform_submap_odom.header.frame_id = 'odom' #实际上这个用来变换点云的函数并没有用到这个frameID

        outputpoints = do_transform_cloud(showpoints,transform_submap_odom)
        return outputpoints

    def gen_descriptor_pntcld(self):
        self.filter_point() #把点云进行滤波
        self.freezed = True
        # TODO 讲道理应该用点云生成描述子
        # use self.submap_point_clouds to extract
        # self.descriptor = np.array([self.submap_pose_odom.position.x, self.submap_pose_odom.position.y, self.submap_pose_odom.position.z, self.submap_pose_odom.orientation.x, self.submap_pose_odom.orientation.y, self.submap_pose_odom.orientation.z, self.submap_pose_odom.orientation.w ]) # 现在就是用 odom 的位置凑合一下
        #np.savetxt('/home/xuzl/catkin_ws/src/gmm_map_python/script/pc.txt', self.submap_point_clouds)
        #print('point cloud saved.')
        self.Descriptor.point_cloud = self.submap_point_clouds.copy()
        self.Descriptor.filter()
        self.descriptor = self.Descriptor.generate_descriptor()
        # normalize
        self.descriptor = self.descriptor / (np.linalg.norm(self.descriptor))
        tmp = 1

    def get_submap_pose_stamped(self):
        result = PoseStamped()
        result.pose = self.submap_pose
        result.header.frame_id = 'submap_base_link_{}'.format(self.submap_index)
        result.header.stamp = self.add_time
        return result

    def get_submap_pose_odom_stamped(self):
        result = PoseStamped()
        result.pose = self.submap_pose_odom
        result.header.frame_id = 'submap_base_link_{}'.format(self.submap_index)
        result.header.stamp = self.add_time
        return result


class TrajMapBuilder:
    def __init__(self, self_id):
        self.self_robot_id = self_id
        self.self_constraint = [] #优化的单个机器人内部约束的
        self.new_self_submap = True # 如果有是新的关键帧,这个会变成 True,在 callback_self_pointcloud 中接收到新的关键帧率.并把他改成 False
        self.new_self_loop = False # 如果有是新的关键帧,这个会变成 True, 在后端优化中,讲这个置为false
        self.new_self_submap_count = 0 #调试过程中记录帧数,用于新建submap
        self.new_self_loop_count = 0 #调试过程中记录帧数,用于新建loop
        self.tf_listener = tf.TransformListener()
        self.tf2_buffer = tf2_ros.Buffer()
        self.tf2_listener = tf2_ros.TransformListener(self.tf2_buffer)

        self.registrar = Registrar()

        self.current_submap_id = 0
        self.list_of_submap = []
        # import pdb; pdb.set_trace()
        self.baselink_frame = rospy.get_param('~baselink_frame','base_link')
        self.odom_frame = rospy.get_param('~odom_frame','odom')
        self.map_frame = rospy.get_param('~map_frame','map')

        self.br_ego_mapodom = tf.TransformBroadcaster()
        self.tans_ego_mapodom = TransformStamped()
        self.tans_ego_mapodom.header.frame_id = self.map_frame
        self.tans_ego_mapodom.child_frame_id = self.odom_frame
        self.tans_ego_mapodom.transform.rotation.w = 1

        self.path = Path()
        self.pathodom = Path()
        # self.octomap = octomap.OcTree(0.1)

        self.newsubmap_builder = None
        self.prefixsubmap_builder = None
        
        self.Descriptor = Descriptor(model_dir='/home/xuzhl/catkin_ws/src/gmm_map_python/script/model13.ckpt')

        # self.pose_pub = rospy.Publisher('pose', PoseStamped, queue_size=1)
        self.path_pub = rospy.Publisher('path', Path, queue_size=1)
        self.path_pubodom = rospy.Publisher('pathodom', Path, queue_size=1)
        self.submap_publisher = rospy.Publisher('/all_submap',String, queue_size=1)

        self.test_pc2_pub = rospy.Publisher('testpoints', PointCloud2,queue_size=1)
        self.self_pc_sub = rospy.Subscriber('sampled_points', PointCloud2, self.callback_self_pointcloud,queue_size=1)
        self.new_self_submap_sub = rospy.Subscriber('sampled_points', PointCloud2, self.callback_new_self_pointcloud,queue_size=1)
        # self.new_self_loop_sub = rospy.Subscriber('sampled_points', PointCloud2, self.callback_add_sim_loop)

        self.new_submap_listener = rospy.Subscriber('/all_submap', String, self.callback_submap_listener)

        self.transform_sub1_odom_br = tf2_ros.StaticTransformBroadcaster()

        #下面主要是为了处理其他的机器人的submap
        self.all_submap_lists = {}
        self.all_submap_locks = {}
        self.all_submap_lists['robot{}'.format(self.self_robot_id)] = self.list_of_submap
        self.all_submap_locks['robot{}'.format(self.self_robot_id)] = threading.Lock()
        
        self.all_self_constraint_lists = {} #优化的单个机器人内部约束的
        self.all_self_constraint_lists['robot{}'.format(self.self_robot_id)] = self.self_constraint
        self.inter_constraint_list = [] #优化的机器人之间约束的
        self.br_inter_robot = tf.TransformBroadcaster()

        self.tf_graph = TFGraph()
        self.tf_graph.new_tf_node(self.self_robot_id)

        self.match_thr = 0.25

        # self.backpubglobalpoint = threading.Thread(target=self.pointmap_single_thread)
        # self.backpubglobalpoint.setDaemon(True)
        # self.backpubglobalpoint.start()
        
        self.transform_firstsubmap_odom = None

        self.backt = threading.Thread(target=self.BackendThread)
        self.backt.setDaemon(True)
        self.backt.start()


    
    def detect_match_candidate_onemap(self, inputsubmap, targetsubmap):
        # TODO shency fix 
        # dist = np.linalg.norm(inputsubmap.descriptor[0:3] - targetsubmap.descriptor[0:3])
        # cosine_orient = np.dot(inputsubmap.descriptor[3:], targetsubmap.descriptor[3:].T)
        cosine_orient = np.dot(inputsubmap.descriptor, targetsubmap.descriptor.T)
        # print('cosine dist: {}'.format(cosine_orient))
        # return (1-cosine_orient)*dist
        return (1-cosine_orient)

    def detect_match_candidate_one_robot(self, inputsubmap, robotsubmaps):
        result = []
        for robotsubmap in robotsubmaps:
            result.append(self.detect_match_candidate_onemap(inputsubmap, robotsubmap) )
        return np.array(result)

    def detect_match_candidate(self, inputsubmap): # 如果输入的 是自己的 submap 需要和其他所有人detection
        result = {}
        for robotname in self.all_submap_lists.keys(): #遍历每个机器人的轨迹,找到可能的阈值
            if robotname == "robot{}".format(inputsubmap.robot_id): #寻找自己的回环
                thisrobotsubmap = self.all_submap_lists[robotname]
                result[robotname] = self.detect_match_candidate_one_robot(inputsubmap, thisrobotsubmap[:-10])  #过于临近的关键帧就不要匹配了费劲
            if robotname != "robot{}".format(inputsubmap.robot_id): #寻找自己other的回环
                thisrobotsubmap = self.all_submap_lists[robotname]
                result[robotname] = self.detect_match_candidate_one_robot(inputsubmap, thisrobotsubmap) 
        return result
        
    def submap_registration(self,inputsubmap,tosubmap): #输入两个子地图,输出两个地图的变换
        T_to_input = self.registrar.registration2(inputsubmap.submap_point_clouds, tosubmap.submap_point_clouds)
        # T_to_input.transform.translation.z = 1
        # T_to_input.transform.rotation.w = 1
        T_to_input.header.frame_id = 'submap_base_link_{}'.format(tosubmap.submap_index)
        T_to_input.child_frame_id = 'submap_base_link_{}'.format(inputsubmap.submap_index)
        # print T_to_input
        return T_to_input

    def submaps_to_constraint(self,inputsubmap,tosubmap): #输入两个子地图,直接构造成constraint
        trans_tmp = self.submap_registration( inputsubmap, tosubmap )
        result = ConstraintTransform(inputsubmap.robot_id, inputsubmap.submap_index, tosubmap.robot_id, tosubmap.submap_index, trans_tmp ) #都是记录 from 是新的 to 是旧的
        return result

    def callback_submap_listener(self, data): #主要是为了处理回环,包括自己的回环和别人的回环  
        recvstr = data.data
        recvsubmap = pickle.loads(recvstr)
        # print(recvsubmap.robot_id)
        # print(recvsubmap.submap_index)
        # print(recvsubmap)
        current_robot_id="robot{}".format(recvsubmap.robot_id)
        if not self.all_submap_lists.has_key(current_robot_id): #如果听到的是全新的机器人,就新增一个
            self.all_submap_locks[current_robot_id] = threading.Lock()
            self.all_submap_lists[current_robot_id] = []
            self.all_self_constraint_lists[current_robot_id] = []
            self.tf_graph.new_tf_node(recvsubmap.robot_id)
            print pygraph.graph_to_dot(self.tf_graph)
        if not (current_robot_id == 'robot{}'.format(self.self_robot_id) ): #如果听到的不是自己机器人的地图,就保存下来(自己的地图已经保存过了)
            # 计算和前一帧的 constraint
            self.all_submap_locks[current_robot_id].acquire()
            if len(self.all_submap_lists[current_robot_id]) != 0:
                cur_odom_base = pose2Rigidtrans(recvsubmap.submap_pose_odom, 'submap_base_link_{}'.format(recvsubmap.submap_index), 'robot{}/odom'.format(recvsubmap.robot_id))
                pre_odom_base = pose2Rigidtrans(self.all_submap_lists[current_robot_id][-1].submap_pose_odom, 'submap_base_link_{}'.format(self.all_submap_lists[current_robot_id][-1].submap_index), 'robot{}/odom'.format(recvsubmap.robot_id))                
                Tmsg_pre_cur = pre_odom_base.inverse()*cur_odom_base #计算from当前 to 上一帧
                T_pre_cur = Rigidtrans2transstamped(Tmsg_pre_cur)
                new_constraint = ConstraintTransform(recvsubmap.robot_id,recvsubmap.submap_index,recvsubmap.robot_id, self.all_submap_lists[current_robot_id][-1].submap_index, T_pre_cur)
                self.all_self_constraint_lists[current_robot_id].append(new_constraint)

            self.all_submap_lists[current_robot_id].append(recvsubmap)
            self.all_submap_locks[current_robot_id].release()
        
        # if (current_robot_id == 'robot{}'.format(self.self_robot_id) ): #如果收到的是自己的submap,就去和包括自己在内的所有轨迹进行匹配
        match = self.detect_match_candidate(recvsubmap)
        for robotname in match.keys():
            if robotname == current_robot_id: #寻找自己的回环
                # print(match[robotname])
                match_valid = match[robotname] < self.match_thr
                # print(match_valid)
                match_index = np.where(match_valid)
                print('match_index[0]:' + str(match_index[0]))
                # if len(match_valid) > 0: #说明有自身回环
                for valid_index in match_index[0]: #因为只有一个维度
                    innerconstraint = self.submaps_to_constraint(recvsubmap, self.all_submap_lists[robotname][valid_index] )
                    self.all_self_constraint_lists[current_robot_id].append(innerconstraint)
                    self.new_self_loop = True
            else: #寻找机器人之间的约束
                # print(match[robotname])
                match_valid = match[robotname] < self.match_thr
                # print(match_valid)
                match_index = np.where(match_valid)
                print('match_index[0]:' + str(match_index[0]))
                # if len(match_valid[0]) > 0: #说明有约束 
                for valid_index in match_index[0]: #因为只有一个维度
                    innerconstraint = self.submaps_to_constraint(recvsubmap, self.all_submap_lists[robotname][valid_index] )
                    # innerconstraint.show()
                    self.inter_constraint_list.append(innerconstraint)

                    match_set = None
                    curr_set = None
                    if self.tf_graph.get_tf(innerconstraint.from_robot, innerconstraint.to_robot) == None:
                        curr_map_base = pose2Rigidtrans(recvsubmap.submap_pose, 'submap_base_link_{}'.format(recvsubmap.submap_index), 'robot{}/map'.format(recvsubmap.robot_id))
                        match_map_base = pose2Rigidtrans(self.all_submap_lists[robotname][valid_index].submap_pose, 'submap_base_link_{}'.format(self.all_submap_lists[robotname][valid_index].submap_index), 'robot{}/map'.format(innerconstraint.to_robot))
                        inter_map_cons = match_map_base * transstamp2Rigidtrans(innerconstraint.constraint) * curr_map_base.inverse()
                        self.tf_graph.new_tf_edge(innerconstraint.from_robot, innerconstraint.to_robot, inter_map_cons)
                        print pygraph.graph_to_dot(self.tf_graph)
                        # transresult = Rigidtrans2transstamped(inter_map_cons)
                        # transresult.header.stamp = rospy.Time.now()
                        # print transresult
                        # self.br_inter_robot.sendTransformMessage(transresult)

                    self.new_self_loop = True

                        
                        


            



    def refresh_ego_mapodom_tf(self): #这一帧,根据MAP和ODOM最近的submap,计算map和odom的偏差
        # import pdb; pdb.set_trace()
        odom_last_pose = self.newsubmap_builder.submap_pose_odom
        map_last_pose = self.newsubmap_builder.submap_pose
        T_odom_nowpose = pose2Rigidtrans(odom_last_pose,'current',self.odom_frame)
        T_map_nowpose = pose2Rigidtrans(map_last_pose,'current',self.map_frame)
        T_map_odom =  T_map_nowpose * T_odom_nowpose.inverse()
        # print(T_map_odom)
        transresult = Rigidtrans2transstamped(T_map_odom)
        self.tans_ego_mapodom = transresult
        self.tans_ego_mapodom.header.stamp = rospy.Time.now()
        # print(self.tans_ego_mapodom)
        self.br_ego_mapodom.sendTransformMessage(self.tans_ego_mapodom)

        tmp = 1

    def BackendThread(self):
        while True:
            time.sleep(2)
            print("BackendOpt..........running")
            if self.new_self_loop: #只有找到新的loop才进行后端优化
                # for cons in self.inter_constraint_list:
                #     cons.show()
                assert(len(self.self_constraint) > 0)
                self.new_self_loop = False

                for lock in self.all_submap_locks.values():
                    lock.acquire()
                consopt = ConstrOptimizer(self.self_robot_id, self.all_submap_lists, self.all_self_constraint_lists, self.inter_constraint_list, self.tf_graph, self.newsubmap_builder, '/tmp/testg2o_{}'.format(self.self_robot_id))
                consopt.constructproblem()
                for lock in self.all_submap_locks.values():
                    lock.release()

                self.refresh_ego_mapodom_tf()

    def callback_new_self_pointcloud(self, data): #这个函数负责确定何时开始一个新的关键帧率,目前只是用来调试
        if (self.new_self_submap_count < 40):
            self.new_self_submap_count += 1
        else:
            self.new_self_submap = True # 当设置为True说明有一个新的关键帧
            self.new_self_submap_count = 0

    def callback_add_sim_loop(self, data):
        # self.new_self_loop_count += 1
        if (self.new_self_loop_count == 0 and len(self.list_of_submap) == 20):
            self.new_self_loop_count = 0
            T_first_cur = TransformStamped()
            T_first_cur.transform.rotation.w = 1
            new_constraint = ConstraintTransform(self.self_robot_id,self.newsubmap_builder.submap_index,self.self_robot_id,2,T_first_cur )
            self.self_constraint.append(new_constraint)
            self.new_self_loop = True # 当设置为 True 说明有一个新的关键帧
            tmp = 1

    def callback_self_pointcloud(self, data): #监听pointcloud,自己新建地图,并且保存对应的 odom.
        # print("get callback_self_pointcloud")
        assert isinstance(data, PointCloud2)
        # print("PointCloud2 OK")
        pointheader = data.header
        pointtime = pointheader.stamp
        if(pointtime < rospy.Time.now() - rospy.Duration(0.2) ):
            return #这个主要是为了应对输入的点云太快的问题
        # pointtime = rospy.Time(0)
        # print(self.baselink_frame)
        # print(pointtime)
        # print(rospy.Time.now() )
        self.tf_listener.waitForTransform(self.baselink_frame,pointheader.frame_id,pointtime,rospy.Duration(0.11))
        transform_base_camera = self.tf2_buffer.lookup_transform(self.baselink_frame,pointheader.frame_id, pointtime) #将点云转换到当前 base_link 坐标系的变换(一般来说是固定的), 查询出来的是, source 坐标系在 target 坐标系中的位置.

        baselink_pointcloud = do_transform_cloud(data,transform_base_camera) #将 source(camera) 的点云 变换到 taget(base_link) 坐标系


        self.tf_listener.waitForTransform(self.odom_frame,self.baselink_frame,pointtime,rospy.Duration(0.11))
        transform_odom_base = self.tf2_buffer.lookup_transform(self.odom_frame,self.baselink_frame, pointtime) #得到了 baselink 在odom 坐标系中的位置
        if self.new_self_submap: #说明要增加一个新的关键帧
            self.all_submap_locks['robot{}'.format(self.self_robot_id)].acquire()

            self.prefixsubmap_builder = self.newsubmap_builder

            # self.con
            self.newsubmap_builder = InsubmapProcess( self.current_submap_id, self.self_robot_id, trans2pose(transform_odom_base.transform), trans2pose(transform_odom_base.transform), self.Descriptor, pointtime )
            baselink_pointcloud.header.frame_id = 'submap_base_link_{}'.format(self.newsubmap_builder.submap_index)
            print("self.new_self_submap_{}".format(self.newsubmap_builder.submap_index))
            self.newsubmap_builder.insert_point(baselink_pointcloud) #只调试轨迹的过程中,暂时不需要添加地图

            if not (self.prefixsubmap_builder == None): #如果不是第一帧,就需要把之前的帧给保存下来
                self.prefixsubmap_builder.gen_descriptor_pntcld() #生成对应的描述子(基于点云生成)
                self.list_of_submap.append(self.prefixsubmap_builder) #保存之前的submap
                print("save_self_submap_{}".format(self.prefixsubmap_builder.submap_index))
                # 计算和前一帧的 constraint
                cur_odom_base = transstamp2Rigidtrans(transform_odom_base)
                pre_odom_base = pose2Rigidtrans(self.prefixsubmap_builder.submap_pose_odom , 'submap_base_link_{}'.format(self.prefixsubmap_builder.submap_index) ,self.odom_frame)                
                Tmsg_pre_cur = pre_odom_base.inverse()*cur_odom_base #计算from当前 to 上一帧
                T_pre_cur = Rigidtrans2transstamped(Tmsg_pre_cur)
                T_pre_cur.child_frame_id = 'submap_base_link_{}'.format(self.current_submap_id)

                new_constraint = ConstraintTransform(self.self_robot_id,self.current_submap_id,self.self_robot_id,self.prefixsubmap_builder.submap_index,T_pre_cur )
                self.self_constraint.append(new_constraint)
                
                new_pose_map = pose2Rigidtrans(self.prefixsubmap_builder.submap_pose,  'submap_base_link_{}'.format(self.prefixsubmap_builder.submap_index), self.map_frame  )*Tmsg_pre_cur #odompose直接读取,但是map pose 需要累加计算.

                self.newsubmap_builder.submap_pose = trans2pose(Rigidtrans2transstamped(new_pose_map).transform)

                pubsubmap = pickle.dumps(self.prefixsubmap_builder)
                self.submap_publisher.publish(pubsubmap) #将已经建好的submap广播出去
                tmp = 1
            
            self.all_submap_locks['robot{}'.format(self.self_robot_id)].release()


            self.current_submap_id+=1 #下一个关键帧ID增加
            self.new_self_submap = False #准备接收下一个新的关键帧
            
        else: #在已完成初始化的关键帧上做任务
            # print("在已完成初始化的关键帧上做任务")
            cur_odom_base = transstamp2Rigidtrans(transform_odom_base)
            sub_odom_base = pose2Rigidtrans(self.newsubmap_builder.submap_pose_odom , from_frame='submap_base_link_{}'.format(self.newsubmap_builder.submap_index) ,to_frame=self.odom_frame)
            
            Tmsg_sub_cur = sub_odom_base.inverse()*cur_odom_base
            T_sub_cur = Rigidtrans2transstamped(Tmsg_sub_cur)

            sub_pointcloud = do_transform_cloud(baselink_pointcloud,T_sub_cur) #在 submap pose 坐标系中的点云
            sub_pointcloud.header.frame_id = 'submap_base_link_{}'.format(self.newsubmap_builder.submap_index)
            self.newsubmap_builder.insert_point(sub_pointcloud) #只调试轨迹的过程中,暂时不需要添加地图


        #这里是不论接收到哪些帧,都需要的操作
        self.gen_submap_path()
        self.gen_submap_pathodom()
        self.path_pub.publish(self.path) #这个是打印轨迹,相对map
        self.path_pubodom.publish(self.pathodom) #这个是打印轨迹,相对odom
        self.refresh_ego_mapodom_tf() #更新map和pose的tf树
            
        # 调试点云的时候的可视化 
            # showpoints = np2pointcloud2(self.newsubmap_builder.submap_point_clouds,'submap_base_link_{}'.format(self.newsubmap_builder.submap_index) )   
            # transform_submap_odom = TransformStamped()
            # transform_submap_odom.transform = pose2trans(self.newsubmap_builder.submap_pose_odom)
            # transform_submap_odom.child_frame_id = 'submap_base_link_{}'.format(self.newsubmap_builder.submap_index)
            # transform_submap_odom.header.frame_id = self.odom_frame

            # outputpoints = do_transform_cloud(showpoints,transform_submap_odom)

            # self.test_pc2_pub.publish(outputpoints)


        # robot_pose = PoseStamped()
        # robot_pose.pose = trans2pose(transform_odom_base.transform)
        # robot_pose.header.frame_id = self.odom_frame
        # robot_pose.header.stamp = rospy.Time.now()

        # self.pose_pub.publish(robot_pose)
        # self.path.poses.append(robot_pose)
        # self.path.header = robot_pose.header
        # self.path_pub.publish(self.path)

    def gen_submap_path(self):
        self.path = Path()
        self.path.header.frame_id = self.map_frame
        self.path.header.stamp = rospy.Time.now()
        for submap in self.list_of_submap:
            self.path.poses.append(submap.get_submap_pose_stamped() )
        self.path.poses.append(self.newsubmap_builder.get_submap_pose_stamped() )
        return self.path

    def gen_submap_pathodom(self):
        self.pathodom = Path()
        self.pathodom.header.frame_id = self.odom_frame
        self.pathodom.header.stamp = rospy.Time.now()
        for submap in self.list_of_submap:
            self.pathodom.poses.append(submap.get_submap_pose_odom_stamped() )
        self.pathodom.poses.append(self.newsubmap_builder.get_submap_pose_odom_stamped() )
        return self.pathodom

    def pointmap_merge_single(self): #将每一帧的点云都合并在一起可视化出来
        # final_result = PointCloud2()
        tmpmap_builder = InsubmapProcess(0,0,Pose(),Pose(),self.Descriptor)
        # if (len(self.list_of_submap) <= 1):
        #     return 
        for submapinst in self.list_of_submap:
            showpoints = submapinst.pointmap2odom() #不同的点云都在odom坐标系中
            tmpmap_builder.insert_point(showpoints, False)
        
        showpoints = np2pointcloud2(tmpmap_builder.submap_point_clouds,self.odom_frame)            
        # # 调试点云的时候的可视化    
        self.test_pc2_pub.publish(showpoints)

    def pointmap_single_thread(self):
        while (1):
            time.sleep(2)
            print("pointmap_thread")
            self.pointmap_merge_single()




def main():
    rospy.init_node('pcl_listener', anonymous=True)
    Robot1 = TrajMapBuilder(rospy.get_param('~robot_id',1))
    rospy.spin()

if __name__ == "__main__":
    main()