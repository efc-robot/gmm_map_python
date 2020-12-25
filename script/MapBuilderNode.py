#! /usr/bin/env python
# -*- coding: utf-8 -*-

from std_msgs.msg import Header
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
import gtsam

COVAR_STR = "1.000000 0.000000 0.000000 0.000000 0.000000 0.000000   1.000000 0.000000 0.000000 0.000000 0.000000   1.000000 0.000000 0.000000 0.000000   1.000000 0.000000 0.000000   1.000000 0.000000   1.000000"


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


class ConstrOptimizer:
    def __init__(self, robot_id, pose_map, constr, newsubmap_builder, g2ofname):
        self.robot_id = robot_id
        self.pose_map = pose_map
        self.constr = constr
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
                vertexID = robotID2poseID(self.robot_id, VERTEXpose.submap_index)

                print >> g2ofile, "VERTEX_SE3:QUAT {pointID} {tx} {ty} {tz} {rx} {ry} {rz} {rw}".format(pointID=vertexID,tx=VERTEXpose.submap_pose.position.x,ty=VERTEXpose.submap_pose.position.y,tz=VERTEXpose.submap_pose.position.z, 
                    rx = VERTEXpose.submap_pose.orientation.x, ry = VERTEXpose.submap_pose.orientation.y, rz = VERTEXpose.submap_pose.orientation.z, rw = VERTEXpose.submap_pose.orientation.w)

    def AddConstr2g2oEdge(self, g2ofname, constr):
        with open(g2ofname, "a") as g2ofile:
            for EDGEtrans in constr:
                id1 = robotID2poseID(EDGEtrans.from_robot,EDGEtrans.from_index)
                id2 = robotID2poseID(EDGEtrans.to_robot, EDGEtrans.to_index)
                print >> g2ofile, "EDGE_SE3:QUAT {id1} {id2} {tx} {ty} {tz} {rx} {ry} {rz} {rw}  {COVAR_STR}".format(id1=id1,id2=id2, tx =EDGEtrans.constraint.transform.translation.x , ty =EDGEtrans.constraint.transform.translation.y , tz =EDGEtrans.constraint.transform.translation.z, rx =EDGEtrans.constraint.transform.rotation.x, ry =EDGEtrans.constraint.transform.rotation.y, rz =EDGEtrans.constraint.transform.rotation.z, rw =EDGEtrans.constraint.transform.rotation.w, COVAR_STR=COVAR_STR)

    def setPoseWithSubmapID(self, robotID, submapID, resultpose):
        for cursubmap in self.pose_map:
            map_robotID = cursubmap.robot_id
            map_submapID = cursubmap.submap_index
            if map_robotID == robotID and map_submapID == submapID:
                tmpRigit = RigidTransform(resultpose.rotation().quaternion(), resultpose.translation().vector())
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
        self.Posemap2g2oVertex(self.g2ofname, self.pose_map)
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


class InsubmapProcess:
    def __init__(self, submap_index, robot_id, init_pose_odom, init_pose_map, add_time=None ):
        self.submap_index = submap_index
        self.robot_id = robot_id
        self.submap_pose_odom = init_pose_odom #相比于odom坐标系的位姿(介入odom),一旦输入就不会改变
        self.submap_pose = init_pose_map #相比于map坐标系的位姿势,一般是从上一个submap递推得到,优化也优化这个
        self.add_time = add_time
        self.submap_point_clouds = np.zeros((0,3)) #TODO LOCK #讲道理访问这个东西的时候需要加锁
        self.clf = GaussianMixture(n_components=100, covariance_type='diag')
        self.GMMmodel = None
        self.Octomap = None

    
    def insert_point(self, in_point_cloud):
        print(in_point_cloud.header.frame_id)
        # assert( 'odom' == in_point_cloud.header.frame_id)
        assert( 'submap_base_link_{}'.format(self.submap_index) == in_point_cloud.header.frame_id)

        gen_cloud_point = point_cloud2.read_points_list(in_point_cloud, field_names=("x", "y", "z"), skip_nans=True)
        gen_np_cloud = np.array(gen_cloud_point)
        
        self.submap_point_clouds = np.concatenate( (self.submap_point_clouds,gen_np_cloud), axis=0 ) #TODO LOCK

    def model_gmm(self): #用GMM完成地图构建,第一个版本暂时不研究地图,只考虑轨迹合并
        self.GMMmodel = self.clf.fit(self.submap_point_clouds)
        return self.GMMmodel


class TrajMapBuilder:
    def __init__(self, self_id):
        self.self_robot_id = self_id
        self.self_submap_pose_odom = [] #从里程计来的位姿,align输出每个关键帧
        self.self_constraint = [] #优化的单个机器人内部约束的
        self.self_submap_pose = [] # 优化过后的位姿势
        self.new_self_submap = True # 如果有是新的关键帧,这个会变成 True,在 callback_self_pointcloud 中接收到新的关键帧率.并把他改成 False
        self.new_self_loop = False # 如果有是新的关键帧,这个会变成 True, 在后端优化中,讲这个置为false
        self.new_self_submap_count = 0 #调试过程中记录帧数,用于新建submap
        self.new_self_loop_count = 0 #调试过程中记录帧数,用于新建loop
        self.tf_listener = tf.TransformListener()
        self.tf2_buffer = tf2_ros.Buffer()
        self.tf2_listener = tf2_ros.TransformListener(self.tf2_buffer)

        self.current_submap_id = 0
        self.list_of_submap = []

        self.path = Path()

        self.newsubmap_builder = None
        self.prefixsubmap_builder = None


        self.pose_pub = rospy.Publisher('/pose', PoseStamped, queue_size=1)
        self.path_pub = rospy.Publisher('/path', Path, queue_size=1)


        self.test_pc2_pub = rospy.Publisher('/testpoints', PointCloud2,queue_size=1)
        self.self_pc_sub = rospy.Subscriber('/sampled_points', PointCloud2, self.callback_self_pointcloud)
        self.new_self_submap_sub = rospy.Subscriber('/sampled_points', PointCloud2, self.callback_new_self_pointcloud)
        self.new_self_loop_sub = rospy.Subscriber('/sampled_points', PointCloud2, self.callback_add_sim_loop)

        self.backt = threading.Thread(target=self.BackendThread)
        self.backt.setDaemon(True)
        self.backt.start()

    def BackendThread(self):
        while True:
            time.sleep(2)
            print("BackendOpt..........running")
            if self.new_self_loop: #只有找到新的loop才进行后端优化
                assert(len(self.self_constraint) > 0)
                self.new_self_loop = False                  
                consopt = ConstrOptimizer(self.self_robot_id,self.list_of_submap,self.self_constraint, self.newsubmap_builder ,'/tmp/testg2o')
                consopt.constructproblem()

    def callback_new_self_pointcloud(self, data): #这个函数负责确定何时开始一个新的关键帧率,目前只是用来调试
        if (self.new_self_submap_count < 10):
            self.new_self_submap_count += 1
        else:
            self.new_self_submap = True # 当设置为True说明有一个新的关键帧
            self.new_self_submap_count = 0

    def callback_add_sim_loop(self, data):
        self.new_self_loop_count += 1
        if (self.new_self_loop_count == 40):
            self.new_self_loop_count = 0
            T_first_cur = TransformStamped()
            T_first_cur.transform.rotation.w = 1
            new_constraint = ConstraintTransform(self.self_robot_id,self.newsubmap_builder.submap_index,self.self_robot_id,0,T_first_cur )
            self.self_constraint.append(new_constraint)
            self.new_self_loop = True # 当设置为True说明有一个新的关键帧
            tmp = 1



    def callback_self_pointcloud(self, data): #监听pointcloud,自己新建地图,并且保存对应的 odom.
        print("get callback_self_pointcloud")
        assert isinstance(data, PointCloud2)
        pointheader = data.header
        pointtime = pointheader.stamp
        # pointtime = rospy.Time(0)
        self.tf_listener.waitForTransform('base_link',pointheader.frame_id,pointtime,rospy.Duration(0.005))
        transform_base_camera = self.tf2_buffer.lookup_transform('base_link',pointheader.frame_id, pointtime) #将点云转换到当前 base_link 坐标系的变换(一般来说是固定的), 查询出来的是, source 坐标系在 target 坐标系中的位置.

        baselink_pointcloud = do_transform_cloud(data,transform_base_camera) #将 source(camera) 的点云 变换到 taget(base_link) 坐标系


        if self.new_self_submap: #说明要增加一个新的关键帧
            print("self.new_self_submap")
            self.tf_listener.waitForTransform('odom','base_link',pointtime,rospy.Duration(0.005))
            transform_odom_base = self.tf2_buffer.lookup_transform('odom','base_link', pointtime) #得到了 baselink 在odom 坐标系中的位置
            
            self.prefixsubmap_builder = self.newsubmap_builder

                # self.con
            self.newsubmap_builder = InsubmapProcess( self.current_submap_id, self.self_robot_id, trans2pose(transform_odom_base.transform), trans2pose(transform_odom_base.transform), pointtime )
            baselink_pointcloud.header.frame_id = 'submap_base_link_{}'.format(self.newsubmap_builder.submap_index)
            self.newsubmap_builder.insert_point(baselink_pointcloud)

            if not (self.prefixsubmap_builder == None): #如果不是第一帧,就需要把之前的帧给保存下来
                self.list_of_submap.append(self.prefixsubmap_builder) #保存之前的submap
                # 计算和前一帧的 constraint
                cur_odom_base = transstamp2Rigidtrans(transform_odom_base)
                pre_odom_base = pose2Rigidtrans(self.prefixsubmap_builder.submap_pose_odom , 'submap_base_link_{}'.format(self.prefixsubmap_builder.submap_index) ,'odom')                
                Tmsg_pre_cur = pre_odom_base.inverse()*cur_odom_base
                T_pre_cur = Rigidtrans2transstamped(Tmsg_pre_cur)
                T_pre_cur.child_frame_id = 'submap_base_link_{}'.format(self.current_submap_id)

                new_constraint = ConstraintTransform(self.self_robot_id,self.current_submap_id,self.self_robot_id,self.prefixsubmap_builder.submap_index,T_pre_cur )
                self.self_constraint.append(new_constraint)
                
                new_pose_map = pose2Rigidtrans(self.prefixsubmap_builder.submap_pose,  'submap_base_link_{}'.format(self.prefixsubmap_builder.submap_index), "map"  )*Tmsg_pre_cur #odompose直接读取,但是map pose 需要累加计算.

                self.newsubmap_builder.submap_pose = trans2pose(Rigidtrans2transstamped(new_pose_map).transform)

                tmp = 1


            self.current_submap_id+=1 #下一个关键帧ID增加
            self.new_self_submap = False #准备接收下一个新的关键帧
            
        else: #在已完成初始化的关键帧上做任务
            self.tf_listener.waitForTransform('odom','base_link',pointtime,rospy.Duration(0.005))
            transform_odom_base = self.tf2_buffer.lookup_transform('odom','base_link', pointtime) #得到了 baselink 在odom 坐标系中的位置

            cur_odom_base = transstamp2Rigidtrans(transform_odom_base)
            sub_odom_base = pose2Rigidtrans(self.newsubmap_builder.submap_pose_odom , from_frame='submap_base_link_{}'.format(self.newsubmap_builder.submap_index) ,to_frame='odom')
            
            Tmsg_sub_cur = sub_odom_base.inverse()*cur_odom_base
            T_sub_cur = Rigidtrans2transstamped(Tmsg_sub_cur)

            sub_pointcloud = do_transform_cloud(baselink_pointcloud,T_sub_cur) #在 submap pose 坐标系中的点云
            sub_pointcloud.header.frame_id = 'submap_base_link_{}'.format(self.newsubmap_builder.submap_index)
            self.newsubmap_builder.insert_point(sub_pointcloud)


            showpoints = np2pointcloud2(self.newsubmap_builder.submap_point_clouds,'submap_base_link_{}'.format(self.newsubmap_builder.submap_index) )
            
        # 调试点云的时候的可视化    
            transform_submap_odom = TransformStamped()
            transform_submap_odom.transform = pose2trans(self.newsubmap_builder.submap_pose_odom)
            transform_submap_odom.child_frame_id = 'submap_base_link_{}'.format(self.newsubmap_builder.submap_index)
            transform_submap_odom.header.frame_id = 'odom'

            outputpoints = do_transform_cloud(showpoints,transform_submap_odom)

            self.test_pc2_pub.publish(outputpoints)


        robot_pose = PoseStamped()
        robot_pose.pose = trans2pose(transform_odom_base.transform)
        robot_pose.header.frame_id = 'odom'
        robot_pose.header.stamp = rospy.Time.now()

        self.pose_pub.publish(robot_pose)
        self.path.poses.append(robot_pose)
        self.path.header = robot_pose.header
        self.path_pub.publish(self.path)
            

def main():
    rospy.init_node('pcl_listener', anonymous=True)
    Robot1 = TrajMapBuilder(1)
    rospy.spin()

if __name__ == "__main__":
    main()