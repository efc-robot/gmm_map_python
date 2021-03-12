import pcl
import numpy as np
from autolab_core import RigidTransform
from geometry_msgs.msg import Pose, PoseStamped, Transform, TransformStamped

class Registrar:
    def __init__(self):
        self.icp = pcl.IterativeClosestPoint()
        self.point_cloud1 = pcl.PointCloud()
        self.point_cloud2 = pcl.PointCloud()

    def registration(self, point_cloud1, point_cloud2):
        self.T = self.icp.icp(point_cloud1, point_cloud2)
        return self.T

    def registration2(self, point_cloud1_np, point_cloud2_np):
        point_cloud1_np = point_cloud1_np.astype(np.float32)
        point_cloud2_np = point_cloud2_np.astype(np.float32)
        point_cloud1 = pcl.PointCloud(point_cloud1_np)
        point_cloud2 = pcl.PointCloud(point_cloud2_np)
        self.T = self.icp.icp(point_cloud1, point_cloud2)
        print self.T
        T_to_input = None
        if self.T[0]:
            T = self.T[1]
            T = RigidTransform(T[0:3,0:3], T[0:3,3].reshape(3))
            pose = T.pose_msg
            T_to_input = TransformStamped()
            T_to_input.transform.rotation = pose.orientation
            T_to_input.transform.translation = pose.position

        return T_to_input, self.T[3]


if __name__ == '__main__':
    reg = Registrar()
    pc1 = np.eye(3, dtype=np.float32)
    pc2 = np.eye(3, dtype=np.float32) + np.ones([3,3], dtype=np.float32)
    print "pc1:" + str(pc1)
    print "pc2:" + str(pc2)
    T = reg.registration2(pc1, pc2)
    # print "T:" + str(T)
    # T = T[1]
    # print "r:" + str(T[0:3,0:3])
    # print "t:" + str(T[0:3,3])
    # T_to_input = RigidTransform(T[0:3,0:3], T[0:3,3].reshape(3))
    print T