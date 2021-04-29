# -*- coding: utf-8 -*
import pcl
import numpy as np
from autolab_core import RigidTransform
from geometry_msgs.msg import Pose, PoseStamped, Transform, TransformStamped
from GMMmap import GMMFrame
import scipy.optimize as opt
import transforms3d
import rospy

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
        T = self.T[1]
        T = RigidTransform(T[0:3,0:3], T[0:3,3].reshape(3))
        pose = T.pose_msg
        T_to_input = TransformStamped()
        T_to_input.transform.rotation = pose.orientation
        T_to_input.transform.translation = pose.position
        return T_to_input, self.T[3] #icp的fitness是目标到源的平均欧式距离

    # def registration(source, target, n_components, method, seed_r):
    def GMMreg(self,GMMFrame1,GMMFrame2):
        from_gmm=GMMFrame1.Frame_to_SklearnMixture()
        to_gmm=GMMFrame2.Frame_to_SklearnMixture()
        from_params = from_gmm._get_parameters()
        to_params = to_gmm._get_parameters()
        
        # from_params = GMMFrame1._get_parameters()
        # to_params = GMMFrame2._get_parameters()
        print("in GMMreg function:",from_params[2].shape)
        print("in GMMreg function:",to_params[2].shape)

        if len(from_params[2].shape) != 3:
            from_params = list(from_params)
            to_params = list(to_params)
            from_cov = np.zeros((from_params[2].shape[0],3,3))
            to_cov = np.zeros((to_params[2].shape[0],3,3))
            for i in range(from_params[2].shape[0]):
                from_cov[i] = np.diag(from_params[2][i])
            for i in range(to_params[2].shape[0]):
                to_cov[i] = np.diag(to_params[2][i])
            from_params[2] = from_cov
            to_params[2] = to_cov

        def loss_gmm_to_gmm(x):
            qs = x[:4]
            ts = x[4:]
            qs = qs/np.linalg.norm(qs)
            Ms = transforms3d.quaternions.quat2mat(qs)

            y = from_params[1] - np.matmul(Ms , to_params[1].T).T - ts
            # y = from_params[1] - (Ms @ to_params[1].T).T - ts
            sigma = from_params[2] + np.matmul(np.matmul(Ms , to_params[2]) , Ms.T)
            # sigma = from_params[2] + Ms @ to_params[2] @ Ms.T
            inv_sigma = np.array([np.linalg.inv(s) for s in sigma])

            dist = np.matmul(np.matmul(y[:,None,:] , inv_sigma) , y[:,:,None])
            # dist = y[:,None,:] @ inv_sigma @ y[:,:,None]
            f = np.sqrt(np.linalg.det(inv_sigma))* from_params[0] * to_params[0] * np.exp(-0.5 * dist)
            return -f.sum()

        def loss_gmm_to_gmm_raw(x,mix_num): #一共9维
            # print(x,mix_num)
            qs = x[:4]
            ts = x[4:]
            mix_num1=mix_num[0]
            mix_num2=mix_num[1]
            qs = qs/np.linalg.norm(qs)
            Ms = transforms3d.quaternions.quat2mat(qs)
            total = 0
            # print("start loop",mix_num1,mix_num2)
            for i in range(mix_num1):
                for j in range(mix_num2):
                    # print("-----in loop:",i,j)
                    y = from_params[1][i:i+1].T - np.matmul(Ms,to_params[1][j:j+1].T) - ts[:,None]
                    # y = from_params[1][i:i+1].T - (Ms @ to_params[1][j:j+1].T) - ts[:,None]
                    sigma = from_params[2][i] + np.matmul(np.matmul(Ms , to_params[2][j]) , Ms.T)
                    # sigma = from_params[2][i] + Ms @ to_params[2][j] @ Ms.T
                    inv_sigma = np.linalg.pinv(sigma)
                    dist = np.matmul(np.matmul(y.T , inv_sigma) , y)
                    # dist = y.T @ inv_sigma @ y
                    dist = dist.sum()
                    f = np.sqrt(np.linalg.det(inv_sigma))* from_params[0][i] * to_params[0][j] * np.exp(-0.5 * dist)
                    total += f
            # print(total)
            return -total

        def loss_gmm_to_gmm_raw2(x,mix_num):
            qs = x[:4]
            ts = x[4:]
            mix_num1=mix_num[0]
            qs = qs/np.linalg.norm(qs)
            Ms = transforms3d.quaternions.quat2mat(qs)
            total = 0
            for i in range(mix_num1):
                y = from_params[1][i:i+1].T - np.matmul(Ms,to_params[1].T) - ts[:,None]
                # y = from_params[1][i:i+1].T - (Ms @ to_params[1].T) - ts[:,None]
                y = y.T
                sigma = from_params[2][i] + np.matmul(np.matmul(Ms , to_params[2]) , Ms.T)
                # sigma = from_params[2][i].T + Ms @ to_params[2] @ Ms.T
                inv_sigma = np.linalg.inv(sigma)
                dist = np.squeeze(np.matmul(np.matmul(y[:,None,:] , inv_sigma) , y[:,:,None]))
                # dist = np.squeeze(y[:,None,:] @ inv_sigma @ y[:,:,None])
                f = np.sqrt(np.linalg.det(inv_sigma))* from_params[0][i] * to_params[0] * np.exp(-0.5 * dist)
                total += f.sum()
            
            # print(total)

            return (-total)/(mix_num[0]*mix_num[1])

        def loss_gmm_to_gmm_raw3(x):
            qs = x[:4]
            ts = x[4:]
            qs = qs/np.linalg.norm(qs)
            Ms = transforms3d.quaternions.quat2mat(qs)
            total = 0
            a = from_params[1] - ts
            
            tmp=np.dot(Ms,to_params[1].T) #二维乘二维，用np.dot即可
            # print("mat_mul.shape:",Ms.shape,to_params[1].T.shape,tmp.shape)

            b = (tmp).T
            # b = (Ms @ to_params[1].T).T
            y = a - b[:, None]
            y = y.reshape((-1,3))
            
            # print("---",np.dot(Ms,to_params[2]).transpose(1,0,2).shape)
            print("mat_mul.shape:",Ms.shape,to_params[2].shape,Ms.T.shape)
            tmp=mat_mul32(mat_mul23(Ms,to_params[2]),Ms.T)[:,None]
            
            sigma = from_params[2] + tmp
            # sigma = from_params[2] + (Ms @ to_params[2] @ Ms.T)[:,None]
            sigma = sigma.reshape((-1,3,3))
            inv_sigma = np.linalg.inv(sigma)

            print(y[:,None,:].shape,inv_sigma.shape,y[:,:,None].shape)
            tmp=mat_mul33(mat_mul33(y[:,None,:],inv_sigma),y[:,:,None])
            print("mat_mul.shape:",y[:,None,:].shape,inv_sigma.shape,y[:,:,None].shape,tmp.shape)
            dist = np.squeeze(tmp)
            # dist = np.squeeze(y[:,None,:] @ inv_sigma @ y[:,:,None])
            dist = dist.reshape((a.shape[0],b.shape[0]))
            scaler = 1#np.sqrt(np.linalg.det(inv_sigma)).reshape((a.shape[0],b.shape[0]))
            f = scaler * from_params[0][None,:] * to_params[0][:,None] * np.exp(-0.5 * dist)
            total += f.sum()
            return -total
        
        print("---------------------start minimize!")
        t1=rospy.Time.now()
        # res = opt.minimize(loss_gmm_to_gmm,np.array([1,0,0,0,0,0,0,]),method=None,options={'maxiter':1})
        res = opt.minimize(loss_gmm_to_gmm_raw2,np.array([1,0,0,0,0,0,0,]),args=([from_gmm.n_components,to_gmm.n_components]),tol=1.0e-2,method=None,options={'maxiter':1})
        # res = opt.minimize(loss_gmm_to_gmm_raw2,np.array([1,0,0,0,0,0,0,]),args=([GMMFrame1.n_components,GMMFrame2.n_components]),tol=1.0e-2,method=None,options={'maxiter':1})

        t2=rospy.Time.now()
        print("--------------------finish minimize!",(t2-t1))

        fitness=-res.fun #目标函数的值
        x = res.x
        print(x)
        qe = x[:4]
        qe = qe/np.linalg.norm(qe)
        # tmp=qe[3]
        # qe[0]=qe[3]
        # qe[0]=tmp
        te = x[4:]
        T_to_input = TransformStamped()
        pose=Pose()
        pose.orientation.w=qe[0]
        pose.orientation.x=qe[1]
        pose.orientation.y=qe[2]
        pose.orientation.z=qe[3]
        pose.position.x=te[0]
        pose.position.y=te[1]
        pose.position.z=te[2]
        T_to_input.transform.rotation = pose.orientation
        T_to_input.transform.translation = pose.position
        return T_to_input,fitness


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