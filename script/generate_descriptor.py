import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.autograd import Variable
from torch.backends import cudnn

# from loading_pointclouds import *
import PointNetVlad as PNV

import config as cfg

cudnn.enabled = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Descriptor:
    def __init__(self, point_cloud=None, point_num=4096,model_dir=None):
        self.model = PNV.PointNetVlad(global_feat=True, feature_transform=True,\
            max_pool=False, output_dim=cfg.FEATURE_OUTPUT_DIM, num_points=cfg.NUM_POINTS)
        self.model.to(device)
        print(device)
        self.point_cloud = point_cloud
        self.point_num = point_num
        self.filtered_pc = None
        # self.filter()
        print("load model weights from  {}\n".format(model_dir))
        if model_dir:
            model_dict = torch.load(model_dir)
            self.model.load_state_dict(model_dict)
        #self.filter()
    def filter(self):
        print("point_cloud: {}\n".format(self.point_cloud.shape[0]))
        idx = np.random.choice(self.point_cloud.shape[0], self.point_num, replace=True)
        self.filtered_pc = self.point_cloud[idx, :]
        mean = np.mean(self.filtered_pc, axis=0)
        mean = mean[np.newaxis, :]
        self.filtered_pc -= mean
        self.filtered_pc /= np.max(np.abs(self.filtered_pc))
        return self.filtered_pc
        # nums = 0
        # idx = list()
        # # points = copy.copy(self.point_cloud)
        # filtered_points = np.zeros((self.point_num, 3))
        # dist = np.zeros(self.point_cloud.shape[0]) + float("inf")
        # dist[0] = 0
        # idx.append(0)
        # # filtered_points[0, :] = self.point_cloud[0, :]
        # nums += 1
        # while nums < self.point_num:
        #     for i in range(self.point_cloud.shape[0] - nums):
        #         if i in idx:
        #             continue
        #         dist = min(np.linalg.norm(self.point_cloud - self.point_cloud[idx[nums-1, :]], axis=1)
        #         dist[i] = min(np.linalg.norm(self.point_cloud[i, :] - filtered_points[nums - 1, :]), dist[i])
            
        #     s = np.argsort(dist)
        #     for i in np.arange(self.point_cloud.shape[0]-1, -1, -1):
        #         if s[i] in idx:
        #             continue
        #         idx.append(s[i])
        #         nums+=1
        #         break
        #     print(nums)
        # return self.point_cloud[idx, :]

    def generate_descriptor(self):
        model_in = Variable(torch.tensor(self.filtered_pc.reshape((1, 1, self.point_num, 3)), dtype=torch.float)).to(device)
        # print(model_in.shape)
        self.model.eval()
        with torch.no_grad():
            model_out = self.model(model_in)
        return model_out.cpu().numpy()
            

if __name__ == '__main__':
    pc = np.loadtxt('pc.txt')
    # pc = pc[1:10000, :]

    D = Descriptor(point_cloud=pc, model_dir='model13.ckpt')
    print(D.generate_descriptor())

        


