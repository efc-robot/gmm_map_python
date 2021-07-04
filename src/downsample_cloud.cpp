#include <iostream>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Path.h>
#include<tf/transform_listener.h>
#include <ros/duration.h>
#include<thread>
#include<mutex>
#include "math.h"
#include <climits>// give variable the biggest number

#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>	
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>

#include<Eigen/Dense>
#include <Eigen/Eigen>
#include <Eigen/StdVector>
#include <tf/transform_broadcaster.h>


#include <vector>
#include <ctime>
#include <boost/thread/thread.hpp>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/features/eigen.h>
#include <pcl/features/feature.h>
#include <pcl/features/normal_3d.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/features/boundary.h>
#include <pcl/filters/passthrough.h>

using namespace std;



class DownSamplePointCloud2{
public:
    void initFilter(ros::NodeHandle nh);
    void pointcloudCallback(const sensor_msgs::PointCloud2 img);
private:
    ros::Subscriber point_sub_;
    ros::Publisher point_pub_;
    pcl::VoxelGrid<pcl::PointXYZ> sor_;
    pcl::PointCloud<pcl::PointXYZ> cloud_input_;
    ros::Time current_time = ros::Time(0.1);

};

void DownSamplePointCloud2::pointcloudCallback(sensor_msgs::PointCloud2 img){
    std::cout<< ros::Time::now() -img.header.stamp <<std::endl;
    if ( img.header.stamp < ros::Time::now() - ros::Duration(0.5) ){

      std::cout<<"discord!"<<std::endl;
      return;
    }
    current_time = img.header.stamp;
    pcl::fromROSMsg(img, cloud_input_);
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(cloud_input_,cloud_input_, indices);
    sor_.setInputCloud(cloud_input_.makeShared());
    sor_.setLeafSize(0.1f, 0.1f, 0.1f);
    sor_.filter(cloud_input_);

    // //边缘检测
    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    // cloud = cloud_input_.makeShared();
    // std::cout << "points sieze is:" << cloud->size() << std::endl;
    // pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    // pcl::PointCloud<pcl::Boundary> boundaries;
    // pcl::BoundaryEstimation<pcl::PointXYZ, pcl::Normal, pcl::Boundary> est;
    // pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());    
    // pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;  //创建一个快速k近邻查询,查询的时候若该点在点云中，则第一个近邻点是其本身
    // kdtree.setInputCloud(cloud);
    // int k =2;
    // float everagedistance =0;
    // for (int i =0; i < cloud->size()/2;i++)
    // {
    //         vector<int> nnh ;
    //         vector<float> squaredistance;
    //         //  pcl::PointXYZ p;
    //         //   p = cloud->points[i];
    //         kdtree.nearestKSearch(cloud->points[i],k,nnh,squaredistance);
    //         everagedistance += sqrt(squaredistance[1]);
    //         //   cout<<everagedistance<<endl;
    // }
    // everagedistance = everagedistance/(cloud->size()/2);
    // cout<<"everage distance is : "<<everagedistance<<endl;
    // pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normEst;  //其中pcl::PointXYZ表示输入类型数据，pcl::Normal表示输出类型,且pcl::Normal前三项是法向，最后一项是曲率
    // normEst.setInputCloud(cloud);
    // normEst.setSearchMethod(tree);
    // // normEst.setRadiusSearch(2);  //法向估计的半径
    // normEst.setKSearch(9);  //法向估计的点数
    // normEst.compute(*normals);
    // cout << "normal size is " << normals->size() << endl;
    // //normal_est.setViewPoint(0,0,0); //这个应该会使法向一致
    // est.setInputCloud(cloud);
    // est.setInputNormals(normals);
    // //  est.setAngleThreshold(90);
    // //   est.setSearchMethod (pcl::search::KdTree<pcl::PointXYZ>::Ptr (new pcl::search::KdTree<pcl::PointXYZ>));
    // est.setSearchMethod(tree);
    // est.setKSearch(50);  //一般这里的数值越高，最终边界识别的精度越好
    // //  est.setRadiusSearch(everagedistance);  //搜索半径
    // est.compute(boundaries);
    // //  pcl::PointCloud<pcl::PointXYZ> boundPoints;
    // pcl::PointCloud<pcl::PointXYZ>::Ptr boundPoints(new               pcl::PointCloud<pcl::PointXYZ>);
    // pcl::PointCloud<pcl::PointXYZ> noBoundPoints;
    // int countBoundaries = 0;
    // for (int i = 0; i < cloud->size(); i++) {
    //     uint8_t x = (boundaries.points[i].boundary_point);
    //     int a = static_cast<int>(x); //该函数的功能是强制类型转换
    //     if (a == 1)
    //     {
    //         //  boundPoints.push_back(cloud->points[i]);
    //         (*boundPoints).push_back(cloud->points[i]);
    //         countBoundaries++;
    //     }
    //     else
    //         noBoundPoints.push_back(cloud->points[i]);

    // }
    // std::cout << "boudary size is：" << countBoundaries << std::endl;
    
    //实车 观测范围滤波
    // pcl::PassThrough<pcl::PointXYZ> pass;
    // pass.setInputCloud(cloud_input_.makeShared());
    // pass.setFilterFieldName("x");
    // pass.setFilterLimits(-1.5,1.5);
    // pass.setFilterFieldName("y");
    // pass.setFilterLimits(-1.5,1.5);
    // pass.setFilterFieldName("z");
    // pass.setFilterLimits(-1.5,1.5);
    // pass.filter(cloud_input_);

    //点云转消息
    sensor_msgs::PointCloud2 img_tmp;
    // pcl::toROSMsg(*boundPoints,img_tmp);
    pcl::toROSMsg(cloud_input_,img_tmp);// to do: add other info, like tf, to this msg
    img_tmp.header=img.header;   
    std::cout << "ros time" << img_tmp.header.stamp << std::endl;
    std::cout << "ros now time" << ros::Time::now() << std::endl;
    point_pub_.publish(img_tmp);

}

void DownSamplePointCloud2::initFilter(ros::NodeHandle nh){
    point_sub_ = nh.subscribe<sensor_msgs::PointCloud2>("points", 0, &DownSamplePointCloud2::pointcloudCallback,this);
    point_pub_ = nh.advertise<sensor_msgs::PointCloud2>("sampled_points", 0,this);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "DownSampleNode");
//   ros::NodeHandle node;
  ros::NodeHandle nh("~");
  std::cout<<"Robot1 begin"<<std::endl;
  std::cout<<"Robot1 end"<<std::endl;
  DownSamplePointCloud2 DSCL;
  DSCL.initFilter(nh);
  ros::spin();
  return 0;
}