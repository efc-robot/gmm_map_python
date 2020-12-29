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

class DownSamplePointCloud2{
public:
    void initFilter(ros::NodeHandle nh);
    void pointcloudCallback(const sensor_msgs::PointCloud2 img);
private:
    ros::Subscriber point_sub_;
    ros::Publisher point_pub_;
    pcl::VoxelGrid<pcl::PointXYZ> sor_;
    pcl::PointCloud<pcl::PointXYZ> cloud_input_;

};

void DownSamplePointCloud2::pointcloudCallback(sensor_msgs::PointCloud2 img){
    pcl::fromROSMsg(img, cloud_input_);
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(cloud_input_,cloud_input_, indices);
    sor_.setInputCloud(cloud_input_.makeShared());
    sor_.setLeafSize(0.1f, 0.1f, 0.1f);
    sor_.filter(cloud_input_);
    sensor_msgs::PointCloud2 img_tmp;
    pcl::toROSMsg(cloud_input_,img_tmp);// to do: add other info, like tf, to this msg
    img_tmp.header=img.header;
    point_pub_.publish(img_tmp);

}

void DownSamplePointCloud2::initFilter(ros::NodeHandle nh){
    point_sub_ = nh.subscribe<sensor_msgs::PointCloud2>("points", 1, &DownSamplePointCloud2::pointcloudCallback,this);
    point_pub_ = nh.advertise<sensor_msgs::PointCloud2>("sampled_points",1,this);
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