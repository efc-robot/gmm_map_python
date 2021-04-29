#include <iostream>
#include <vector>
#include <ctime>
#include <boost/thread/thread.hpp>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/features/eigen.h>
#include <pcl/features/feature.h>
#include <pcl/features/normal_3d.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/features/boundary.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
using namespace std;

int feature_callback()
{
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    // // if (pcl::io::loadPCDFile<pcl::PointXYZ>("/home/yxg/pcl/pcd/mid.pcd",*cloud) == -1)
    // if (pcl::io::loadPCDFile<pcl::PointXYZ>("C:\\Users\\fhlhc\\Desktop\\chairfilter_right.pcd", *cloud) == -1)
    // {
    //     PCL_ERROR("COULD NOT READ FILE mid.pcl \n");
    //     return (-1);
    // }

    std::cout << "points sieze is:" << cloud->size() << std::endl;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::Boundary> boundaries;
    pcl::BoundaryEstimation<pcl::PointXYZ, pcl::Normal, pcl::Boundary> est;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;  //创建一个快速k近邻查询,查询的时候若该点在点云中，则第一个近邻点是其本身
    kdtree.setInputCloud(cloud);
    int k =2;
    float everagedistance =0;
    for (int i =0; i < cloud->size()/2;i++)
    {
            vector<int> nnh ;
            vector<float> squaredistance;
            //  pcl::PointXYZ p;
            //   p = cloud->points[i];
            kdtree.nearestKSearch(cloud->points[i],k,nnh,squaredistance);
            everagedistance += sqrt(squaredistance[1]);
            //   cout<<everagedistance<<endl;
    }
    everagedistance = everagedistance/(cloud->size()/2);
    cout<<"everage distance is : "<<everagedistance<<endl;

    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normEst;  //其中pcl::PointXYZ表示输入类型数据，pcl::Normal表示输出类型,且pcl::Normal前三项是法向，最后一项是曲率
    normEst.setInputCloud(cloud);
    normEst.setSearchMethod(tree);
    // normEst.setRadiusSearch(2);  //法向估计的半径
    normEst.setKSearch(9);  //法向估计的点数
    normEst.compute(*normals);
    cout << "normal size is " << normals->size() << endl;

    //normal_est.setViewPoint(0,0,0); //这个应该会使法向一致
    est.setInputCloud(cloud);
    est.setInputNormals(normals);
    //  est.setAngleThreshold(90);
    //   est.setSearchMethod (pcl::search::KdTree<pcl::PointXYZ>::Ptr (new pcl::search::KdTree<pcl::PointXYZ>));
    est.setSearchMethod(tree);
    est.setKSearch(50);  //一般这里的数值越高，最终边界识别的精度越好
    //  est.setRadiusSearch(everagedistance);  //搜索半径
    est.compute(boundaries);

    //  pcl::PointCloud<pcl::PointXYZ> boundPoints;
    pcl::PointCloud<pcl::PointXYZ>::Ptr boundPoints(new               pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ> noBoundPoints;
    int countBoundaries = 0;
    for (int i = 0; i < cloud->size(); i++) {
        uint8_t x = (boundaries.points[i].boundary_point);
        int a = static_cast<int>(x); //该函数的功能是强制类型转换
        if (a == 1)
        {
            //  boundPoints.push_back(cloud->points[i]);
            (*boundPoints).push_back(cloud->points[i]);
            countBoundaries++;
        }
        else
            noBoundPoints.push_back(cloud->points[i]);

    }
    std::cout << "boudary size is：" << countBoundaries << std::endl;

    // pcl::io::savePLYFileASCII("C:\\Users\\fhlhc\\Desktop\\Boundpoints.ply", *boundPoints);



    //双视口
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("test Viewer"));
    viewer->initCameraParameters();
    int v1(0), v2(0);
    //原始点云窗口
    viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    viewer->setBackgroundColor(0, 0, 0, v1);
    viewer->addText("original", 10, 10, "v1 text", v1);
    viewer->addPointCloud<pcl::PointXYZ>(cloud, "sample cloud1", v1);
    viewer->addCoordinateSystem(1.0);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud1");
    //滤波窗口
    viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
    viewer->setBackgroundColor(0, 0, 0, v2);
    viewer->addText("提取边界", 10, 10, "v2 text", v2);
    viewer->addPointCloud<pcl::PointXYZ>(boundPoints, "sample cloud2", v2);
    viewer->addCoordinateSystem(1.0);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud2");
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);  //刷新
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
}



int main(int argc, char** argv)
{

    return 0;
}
