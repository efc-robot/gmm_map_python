int32 index         # key frame 在某个机器人内部的 ID
int32 robot_id
geometry_msgs/Pose pose_odom
geometry_msgs/Pose pose # key frame的位置, (后端优化会不断优化这个位姿)
time add_time
float64[] point_clouds
gmmFrame submap_gmm
bool freezed #主要是判断这个地图是否不会再更新了
float64[] descriptor
float64[] feature_point
gmmFrame feature_gmm