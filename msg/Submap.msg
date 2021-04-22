int32 robot_id
int32 submap_index # key frame 在某个机器人内部的 ID
geometry_msgs/Pose pose_odom
geometry_msgs/Pose pose # key frame的位置, (后端优化会不断优化这个位姿)
bool is_frozen #主要是判断这个地图是否不会再更新了
time add_time
gmm GMMmodel
float64[] point_clouds
float64[] descriptor