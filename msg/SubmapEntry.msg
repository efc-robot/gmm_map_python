#每个Submap本质上是一个key frame

int32 robot_id
int32 submap_index # key frame 在某个机器人内部的 ID
int32 submap_version # key frame 的版本,主要是在局部轨迹内更新 key frame
geometry_msgs/Pose pose # key frame的位置, (后端优化会不断优化这个位姿)
bool is_frozen #主要是判断这个地图是否不会再更新了