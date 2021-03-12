import rospy
import tf

rospy.init_node('tf_test', anonymous=True)
tf_listener = tf.TransformListener()
while True:
    tf_listener.waitForTransform("baselink_frame","frame_id",rospy.Time(0),rospy.Duration(0.11))
    print "1"