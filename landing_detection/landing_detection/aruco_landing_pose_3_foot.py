import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np

class ArucoPoseEstimator(Node):
    def __init__(self):
        super().__init__('aruco_pose_estimator')
        self.bridge = CvBridge()

        self.camera_topics = ['/cam_1/color/image_raw', '/cam_2/color/image_raw', '/flir_camera/image_raw']
        self.camera_to_drone_tf = {
            '/cam_1/color/image_raw': np.array([[-1.0 , 0.0, 0.0 , 0.0],
                                       		[0.0, 0.0, 1.0, 0.11435],
                                       		[0.0, 1.0, 0.0, 0.12037],
				       		[0.0, 0.0, 0.0, 1.0]]),  # Replace with actual 4x4 transformation matrices, this is the front facing camera
            '/cam_2/color/image_raw': np.array([[1.0 , 0.0, 0.0 , 0.0],
                                       		[0.0, 0.70710678118, -0.70710678118, 0.11435],
                                       		[0.0, 0.707106781180, 0.70710678118, 0.12037],
				       		[0.0, 0.0, 0.0, 1.0]]),  # Replace with actual 4x4 transformation matrices, this is the 45° tilted camera facing camera
            '/flir_camera/image_raw': np.array([[0.0 , 0.0, 1.0 , 0.0],
                                       	[-1.0, 0.0, 0.0, 0.0],
                                       	[0.0, -1.0, 0.0, 0.0],
				       	[0.0, 0.0, 0.0, 1.0]]),
        }

        self.subscribers = [
            self.create_subscription(Image, topic, self.make_callback(topic), 10)
            for topic in self.camera_topics
        ]

        self.pose_pub = self.create_publisher(PoseStamped, '/aruco/pose', 10)
        self.msg_pub = self.create_publisher(String, '/aruco/message', 10)

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL) # Change the dictionnary if you notice that the detection isn't happening
        self.aruco_params = cv2.aruco.DetectorParameters()

        self.camera_calibrations = {
            '/cam_1/color/image_raw': {
                'camera_matrix': np.array([[432.4352, 0.0, 429.1616],
                                        [0.0, 432.3150, 242.0728],
                                        [0.0, 0.0, 1.0]]),
                'dist_coeffs': np.array([-0.038960, 0.032735, 0.001603, 0.000783, 0.0])  # shape (5,)
            },
            '/cam_2/color/image_raw': {
                'camera_matrix': np.array([[413.3101, 0.0, 424.8177],
                                        [0.0, 413.8753, 248.7681],
                                        [0.0, 0.0, 1.0]]),
                'dist_coeffs': np.array([-0.045676, 0.033884, -0.001094, 0.000959, 0.0])
            },
            '/flir_camera/image_raw': {
                'camera_matrix': np.array([[2466.3471145602066, 0.0, 1052.1785889481048],
                                        [0.0, 2464.0089937742105, 772.2511337898699],
                                        [0.0, 0.0, 1.0]]),
                'dist_coeffs': np.array([-0.3959231757313176, 0.1938111775664629, -0.0007922338732447631, -0.0006574750716075775, 0.0])
            }
        }

        self.marker_length = 0.9144  # [m] known side length of ArUco markers

        self.get_logger().info("ArucoPoseEstimator node started.")

    def make_callback(self, topic_name):
        def callback(msg):
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            calib = self.camera_calibrations[topic_name]
            camera_matrix = calib['camera_matrix']
            dist_coeffs = calib['dist_coeffs']

            corners, ids, _ = cv2.aruco.detectMarkers(
                frame, self.aruco_dict, parameters=self.aruco_params
            )

            if ids is not None:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_length, camera_matrix, dist_coeffs
                )

                for i, marker_id in enumerate(ids.flatten()):
                    rvec = rvecs[i]
                    tvec = tvecs[i]

                    T_cam_marker = self.rvec_tvec_to_homogeneous(rvec, tvec)
                    T_cam_drone = self.camera_to_drone_tf[topic_name]
                    T_drone_marker = np.matmul(T_cam_drone, T_cam_marker)

                    pose_msg = self.matrix_to_posestamped(T_drone_marker, msg.header.stamp)
                    self.pose_pub.publish(pose_msg)

                    msg_str = String()
                    msg_str.data = f"Marker ID: {marker_id}"
                    self.msg_pub.publish(msg_str)

                    cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs,
                                    rvec, tvec, self.marker_length * 0.5)

            cv2.imshow(f"View from {topic_name}", frame)
            cv2.waitKey(1)
        return callback

    def rvec_tvec_to_homogeneous(self, rvec, tvec):
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        return T

    def matrix_to_posestamped(self, T, stamp):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = 'drone_base'

        pose_msg.pose.position.x = float(T[0, 3])
        pose_msg.pose.position.y = float(T[1, 3])
        pose_msg.pose.position.z = float(T[2, 3])

        quat = self.rotation_matrix_to_quaternion(T[:3, :3])
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]

        return pose_msg

    def rotation_matrix_to_quaternion(self, R):
        q = np.empty((4,))
        tr = np.trace(R)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            q[3] = 0.25 * S
            q[0] = (R[2, 1] - R[1, 2]) / S
            q[1] = (R[0, 2] - R[2, 0]) / S
            q[2] = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q[3] = (R[2, 1] - R[1, 2]) / S
            q[0] = 0.25 * S
            q[1] = (R[0, 1] + R[1, 0]) / S
            q[2] = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            q[3] = (R[0, 2] - R[2, 0]) / S
            q[0] = (R[0, 1] + R[1, 0]) / S
            q[1] = 0.25 * S
            q[2] = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            q[3] = (R[1, 0] - R[0, 1]) / S
            q[0] = (R[0, 2] + R[2, 0]) / S
            q[1] = (R[1, 2] + R[2, 1]) / S
            q[2] = 0.25 * S
        return q  # [x, y, z, w]

def main(args=None):
    rclpy.init(args=args)
    node = ArucoPoseEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
