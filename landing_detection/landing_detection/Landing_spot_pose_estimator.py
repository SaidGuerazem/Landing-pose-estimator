import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np

class LandingPoseEstimator(Node):
    def __init__(self):
        super().__init__('landing_pose_estimator')
        self.bridge = CvBridge()

        self.camera_topics = ['/flir_camera/image_raw', '/cam_2/color/image_raw']

        # 🔧 Define per-camera intrinsics and transformations
        self.camera_params = {
            '/flir_camera/image_raw': {
                'camera_matrix': np.array([[2466.3471, 0.0, 1052.1786],
                                           [0.0, 2464.0090, 772.2511],
                                           [0.0, 0.0, 1.0]]),
                'dist_coeffs': np.array([-0.395923, 0.193811, -0.000792, -0.000657, 0.0]),
                'T_cam_drone': np.array([[0.0 , -1.0, 0.0 , 0.0],
                                       		[1.0, 0.0, 0.0, 0.0],
                                       		[0.0, 0.0, 1.0, 0.0],
				       		[0.0, 0.0, 0.0, 1.0]]),
            },
            '/cam_2/color/image_raw': {
                'camera_matrix': np.array([[413.3101, 0.0, 424.8177],
                                           [0.0, 413.8753, 248.7681],
                                           [0.0, 0.0, 1.0]]),
                'dist_coeffs': np.array([-0.045676, 0.033884, -0.001094, 0.000959, 0.0]),
                'T_cam_drone': np.array([
                    [0.0,  0.7071,  0.7071,  0.0],
                    [1.0,  0.0,     0.0,    -0.12037],
                    [0.0,  0.7071, -0.7071, -0.11435],
                    [0.0,  0.0,     0.0,     1.0]
                ])
            }
        }

        self.subscribers = [
            self.create_subscription(Image, topic, self.make_callback(topic), 10)
            for topic in self.camera_topics
        ]

        self.pub_red = self.create_publisher(PoseStamped, '/landing/red', 10)
        self.pub_green = self.create_publisher(PoseStamped, '/landing/green', 10)

        self.get_logger().info("LandingPoseEstimator node started.")

    def make_callback(self, topic_name):
        def callback(msg):
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            red_mask = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
            red_mask |= cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))

            green_mask = cv2.inRange(hsv, np.array([35, 100, 100]), np.array([85, 255, 255]))

            for color, mask in [('red', red_mask), ('green', green_mask)]:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if cv2.contourArea(cnt) > 1000:
                        ellipse = cv2.fitEllipse(cnt)
                        image_points = self.get_ellipse_image_points(ellipse)

                        radius = 0.915  # [m]
                        model_points = np.array([
                            [-radius, 0, 0],
                            [ radius, 0, 0],
                            [0, -radius, 0],
                            [0,  radius, 0]
                        ], dtype=np.float32)

                        cam_matrix = self.camera_params[topic_name]['camera_matrix']
                        dist_coeffs = self.camera_params[topic_name]['dist_coeffs']

                        success, rvec, tvec = cv2.solvePnP(
                            model_points, image_points, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
                        )

                        if success:
                            T_cam_marker = self.rvec_tvec_to_homogeneous(rvec, tvec)
                            T_cam_drone = self.camera_params[topic_name]['T_cam_drone']
                            T_drone_marker = T_cam_drone @ T_cam_marker

                            pose_msg = self.matrix_to_posestamped(T_drone_marker, msg.header.stamp)

                            if color == 'red':
                                self.pub_red.publish(pose_msg)
                            else:
                                self.pub_green.publish(pose_msg)

                            # cv2.ellipse(frame, ellipse, (0, 255, 255), 2)

            # cv2.imshow(f"Detection from {topic_name}", frame)
            # cv2.waitKey(1)
        return callback

    def get_ellipse_image_points(self, ellipse):
        (x, y), (MA, ma), angle = ellipse
        a, b = MA / 2, ma / 2
        rad = np.radians(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        return np.array([
            [x - a * cos_a, y - a * sin_a],  # Left
            [x + a * cos_a, y + a * sin_a],  # Right
            [x - b * sin_a, y + b * cos_a],  # Top
            [x + b * sin_a, y - b * cos_a]   # Bottom
        ], dtype=np.float32)

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
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
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
        return q  # (x, y, z, w)

def main(args=None):
    rclpy.init(args=args)
    node = LandingPoseEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

