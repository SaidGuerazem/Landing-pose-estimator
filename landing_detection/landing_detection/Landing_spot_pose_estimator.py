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
        self.subscription = self.create_subscription(
            Image,
            '/cam/color/image_raw',
            self.image_callback,
            10)

        self.pub_red = self.create_publisher(PoseStamped, '/landing/red', 10)
        self.pub_green = self.create_publisher(PoseStamped, '/landing/green', 10)

        # 🔧 Camera Intrinsics (you must replace these with your calibrated values)
        self.camera_matrix = np.array([
            [600.0, 0.0, 320.0],
            [0.0, 600.0, 240.0],
            [0.0, 0.0, 1.0]
        ])
        self.dist_coeffs = np.zeros((5, 1))  # Replace with actual distortion if available

        self.get_logger().info("LandingPoseEstimator node started.")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Color thresholds
        red_mask = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
        red_mask |= cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))

        green_mask = cv2.inRange(hsv, np.array([35, 100, 100]), np.array([85, 255, 255]))

        for color, mask in [('red', red_mask), ('green', green_mask)]:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 1000:
                    ellipse = cv2.fitEllipse(cnt)

                    # Get image points
                    image_points = self.get_ellipse_image_points(ellipse)

                    # Define 3D model points (circle on ground plane)
                    radius = 0.915  # meters
                    model_points = np.array([
                        [-radius, 0, 0],       # Left
                        [radius, 0, 0],        # Right
                        [0, -radius, 0],       # Top
                        [0, radius, 0]         # Bottom
                    ], dtype=np.float32)

                    success, rvec, tvec = cv2.solvePnP(model_points, image_points,
                                                       self.camera_matrix,
                                                       self.dist_coeffs,
                                                       flags=cv2.SOLVEPNP_ITERATIVE)

                    if success:
                        pose_msg = PoseStamped()
                        pose_msg.header.stamp = msg.header.stamp
                        pose_msg.header.frame_id = 'camera_link'

                        # Rotation vector to quaternion conversion
                        R, _ = cv2.Rodrigues(rvec)
                        quaternion = self.rotation_matrix_to_quaternion(R)

                        pose_msg.pose.position.x = float(tvec[0])
                        pose_msg.pose.position.y = float(tvec[1])
                        pose_msg.pose.position.z = float(tvec[2])
                        pose_msg.pose.orientation.x = quaternion[0]
                        pose_msg.pose.orientation.y = quaternion[1]
                        pose_msg.pose.orientation.z = quaternion[2]
                        pose_msg.pose.orientation.w = quaternion[3]

                        if color == 'red':
                            self.pub_red.publish(pose_msg)
                        else:
                            self.pub_green.publish(pose_msg)

                        cv2.ellipse(frame, ellipse, (0, 255, 255), 2)

        cv2.imshow("Landing Pose Estimation", frame)
        cv2.waitKey(1)

    def get_ellipse_image_points(self, ellipse):
        (x, y), (MA, ma), angle = ellipse
        # Approximated image points (left, right, top, bottom of ellipse)
        a = MA / 2
        b = ma / 2
        rad = np.radians(angle)

        cos_a, sin_a = np.cos(rad), np.sin(rad)
        pts = np.array([
            [x - a * cos_a, y - a * sin_a],  # Left
            [x + a * cos_a, y + a * sin_a],  # Right
            [x - b * sin_a, y + b * cos_a],  # Top
            [x + b * sin_a, y - b * cos_a]   # Bottom
        ], dtype=np.float32)
        return pts

    def rotation_matrix_to_quaternion(self, R):
        # Convert 3x3 rotation matrix to quaternion (x, y, z, w)
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
