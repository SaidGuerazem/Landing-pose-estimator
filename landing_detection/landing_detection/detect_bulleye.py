import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np

# Hardcoded intrinsics
fx, fy, cx, cy = 2466.3471145602066, 2464.0089937742105, 1052.1785889481048, 772.2511337898699
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
dist_coeffs = np.array([-0.3959231757313176, 0.1938111775664629, -0.0007922338732447631, -0.0006574750716075775, 0.0], dtype=np.float64)

T_cam_to_drone = np.array([
    [1, 0, 0, 0.1],
    [0, 1, 0, 0.0],
    [0, 0, 1, 0.0],
    [0, 0, 0, 1]
], dtype=np.float64)

class BullseyePoseEstimator(Node):
    def __init__(self):
        super().__init__('bullseye_pose_estimator')
        self.declare_parameter('image_topic', '/flir_camera/image_raw')
        self.declare_parameter('camera_frame', 'camera_frame')
        self.declare_parameter('target_diameter', 0.6096)

        self.image_topic = self.get_parameter('image_topic').value
        self.frame_id = self.get_parameter('camera_frame').value
        self.target_diameter = self.get_parameter('target_diameter').value

        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/bullseye_pose', 10)

        self.last_display_time = self.get_clock().now()
        self.display_interval = rclpy.duration.Duration(seconds=0.1)  # ~10 FPS

        cv2.startWindowThread()

    def image_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 2)
        gray = cv2.equalizeHist(gray)

        detected = False
        ix, iy, ir, ox, oy, orad = None, None, None, None, None, None

        inner_circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
            param1=100, param2=30, minRadius=40, maxRadius=250
        )

        if inner_circles is not None:
            ix, iy, ir = np.around(inner_circles[0][0]).astype(np.int32)
            cv2.circle(img, (ix, iy), ir, (0, 255, 0), 2)

            masked = gray.copy()
            cv2.circle(masked, (ix, iy), ir + 5, 255, -1)

            outer_circles = cv2.HoughCircles(
                masked, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                param1=100, param2=30, minRadius=ir+20, maxRadius=ir*4
            )

            if outer_circles is not None:
                outer_candidates = np.around(outer_circles[0]).astype(np.int32)
                ox, oy, orad = min(
                    outer_candidates, key=lambda c: np.hypot(c[0] - ix, c[1] - iy)
                )
                center_dist = np.hypot(ox - ix, oy - iy)
                radius_ratio = orad / ir

                if center_dist < 0.1 * orad and 1.2 < radius_ratio < 3.5:
                    detected = True
                    cv2.circle(img, (ox, oy), orad, (255, 0, 0), 2)

        if detected:
            R = self.target_diameter / 2
            model_points = np.array([
                [0.0, 0.0, 0.0],
                [R, 0.0, 0.0],
                [0.0, R, 0.0],
                [R * np.cos(np.pi/4), R * np.sin(np.pi/4), 0.0]
            ], dtype=np.float64)

            image_points = np.array([
                [ix, iy],
                [ix + R * 0.33 * fx, iy],
                [ix, iy - R * 0.33 * fy],
                [ix + R * 0.23 * fx, iy - R * 0.23 * fy]
            ], dtype=np.float64)

            success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

            if success:
                cam_pose = np.eye(4)
                R_mat, _ = cv2.Rodrigues(rvec)
                cam_pose[:3, :3] = R_mat
                cam_pose[:3, 3] = tvec.flatten()

                drone_pose = T_cam_to_drone @ cam_pose

                pose_msg = PoseStamped()
                pose_msg.header.stamp = msg.header.stamp
                pose_msg.header.frame_id = 'drone_base'
                pose_msg.pose.position.x = drone_pose[0, 3]
                pose_msg.pose.position.y = drone_pose[1, 3]
                pose_msg.pose.position.z = drone_pose[2, 3]
                q = self.rotation_matrix_to_quaternion(drone_pose[:3, :3])
                pose_msg.pose.orientation.x = q[0]
                pose_msg.pose.orientation.y = q[1]
                pose_msg.pose.orientation.z = q[2]
                pose_msg.pose.orientation.w = q[3]
                self.pose_pub.publish(pose_msg)

        cv2.putText(img, 'Bullseye Detected' if detected else 'Detection Failed',
                    (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if detected else (0,0,255), 2)

        now = self.get_clock().now()
        if now - self.last_display_time > self.display_interval:
            cv2.imshow('Bullseye Detection Debug', img)
            cv2.waitKey(1)
            self.last_display_time = now

    def rotation_matrix_to_quaternion(self, R):
        q = np.empty((4,))
        tr = np.trace(R)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            q[3] = 0.25 * S
            q[0] = (R[2, 1] - R[1, 2]) / S
            q[1] = (R[0, 2] - R[2, 0]) / S
            q[2] = (R[1, 0] - R[0, 1]) / S
        else:
            i = np.argmax(np.diagonal(R))
            if i == 0:
                S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                q[3] = (R[2, 1] - R[1, 2]) / S
                q[0] = 0.25 * S
                q[1] = (R[0, 1] + R[1, 0]) / S
                q[2] = (R[0, 2] + R[2, 0]) / S
            elif i == 1:
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
        return q[[0, 1, 2, 3]]

def main(args=None):
    rclpy.init(args=args)
    node = BullseyePoseEstimator()
    rclpy.spin(node)
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

