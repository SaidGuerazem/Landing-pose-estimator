import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np

class ArucoCentroidDetection(Node):
    def __init__(self):
        super().__init__('aruco_centroid_detection')
        self.bridge = CvBridge()

        self.topic_camera_map = {
            '/cam_2/color/image_raw': 'forward',
            '/flir_camera/image_raw': 'down',
        }

        self.camera_topics = list(self.topic_camera_map.keys())
        self.subscribers = [
            self.create_subscription(Image, topic, self.make_callback(topic), 10)
            for topic in self.camera_topics
        ]

        self.aruco_publisher = self.create_publisher(String, '/aruco_detections', 10)

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_params = cv2.aruco.DetectorParameters()

        self.get_logger().info("Unified Aruco centroid publisher node started.")

    def make_callback(self, topic_name):
        def callback(msg):
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            height, width = frame.shape[:2]

            corners, ids, _ = cv2.aruco.detectMarkers(
                frame, self.aruco_dict, parameters=self.aruco_params
            )

            detection_strs = []
            camera_label = self.topic_camera_map.get(topic_name, None)
            if ids is not None and camera_label:
                for i, marker_id in enumerate(ids.flatten()):
                    marker_corners = corners[i][0]  # shape (4, 2)
                    cx = int(np.mean(marker_corners[:, 0]))
                    cy = int(np.mean(marker_corners[:, 1]))
                    norm_cx = cx / width
                    norm_cy = cy / height
                    detection_strs.append(f"{{{camera_label}, {marker_id}, {norm_cy:.4f}, {norm_cx:.4f}}}")

            if detection_strs:
                msg_str = String()
                msg_str.data = ' '.join(detection_strs)
            else:
                msg_str = String()
                msg_str.data = "None"

            self.aruco_publisher.publish(msg_str)
            self.get_logger().info(f"Detections from {camera_label or topic_name}: {msg_str.data}")
        return callback

def main(args=None):
    rclpy.init(args=args)
    node = ArucoCentroidDetection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

