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

        self.camera_topics = ['/cam_1/color/image_raw', '/cam_2/color/image_raw', '/flir_camera/image_raw']
        self.subscribers = [
            self.create_subscription(Image, topic, self.make_callback(topic), 10)
            for topic in self.camera_topics
        ]

        self.aruco_publishers = {
            '/cam_1/color/image_raw': self.create_publisher(String, '/cam_1/aruco', 10),
            '/cam_2/color/image_raw': self.create_publisher(String, '/cam_2/aruco', 10),
            '/flir_camera/image_raw': self.create_publisher(String, '/flir_camera/aruco', 10)
        }

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_params = cv2.aruco.DetectorParameters()

        self.get_logger().info("Aruco centroid publisher node started.")

    def make_callback(self, topic_name):
        def callback(msg):
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            height, width = frame.shape[:2]

            corners, ids, _ = cv2.aruco.detectMarkers(
                frame, self.aruco_dict, parameters=self.aruco_params
            )

            if ids is not None:
                centroids = []
                for i, marker_id in enumerate(ids.flatten()):
                    marker_corners = corners[i][0]  # shape (4, 2)
                    cx = int(np.mean(marker_corners[:, 0]))
                    cy = int(np.mean(marker_corners[:, 1]))
                    norm_cx = cx / width
                    norm_cy = cy / height
                    centroids.append({"id": int(marker_id), "x": norm_cx, "y": norm_cy})

                msg_str = String()
                msg_str.data = str(centroids)
            else:
                msg_str = String()
                msg_str.data = "None"

            self.aruco_publishers[topic_name].publish(msg_str)
            self.get_logger().info(f"{topic_name} centroids: {msg_str.data}")
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

