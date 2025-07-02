# Landing_detection



## ArUco Centroid Detection Node

This ROS 2 node detects ArUco markers in camera images, calculates the normalized centroid of each marker, and publishes all detections to a unified topic `/aruco_detections`. It supports multiple cameras and formats detections for downstream pose estimation or guidance systems.

---

## Features

- Detects multiple ArUco markers per frame.
- Supports multiple camera topics.
- Publishes all detections to a **single topic** (`/aruco_detections`).
- Outputs are formatted as:
  ```
  {camera, id, cy, cx}
  ```
  where `camera` is one of:
  - `forward` → `/cam_2/color/image_raw`
  - `down` → `/flir_camera/image_raw`

---

## Package Name

```bash
landing_detection
```

---

## Dependencies



- ROS 2 (tested with Humble/Foxy)
- `cv_bridge`
- `OpenCV`
- `sensor_msgs`, `std_msgs`



## Run the Node


```bash
ros2 run landing_detection aruco_centroid_detection
```

---


---

## Published Topic

| Topic               | Message Type     | Description                            |
|--------------------|------------------|----------------------------------------|
| `/aruco_detections`| `std_msgs/String`| List of detections in format `{camera, id, cy, cx}` |

---

## Example Output

```
{forward, 23, 0.5214, 0.4823} {down, 10, 0.4000, 0.6000}
```

If no detections are found:

```
None
```

---

## Notes

- Marker centroids are normalized to `[0, 1]` range.
- Marker IDs are integers as defined by the ArUco dictionary (`DICT_ARUCO_ORIGINAL`).
- You can adapt the node to support more cameras or structured messages as needed.

---

## 📄 License

MIT License

