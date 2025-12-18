---
id: sensor-simulation
title: Sensor Simulation
sidebar_position: 6
description: Comprehensive guide to simulating various sensors in robotics, including LiDAR, cameras, and IMUs
keywords: [sensor simulation, lidar, cameras, imu, robotics sensors, sensor fusion, perception]
---

# Sensor Simulation

Sensor simulation is a critical component of robotics development, enabling safe testing of perception algorithms without physical hardware. This chapter covers the simulation of various sensors including LiDAR, cameras, and IMUs.

## Learning Objectives

- Understand the principles of sensor simulation in robotics
- Implement LiDAR sensor simulation with realistic noise models
- Create camera sensor simulation with proper distortion models
- Simulate IMU and other inertial sensors
- Apply sensor fusion techniques in simulation
- Validate sensor data quality and accuracy

## Introduction to Sensor Simulation

### Why Simulate Sensors?

Sensor simulation provides several key benefits:
- **Safety**: Test perception algorithms without physical risk
- **Cost Efficiency**: Reduce hardware costs and wear
- **Repeatability**: Exact same conditions for consistent testing
- **Scalability**: Run multiple experiments in parallel
- **Accessibility**: Develop algorithms without physical sensors
- **Edge Cases**: Simulate rare or dangerous scenarios safely

### Sensor Simulation Challenges

- **Realism**: Making simulated data resemble real sensor data
- **Computational Cost**: Balancing accuracy with performance
- **Noise Modeling**: Accurately simulating sensor imperfections
- **Environmental Effects**: Modeling lighting, weather, etc.
- **Latency**: Simulating real-world sensor timing

## LiDAR Simulation

### LiDAR Fundamentals

LiDAR (Light Detection and Ranging) sensors emit laser pulses and measure the time it takes for reflections to return, creating 3D point cloud data.

### Basic LiDAR Simulation in Gazebo

```xml
<!-- LiDAR sensor configuration -->
<sensor name="lidar" type="ray">
  <pose>0.2 0 0.1 0 0 0</pose> <!-- Position relative to parent link -->
  <ray>
    <scan>
      <horizontal>
        <samples>720</samples> <!-- Number of beams per revolution -->
        <resolution>1</resolution>
        <min_angle>-3.14159</min_angle> <!-- -180 degrees -->
        <max_angle>3.14159</max_angle>  <!-- 180 degrees -->
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>    <!-- Minimum detection range -->
      <max>30.0</max>   <!-- Maximum detection range -->
      <resolution>0.01</resolution> <!-- Range resolution -->
    </range>
  </ray>
  <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
    <ros>
      <namespace>/my_robot</namespace>
      <remapping>~/out:=scan</remapping>
    </ros>
    <output_type>sensor_msgs/LaserScan</output_type>
    <frame_name>lidar_frame</frame_name>
  </plugin>
</sensor>
```

### LiDAR Noise Modeling

Real LiDAR sensors have various sources of noise and error:

```xml
<sensor name="lidar" type="ray">
  <ray>
    <!-- ... previous configuration ... -->
  </ray>
  <noise type="gaussian">
    <mean>0.0</mean>
    <stddev>0.01</stddev> <!-- 1cm standard deviation -->
  </noise>
</sensor>
```

### Advanced LiDAR Configuration

```xml
<sensor name="3d_lidar" type="ray">
  <pose>0.3 0 0.5 0 0 0</pose>
  <ray>
    <scan>
      <horizontal>
        <samples>1024</samples>
        <resolution>1</resolution>
        <min_angle>-3.14159</min_angle>
        <max_angle>3.14159</max_angle>
      </horizontal>
      <vertical>
        <samples>64</samples>
        <resolution>1</resolution>
        <min_angle>-0.5236</min_angle> <!-- -30 degrees -->
        <max_angle>0.3491</max_angle>   <!-- 20 degrees -->
      </vertical>
    </scan>
    <range>
      <min>0.1</min>
      <max>100.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <plugin name="velodyne_vlp16" filename="libgazebo_ros_velodyne_gpu_laser.so">
    <ros>
      <namespace>/my_robot</namespace>
      <remapping>~/out:=velodyne_points</remapping>
    </ros>
    <topic_name>velodyne_points</topic_name>
    <frame_name>velodyne</frame_name>
    <min_range>0.9</min_range>
    <max_range>130.0</max_range>
    <gaussian_noise>0.008</gaussian_noise>
  </plugin>
</sensor>
```

### LiDAR Processing in ROS

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import Header
import numpy as np

class LidarProcessor(Node):
    def __init__(self):
        super().__init__('lidar_processor')

        # Subscribe to LiDAR data
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/my_robot/scan',
            self.scan_callback,
            10
        )

        # Publish processed data
        self.cloud_pub = self.create_publisher(
            PointCloud2,
            '/my_robot/processed_cloud',
            10
        )

        self.get_logger().info('LiDAR processor initialized')

    def scan_callback(self, msg):
        """Process incoming LiDAR scan data"""
        try:
            # Convert scan to points
            points = []
            angle = msg.angle_min

            for range_val in msg.ranges:
                if msg.range_min <= range_val <= msg.range_max:
                    x = range_val * np.cos(angle)
                    y = range_val * np.sin(angle)
                    z = 0.0  # Assuming 2D scan
                    points.append([x, y, z])

                angle += msg.angle_increment

            # Process points (e.g., filtering, clustering)
            processed_points = self.process_points(points)

            # Publish as PointCloud2
            cloud_msg = self.create_pointcloud2(processed_points)
            self.cloud_pub.publish(cloud_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing scan: {e}')

    def process_points(self, points):
        """Apply processing to LiDAR points"""
        # Remove points that are too close (noise filtering)
        filtered_points = [p for p in points if np.linalg.norm(p) > 0.2]

        # Add basic clustering logic here
        # For now, just return filtered points
        return filtered_points

    def create_pointcloud2(self, points):
        """Create PointCloud2 message from points list"""
        # This is a simplified version - in practice, use sensor_msgs_py
        cloud_msg = PointCloud2()
        cloud_msg.header = Header()
        cloud_msg.header.stamp = self.get_clock().now().to_msg()
        cloud_msg.header.frame_id = 'lidar_frame'
        # Implementation would continue with proper PointCloud2 construction
        return cloud_msg

def main(args=None):
    rclpy.init(args=args)
    processor = LidarProcessor()
    rclpy.spin(processor)
    processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Camera Simulation

### Camera Fundamentals

Camera sensors simulate visual perception by capturing images of the environment. Key parameters include:
- **Resolution**: Image dimensions
- **Field of View**: Angular extent of the scene
- **Distortion**: Lens imperfections
- **Frame Rate**: Images per second

### Basic Camera Configuration

```xml
<sensor name="camera" type="camera">
  <pose>0.1 0 0.2 0 0 0</pose> <!-- Position relative to parent link -->
  <camera name="head_camera">
    <horizontal_fov>1.047</horizontal_fov> <!-- 60 degrees in radians -->
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near> <!-- Near clipping plane -->
      <far>100</far>   <!-- Far clipping plane -->
    </clip>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.007</stddev>
    </noise>
  </camera>
  <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
    <ros>
      <namespace>/my_robot</namespace>
      <remapping>image_raw:=image_raw</remapping>
      <remapping>camera_info:=camera_info</remapping>
    </ros>
    <camera_name>camera</camera_name>
    <frame_name>camera_frame</frame_name>
    <hack_baseline>0.07</hack_baseline>
    <distortion_k1>0.0</distortion_k1>
    <distortion_k2>0.0</distortion_k2>
    <distortion_k3>0.0</distortion_k3>
    <distortion_t1>0.0</distortion_t1>
    <distortion_t2>0.0</distortion_t2>
  </plugin>
</sensor>
```

### Advanced Camera with Distortion

```xml
<sensor name="stereo_camera" type="multicamera">
  <pose>0.1 0 0.2 0 0 0</pose>
  <camera name="left_camera">
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>1280</width>
      <height>720</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>30</far>
    </clip>
    <distortion>
      <k1>-0.17187</k1>
      <k2>0.03843</k2>
      <k3>-0.00076</k3>
      <p1>0.00031</p1>
      <p2>-0.00014</p2>
    </distortion>
  </camera>
  <camera name="right_camera">
    <pose>0.2 0 0 0 0 0</pose> <!-- 20cm baseline -->
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>1280</width>
      <height>720</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>30</far>
    </clip>
    <distortion>
      <k1>-0.17234</k1>
      <k2>0.04012</k2>
      <k3>-0.00089</k3>
      <p1>0.00028</p1>
      <p2>-0.00017</p2>
    </distortion>
  </camera>
  <plugin name="stereo_camera_controller" filename="libgazebo_ros_multicamera.so">
    <ros>
      <namespace>/my_robot</namespace>
    </ros>
    <camera_name>stereo</camera_name>
    <image_topic_name>image_raw</image_topic_name>
    <camera_info_topic_name>camera_info</camera_info_topic_name>
    <frame_name>stereo_camera_frame</frame_name>
    <hack_baseline>0.2</hack_baseline>
  </plugin>
</sensor>
```

### Camera Processing in ROS

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np

class CameraProcessor(Node):
    def __init__(self):
        super().__init__('camera_processor')

        self.bridge = CvBridge()

        # Subscribe to camera data
        self.image_sub = self.create_subscription(
            Image,
            '/my_robot/camera/image_raw',
            self.image_callback,
            10
        )

        # Subscribe to camera info for calibration
        self.info_sub = self.create_subscription(
            CameraInfo,
            '/my_robot/camera/camera_info',
            self.info_callback,
            10
        )

        # Publish processed image
        self.processed_pub = self.create_publisher(
            Image,
            '/my_robot/camera/processed_image',
            10
        )

        self.camera_matrix = None
        self.distortion_coeffs = None

        self.get_logger().info('Camera processor initialized')

    def info_callback(self, msg):
        """Receive camera calibration info"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.d)
        self.get_logger().info('Camera calibration received')

    def image_callback(self, msg):
        """Process incoming camera image"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Apply processing (undistortion, feature detection, etc.)
            if self.camera_matrix is not None and self.distortion_coeffs is not None:
                # Undistort image
                h, w = cv_image.shape[:2]
                new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                    self.camera_matrix,
                    self.distortion_coeffs,
                    (w, h),
                    1,
                    (w, h)
                )
                cv_image = cv2.undistort(
                    cv_image,
                    self.camera_matrix,
                    self.distortion_coeffs,
                    None,
                    new_camera_matrix
                )

            # Apply computer vision processing
            processed_image = self.process_cv(cv_image)

            # Convert back to ROS Image
            processed_msg = self.bridge.cv2_to_imgmsg(processed_image, encoding='bgr8')
            processed_msg.header = msg.header

            # Publish processed image
            self.processed_pub.publish(processed_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def process_cv(self, image):
        """Apply computer vision processing"""
        # Example: Edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Convert back to color for visualization
        result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return result

def main(args=None):
    rclpy.init(args=args)
    processor = CameraProcessor()
    rclpy.spin(processor)
    processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## IMU Simulation

### IMU Fundamentals

An IMU (Inertial Measurement Unit) typically combines:
- **Accelerometer**: Measures linear acceleration
- **Gyroscope**: Measures angular velocity
- **Magnetometer**: Measures magnetic field (compass)

### Basic IMU Configuration

```xml
<sensor name="imu" type="imu">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <pose>0 0 0.1 0 0 0</pose> <!-- Position in robot -->
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.001</stddev> <!-- 1 mrad/s -->
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.001</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.001</stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.017</stddev> <!-- 1.7 mg -->
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.017</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.017</stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
  <plugin name="imu_controller" filename="libgazebo_ros_imu.so">
    <ros>
      <namespace>/my_robot</namespace>
      <remapping>~/out:=imu/data</remapping>
    </ros>
    <frame_name>imu_link</frame_name>
    <topic_name>imu/data</topic_name>
    <update_rate>100</update_rate>
    <gaussian_noise>0.01</gaussian_noise>
  </plugin>
</sensor>
```

### IMU Processing in ROS

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3
import numpy as np
from scipy.spatial.transform import Rotation as R

class IMUProcessor(Node):
    def __init__(self):
        super().__init__('imu_processor')

        # Subscribe to IMU data
        self.imu_sub = self.create_subscription(
            Imu,
            '/my_robot/imu/data',
            self.imu_callback,
            10
        )

        # Publish processed data
        self.orientation_pub = self.create_publisher(
            Imu,
            '/my_robot/imu/filtered',
            10
        )

        # Initialize orientation filter
        self.orientation = R.from_quat([0, 0, 0, 1])  # Identity rotation
        self.last_time = None

        self.get_logger().info('IMU processor initialized')

    def imu_callback(self, msg):
        """Process incoming IMU data"""
        try:
            # Extract angular velocity
            angular_vel = np.array([
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            ])

            # Extract linear acceleration
            linear_acc = np.array([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ])

            # Get current time
            current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            if self.last_time is not None:
                dt = current_time - self.last_time

                # Integrate angular velocity to get orientation change
                # Simple integration (in practice, use more sophisticated methods)
                delta_angle = angular_vel * dt
                delta_rotation = R.from_rotvec(delta_angle)

                # Update orientation
                self.orientation = self.orientation * delta_rotation

            self.last_time = current_time

            # Create filtered IMU message
            filtered_msg = Imu()
            filtered_msg.header = msg.header
            quat = self.orientation.as_quat()
            filtered_msg.orientation.x = quat[0]
            filtered_msg.orientation.y = quat[1]
            filtered_msg.orientation.z = quat[2]
            filtered_msg.orientation.w = quat[3]

            # Copy angular velocity and linear acceleration
            filtered_msg.angular_velocity = msg.angular_velocity
            filtered_msg.linear_acceleration = msg.linear_acceleration

            # Publish filtered data
            self.orientation_pub.publish(filtered_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing IMU: {e}')

def main(args=None):
    rclpy.init(args=args)
    processor = IMUProcessor()
    rclpy.spin(processor)
    processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Other Sensor Types

### GPS Simulation

```xml
<sensor name="gps" type="gps">
  <always_on>true</always_on>
  <update_rate>1</update_rate>
  <pose>0.5 0 0.5 0 0 0</pose>
  <plugin name="gps_controller" filename="libgazebo_ros_gps.so">
    <ros>
      <namespace>/my_robot</namespace>
      <remapping>~/out:=gps/fix</remapping>
    </ros>
    <frame_name>gps_link</frame_name>
    <topic_name>fix</topic_name>
    <update_rate>1.0</update_rate>
    <gaussian_noise>0.01</gaussian_noise>
    <offset>0 0 0</offset>
  </plugin>
</sensor>
```

### Force/Torque Sensor

```xml
<sensor name="force_torque" type="force_torque">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <pose>0 0 0 0 0 0</pose>
  <plugin name="ft_sensor" filename="libgazebo_ros_ft_sensor.so">
    <ros>
      <namespace>/my_robot</namespace>
      <remapping>~/out:=wrench</remapping>
    </ros>
    <frame_name>ft_sensor_link</frame_name>
    <topic_name>wrench</topic_name>
    <update_rate>100</update_rate>
  </plugin>
</sensor>
```

## Sensor Fusion

### Combining Multiple Sensors

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped
import numpy as np

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion')

        # Subscribe to multiple sensors
        self.imu_sub = self.create_subscription(
            Imu,
            '/my_robot/imu/data',
            self.imu_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/my_robot/scan',
            self.scan_callback,
            10
        )

        # Publish fused pose estimate
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/my_robot/pose_fused',
            10
        )

        # Initialize state
        self.orientation = None
        self.scan_data = None
        self.last_update = None

        self.get_logger().info('Sensor fusion node initialized')

    def imu_callback(self, msg):
        """Handle IMU data"""
        self.orientation = [
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ]

    def scan_callback(self, msg):
        """Handle LiDAR data"""
        self.scan_data = msg.ranges

    def fuse_sensors(self):
        """Combine sensor data for improved estimate"""
        if self.orientation is None or self.scan_data is None:
            return None

        # Simple fusion approach
        # In practice, use Kalman filters, particle filters, or other methods

        # Extract orientation information
        orientation = np.array(self.orientation)

        # Process LiDAR data for position estimation
        # This is a simplified example
        valid_ranges = [r for r in self.scan_data if 0.1 < r < 10.0]
        if valid_ranges:
            # Estimate position based on nearby obstacles
            avg_range = np.mean(valid_ranges)
            # This is a placeholder - real fusion would be more complex
            position_estimate = [avg_range, 0, 0]  # Simplified
        else:
            position_estimate = [0, 0, 0]

        return position_estimate, orientation

    def publish_fused_pose(self):
        """Publish fused pose estimate"""
        result = self.fuse_sensors()
        if result is not None:
            pos, orient = result

            pose_msg = PoseWithCovarianceStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = 'map'

            pose_msg.pose.pose.position.x = pos[0]
            pose_msg.pose.pose.position.y = pos[1]
            pose_msg.pose.pose.position.z = pos[2]

            pose_msg.pose.pose.orientation.x = orient[0]
            pose_msg.pose.pose.orientation.y = orient[1]
            pose_msg.pose.pose.orientation.z = orient[2]
            pose_msg.pose.pose.orientation.w = orient[3]

            # Set covariance (simplified)
            pose_msg.pose.covariance = [0.1] * 36  # Placeholder

            self.pose_pub.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    fusion_node = SensorFusionNode()

    # Run fusion periodically
    timer = fusion_node.create_timer(0.1, fusion_node.publish_fused_pose)

    rclpy.spin(fusion_node)
    fusion_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Sensor Validation and Testing

### Sensor Data Quality Assessment

```python
import numpy as np
from scipy import stats

def validate_lidar_data(ranges, intensity=None):
    """Validate LiDAR data quality"""
    # Check for NaN or inf values
    if np.any(np.isnan(ranges)) or np.any(np.isinf(ranges)):
        return False, "NaN or inf values detected"

    # Check range bounds
    valid_mask = (ranges >= 0.1) & (ranges <= 30.0)  # Typical LiDAR range
    if np.sum(valid_mask) / len(ranges) < 0.1:  # Less than 10% valid
        return False, "Too many out-of-range measurements"

    # Check for consistency (optional)
    range_diffs = np.diff(ranges[valid_mask])
    if len(range_diffs) > 0 and np.std(range_diffs) > 5.0:
        # Potentially too much variation
        pass

    return True, "Data appears valid"

def validate_camera_data(image_data):
    """Validate camera data quality"""
    # Check image dimensions and format
    if len(image_data.shape) not in [2, 3]:
        return False, "Invalid image dimensions"

    # Check for uniform or near-uniform images (potential sensor failure)
    if len(image_data.shape) == 3:  # Color image
        gray = np.mean(image_data, axis=2)
    else:  # Grayscale
        gray = image_data

    # Check variance - too low variance might indicate sensor issues
    variance = np.var(gray)
    if variance < 10:  # Threshold may need adjustment
        return False, f"Low image variance ({variance}), possible sensor issue"

    return True, "Image appears valid"

def validate_imu_data(accel, gyro, mag=None):
    """Validate IMU data quality"""
    # Check for NaN or inf values
    if (np.any(np.isnan(accel)) or np.any(np.isinf(accel)) or
        np.any(np.isnan(gyro)) or np.any(np.isinf(gyro))):
        return False, "NaN or inf values detected"

    # Check acceleration magnitude (should be ~9.81 m/s² when static)
    accel_mag = np.linalg.norm(accel)
    if abs(accel_mag - 9.81) > 2.0:  # Allow for motion
        # This might be okay if robot is moving
        pass

    # Check gyroscope magnitude (should be low when static)
    gyro_mag = np.linalg.norm(gyro)
    if gyro_mag > 1.0 and abs(accel_mag - 9.81) < 0.5:
        # Robot is static but gyroscope shows motion
        return False, "Gyroscope shows motion while accelerometer suggests static state"

    return True, "IMU data appears valid"
```

## Performance Optimization

### Efficient Sensor Simulation

```xml
<!-- Optimize sensor update rates based on application needs -->
<sensor name="low_freq_camera" type="camera">
  <update_rate>5</update_rate> <!-- Lower rate for distant object detection -->
  <!-- ... configuration ... -->
</sensor>

<sensor name="high_freq_imu" type="imu">
  <update_rate>200</update_rate> <!-- Higher rate for control applications -->
  <!-- ... configuration ... -->
</sensor>

<!-- Use appropriate resolutions -->
<sensor name="navigation_camera" type="camera">
  <image>
    <width>320</width>  <!-- Lower resolution for navigation -->
    <height>240</height>
    <!-- ... -->
  </image>
</sensor>

<sensor name="detection_camera" type="camera">
  <image>
    <width>1280</width> <!-- Higher resolution for object detection -->
    <height>960</height>
    <!-- ... -->
  </image>
</sensor>
```

### Sensor Data Compression

For high-bandwidth sensors like cameras:

```python
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage

def compress_image(image, quality=85):
    """Compress image for transmission"""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', image, encode_param)
    if result:
        return encimg.tobytes()
    return None

def decompress_image(compressed_data):
    """Decompress image data"""
    nparr = np.frombuffer(compressed_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image
```

## Learning Objectives Review

- Understand the principles of sensor simulation in robotics ✓
- Implement LiDAR sensor simulation with realistic noise models ✓
- Create camera sensor simulation with proper distortion models ✓
- Simulate IMU and other inertial sensors ✓
- Apply sensor fusion techniques in simulation ✓
- Validate sensor data quality and accuracy ✓

## Practical Exercise

1. Set up a Gazebo world with multiple sensor types (LiDAR, camera, IMU)
2. Configure realistic noise models for each sensor
3. Create ROS nodes to process each sensor's data
4. Implement a basic sensor fusion algorithm combining LiDAR and IMU data
5. Validate the sensor data quality and accuracy

## Assessment Questions

1. Explain the importance of noise modeling in sensor simulation.
2. How do you configure camera distortion parameters in Gazebo?
3. What are the key differences between processing LiDAR and camera data?
4. Describe the process of sensor fusion and its benefits.

## Further Reading

- Robot Operating System (ROS) Sensor Messages: http://docs.ros.org/en/api/sensor_msgs/html/index-msg.html
- Gazebo Sensor Tutorial: http://gazebosim.org/tutorials?tut=ros_gzplugins
- "Probabilistic Robotics" by Sebastian Thrun, Wolfram Burgard, and Dieter Fox
- OpenCV Documentation for Computer Vision: https://docs.opencv.org/

## Next Steps

Continue to [Module 2 Assessment](./assessment.md) to test your knowledge of simulation environments.