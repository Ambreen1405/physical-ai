---
id: python-rclpy
title: Python with rclpy
sidebar_position: 4
description: Comprehensive guide to developing ROS 2 nodes using Python and rclpy
keywords: [rclpy, python, ros2 python, robotics programming, python nodes]
---

# Python with rclpy

This chapter provides a comprehensive guide to developing ROS 2 nodes using Python and the rclpy client library. Python is an excellent choice for rapid prototyping and development in robotics.

## Learning Objectives

- Master the rclpy client library for ROS 2 Python development
- Create complex nodes with multiple publishers, subscribers, and services
- Implement advanced features like parameters, timers, and callbacks
- Debug and optimize Python-based ROS 2 nodes
- Structure Python packages for ROS 2 projects

## Introduction to rclpy

### What is rclpy?

rclpy is the Python client library for ROS 2. It provides a Python API to ROS 2 concepts such as nodes, publishers, subscribers, services, and parameters. It's built on top of the C client library (rcl) and provides Pythonic interfaces to ROS 2 functionality.

### Installation and Setup

```bash
# Install ROS 2 Python packages
sudo apt update
sudo apt install python3-ros-foxy-*
# For Humble Hawksbill, replace 'foxy' with 'humble'
```

### Basic Node Structure

```python
import rclpy
from rclpy.node import Node

def main(args=None):
    # Initialize the ROS client library
    rclpy.init(args=args)

    # Create an instance of your node class
    my_node = MyNodeClass()

    # Start spinning to process callbacks
    rclpy.spin(my_node)

    # Destroy the node explicitly (optional)
    my_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Creating Nodes with rclpy

### Basic Node Implementation

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        # Initialize the Node class with a name
        super().__init__('minimal_publisher')

        # Create a publisher
        self.publisher_ = self.create_publisher(String, 'topic', 10)

        # Create a timer
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Node with Multiple Communication Patterns

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import SetBool
from rclpy.qos import QoSProfile

class ComplexNode(Node):
    def __init__(self):
        super().__init__('complex_node')

        # Publisher
        qos_profile = QoSProfile(depth=10)
        self.publisher_ = self.create_publisher(String, 'status', qos_profile)

        # Subscriber
        self.subscription = self.create_subscription(
            String,
            'commands',
            self.command_callback,
            qos_profile
        )

        # Service
        self.srv = self.create_service(SetBool, 'enable', self.enable_callback)

        # Timer
        self.timer = self.create_timer(1.0, self.status_timer_callback)

        self.enabled = True
        self.status_counter = 0

    def command_callback(self, msg):
        self.get_logger().info(f'Received command: {msg.data}')
        if msg.data == 'enable':
            self.enabled = True
        elif msg.data == 'disable':
            self.enabled = False

    def enable_callback(self, request, response):
        self.enabled = request.data
        response.success = True
        response.message = f'Node {"enabled" if self.enabled else "disabled"}'
        return response

    def status_timer_callback(self):
        if self.enabled:
            msg = String()
            msg.data = f'Node operational: {self.status_counter}'
            self.publisher_.publish(msg)
            self.status_counter += 1

def main(args=None):
    rclpy.init(args=args)
    node = ComplexNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Publishers and Subscribers

### Advanced Publisher Features

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

class AdvancedPublisher(Node):
    def __init__(self):
        super().__init__('advanced_publisher')

        # Different QoS profiles for different use cases
        # High-frequency sensor data
        sensor_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        # Critical commands
        command_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # Create publishers with different QoS
        self.sensor_pub = self.create_publisher(String, 'sensor_data', sensor_qos)
        self.command_pub = self.create_publisher(String, 'commands', command_qos)

        # Publisher with custom callback group
        self.status_pub = self.create_publisher(String, 'status', 10)

        # Timer for publishing
        self.timer = self.create_timer(0.1, self.publish_callback)

    def publish_callback(self):
        # Publish sensor data
        sensor_msg = String()
        sensor_msg.data = f'Sensor reading: {self.get_clock().now().nanoseconds}'
        self.sensor_pub.publish(sensor_msg)

        # Publish status
        status_msg = String()
        status_msg.data = f'Node running at {self.get_clock().now().nanoseconds}'
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    publisher = AdvancedPublisher()
    rclpy.spin(publisher)
    publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Advanced Subscriber Features

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.qos import QoSProfile, ReliabilityPolicy

class AdvancedSubscriber(Node):
    def __init__(self):
        super().__init__('advanced_subscriber')

        # Multiple subscriptions with different QoS
        qos_sensor = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        qos_command = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        self.sensor_sub = self.create_subscription(
            String,
            'sensor_data',
            self.sensor_callback,
            qos_sensor
        )

        self.command_sub = self.create_subscription(
            String,
            'commands',
            self.command_callback,
            qos_command
        )

        # Subscription with custom callback group
        self.status_sub = self.create_subscription(
            String,
            'status',
            self.status_callback,
            10
        )

        self.sensor_count = 0
        self.command_count = 0

    def sensor_callback(self, msg):
        self.sensor_count += 1
        self.get_logger().info(f'Sensor message #{self.sensor_count}: {msg.data}')

    def command_callback(self, msg):
        self.command_count += 1
        self.get_logger().info(f'Command message #{self.command_count}: {msg.data}')

    def status_callback(self, msg):
        self.get_logger().info(f'Status: {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    subscriber = AdvancedSubscriber()
    rclpy.spin(subscriber)
    subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Services and Clients

### Service Implementation

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts
from std_srvs.srv import SetBool

class CalculatorService(Node):
    def __init__(self):
        super().__init__('calculator_service')

        # Multiple services
        self.add_srv = self.create_service(
            AddTwoInts,
            'add_two_ints',
            self.add_callback
        )

        self.multiply_srv = self.create_service(
            AddTwoInts,
            'multiply_two_ints',
            self.multiply_callback
        )

        # State service
        self.state_srv = self.create_service(
            SetBool,
            'set_operation_state',
            self.set_state_callback
        )

        self.operations_enabled = True

    def add_callback(self, request, response):
        if self.operations_enabled:
            response.sum = request.a + request.b
            self.get_logger().info(f'{request.a} + {request.b} = {response.sum}')
        else:
            response.sum = 0
            self.get_logger().warn('Operations disabled')
        return response

    def multiply_callback(self, request, response):
        if self.operations_enabled:
            response.sum = request.a * request.b
            self.get_logger().info(f'{request.a} * {request.b} = {response.sum}')
        else:
            response.sum = 0
            self.get_logger().warn('Operations disabled')
        return response

    def set_state_callback(self, request, response):
        self.operations_enabled = request.data
        response.success = True
        response.message = f'Operations {"enabled" if self.operations_enabled else "disabled"}'
        return response

def main(args=None):
    rclpy.init(args=args)
    service = CalculatorService()
    rclpy.spin(service)
    service.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Service Client Implementation

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts
from std_srvs.srv import SetBool

class CalculatorClient(Node):
    def __init__(self):
        super().__init__('calculator_client')

        # Create clients
        self.add_client = self.create_client(AddTwoInts, 'add_two_ints')
        self.multiply_client = self.create_client(AddTwoInts, 'multiply_two_ints')
        self.state_client = self.create_client(SetBool, 'set_operation_state')

        # Wait for services
        while not self.add_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Add service not available, waiting again...')

        while not self.multiply_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Multiply service not available, waiting again...')

        while not self.state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('State service not available, waiting again...')

        # Async calls
        self.send_requests()

    def send_requests(self):
        # Send add request
        add_request = AddTwoInts.Request()
        add_request.a = 2
        add_request.b = 3

        self.add_future = self.add_client.call_async(add_request)
        self.add_future.add_done_callback(self.add_response_callback)

        # Send multiply request
        multiply_request = AddTwoInts.Request()
        multiply_request.a = 4
        multiply_request.b = 5

        self.multiply_future = self.multiply_client.call_async(multiply_request)
        self.multiply_future.add_done_callback(self.multiply_response_callback)

    def add_response_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(f'Add result: {response.sum}')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

    def multiply_response_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(f'Multiply result: {response.sum}')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

def main(args=None):
    rclpy.init(args=args)
    client = CalculatorClient()
    rclpy.spin(client)
    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Parameters

### Parameter Declaration and Usage

```python
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values and descriptions
        self.declare_parameter(
            'robot_name',
            'default_robot',
            ParameterDescriptor(description='Name of the robot')
        )

        self.declare_parameter(
            'max_velocity',
            1.0,
            ParameterDescriptor(description='Maximum velocity in m/s')
        )

        self.declare_parameter(
            'safety_distance',
            0.5,
            ParameterDescriptor(description='Safety distance in meters')
        )

        # Declare parameter with constraints
        self.declare_parameter(
            'control_frequency',
            50,
            ParameterDescriptor(description='Control loop frequency in Hz')
        )

        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.safety_distance = self.get_parameter('safety_distance').value
        self.control_frequency = self.get_parameter('control_frequency').value

        self.get_logger().info(
            f'Robot: {self.robot_name}, Max vel: {self.max_velocity}, '
            f'Safety dist: {self.safety_distance}, Freq: {self.control_frequency}'
        )

        # Set up parameter callback
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'max_velocity' and param.type_ == param.Type.DOUBLE:
                if param.value > 5.0:
                    self.get_logger().warn('Max velocity too high!')
                    return rclpy.node.SetParametersResult(successful=False)

        return rclpy.node.SetParametersResult(successful=True)

def main(args=None):
    rclpy.init(args=args)
    node = ParameterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Timers and Callbacks

### Advanced Timer Usage

```python
import rclpy
from rclpy.node import Node
from rclpy.timer import Rate

class TimerNode(Node):
    def __init__(self):
        super().__init__('timer_node')

        # Multiple timers with different periods
        self.timer_100ms = self.create_timer(0.1, self.timer_100ms_callback)
        self.timer_500ms = self.create_timer(0.5, self.timer_500ms_callback)
        self.timer_1s = self.create_timer(1.0, self.timer_1s_callback)

        # Timer with rate limiting
        self.counter = 0
        self.rate_timer = self.create_rate(10)  # 10 Hz

    def timer_100ms_callback(self):
        self.get_logger().info('100ms timer callback')

    def timer_500ms_callback(self):
        self.get_logger().info('500ms timer callback')

    def timer_1s_callback(self):
        self.get_logger().info('1s timer callback')

def main(args=None):
    rclpy.init(args=args)
    node = TimerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Exception Handling and Error Management

### Robust Error Handling

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import traceback

class RobustNode(Node):
    def __init__(self):
        super().__init__('robust_node')

        self.publisher_ = self.create_publisher(String, 'robust_topic', 10)
        self.timer = self.create_timer(0.1, self.robust_timer_callback)
        self.error_count = 0

    def robust_timer_callback(self):
        try:
            # Risky operation
            msg = String()
            msg.data = f'Robust message: {100 / (10 - self.error_count)}'
            self.publisher_.publish(msg)
            self.error_count = 0  # Reset on success

        except ZeroDivisionError:
            self.error_count += 1
            self.get_logger().error('Division by zero error occurred')

        except Exception as e:
            self.error_count += 1
            self.get_logger().error(f'Unexpected error: {str(e)}')
            self.get_logger().error(traceback.format_exc())

        finally:
            # Cleanup operations
            pass

def main(args=None):
    rclpy.init(args=args)
    node = RobustNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Package Structure and Organization

### Creating a ROS 2 Python Package

```
my_robot_package/
├── CMakeLists.txt
├── package.xml
├── setup.py
├── setup.cfg
├── resource/
│   └── my_robot_package
├── test/
│   └── test_copyright.py
│   └── test_flake8.py
│   └── test_pep257.py
│   └── test_my_robot_package.py
└── my_robot_package/
    ├── __init__.py
    ├── publisher_member_function.py
    ├── subscriber_member_function.py
    └── service_member_function.py
```

### setup.py for Python Package

```python
from setuptools import setup

package_name = 'my_robot_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='A Python package for my robot',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = my_robot_package.publisher_member_function:main',
            'listener = my_robot_package.subscriber_member_function:main',
        ],
    },
)
```

## Debugging Techniques

### Debugging with Logging

```python
import rclpy
from rclpy.node import Node
from rclpy.logging import LoggingSeverity

class DebugNode(Node):
    def __init__(self):
        super().__init__('debug_node')

        # Set logging level
        self.get_logger().set_level(LoggingSeverity.DEBUG)

        # Different logging levels
        self.get_logger().debug('Debug message')
        self.get_logger().info('Info message')
        self.get_logger().warn('Warning message')
        self.get_logger().error('Error message')
        self.get_logger().fatal('Fatal message')

        # Log structured data
        self.debug_var = 42
        self.get_logger().info(f'Debug variable value: {self.debug_var}')

def main(args=None):
    rclpy.init(args=args)
    node = DebugNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Performance Optimization

### Efficient Message Handling

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from collections import deque
import time

class PerformanceNode(Node):
    def __init__(self):
        super().__init__('performance_node')

        # Efficient message handling
        self.publisher_ = self.create_publisher(String, 'perf_topic', 10)
        self.subscription = self.create_subscription(
            String, 'perf_topic', self.message_callback, 10
        )

        # Message queue for batch processing
        self.message_queue = deque(maxlen=100)

        # Performance monitoring
        self.message_count = 0
        self.start_time = time.time()

        self.timer = self.create_timer(1.0, self.performance_callback)

    def message_callback(self, msg):
        # Add to queue for batch processing
        self.message_queue.append(msg.data)
        self.message_count += 1

        # Process in batches for efficiency
        if len(self.message_queue) >= 10:
            self.process_batch()

    def process_batch(self):
        # Process multiple messages at once
        while self.message_queue:
            msg_data = self.message_queue.popleft()
            # Process message efficiently
            processed = msg_data.upper()
            self.get_logger().debug(f'Processed: {processed}')

    def performance_callback(self):
        elapsed = time.time() - self.start_time
        rate = self.message_count / elapsed if elapsed > 0 else 0
        self.get_logger().info(f'Message rate: {rate:.2f} Hz')

def main(args=None):
    rclpy.init(args=args)
    node = PerformanceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices

### Code Organization

1. **Use Type Hints**: Improve code readability and IDE support
2. **Follow PEP 8**: Maintain Python code style consistency
3. **Document Functions**: Use docstrings for all public functions
4. **Handle Exceptions**: Implement proper error handling
5. **Resource Management**: Clean up resources properly

### Performance Tips

1. **Minimize Message Copies**: Use efficient message structures
2. **Choose Appropriate QoS**: Match QoS settings to use case
3. **Batch Processing**: Process messages in batches when possible
4. **Efficient Data Structures**: Use appropriate Python data structures

## Next Steps

Continue to [URDF for Humanoids](./urdf-humanoids.md) to learn about robot description using URDF.

## Learning Objectives Review

- Master the rclpy client library for ROS 2 Python development ✓
- Create complex nodes with multiple publishers, subscribers, and services ✓
- Implement advanced features like parameters, timers, and callbacks ✓
- Debug and optimize Python-based ROS 2 nodes ✓
- Structure Python packages for ROS 2 projects ✓

## Practical Exercise

Create a complete robot controller node that:
1. Declares and uses parameters for robot configuration
2. Publishes sensor data using appropriate QoS settings
3. Subscribes to command topics and processes them
4. Provides services for robot control
5. Implements proper error handling and logging

## Assessment Questions

1. What is the difference between rclpy and rcl?
2. How do you declare and use parameters in rclpy?
3. Explain the difference between synchronous and asynchronous service calls.
4. What are the best practices for organizing ROS 2 Python packages?

## Further Reading

- rclpy Documentation: https://docs.ros.org/en/humble/p/rclpy/
- ROS 2 Python Tutorials: https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-Your-First-ROS2-Package.html
- Python Style Guide: https://docs.ros.org/en/humble/The-ROS2-Project/Contributing/Code-Style-Language-Versions.html