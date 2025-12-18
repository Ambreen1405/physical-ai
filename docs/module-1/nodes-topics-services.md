---
id: nodes-topics-services
title: Nodes, Topics, Services
sidebar_position: 3
description: Comprehensive guide to ROS 2 communication patterns: nodes, topics, and services
keywords: [ros2 nodes, topics, services, communication patterns, robotics communication]
---

# Nodes, Topics, Services

This chapter covers the fundamental communication patterns in ROS 2: nodes, topics, and services. These patterns form the backbone of ROS 2's distributed system architecture.

## Learning Objectives

- Create and manage ROS 2 nodes effectively
- Implement publisher-subscriber communication using topics
- Design and implement request-response communication using services
- Apply appropriate communication patterns to different use cases
- Debug and monitor communication between nodes

## Nodes: The Foundation of ROS 2

### Node Definition

A node is the fundamental building block of a ROS 2 system. It's an executable process that performs specific robot functions and communicates with other nodes.

```python
# Example: Basic ROS 2 Node structure
import rclpy
from rclpy.node import Node

class SensorNode(Node):
    def __init__(self):
        # Initialize the node with a name
        super().__init__('sensor_node')

        # Node-specific initialization
        self.sensor_data = None
        self.get_logger().info('Sensor node initialized')
```

### Node Lifecycle

ROS 2 nodes have a well-defined lifecycle that includes initialization, execution, and cleanup phases:

```python
import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState
from rclpy.lifecycle import TransitionCallbackReturn

class LifecycleSensorNode(LifecycleNode):
    def __init__(self):
        super().__init__('lifecycle_sensor_node')

    def on_configure(self, state):
        self.get_logger().info('Configuring sensor node')
        # Setup sensors, initialize variables
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.get_logger().info('Activating sensor node')
        # Start sensor data collection
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self.get_logger().info('Deactivating sensor node')
        # Stop sensor data collection
        return TransitionCallbackReturn.SUCCESS
```

### Node Names and Namespacing

Nodes can be organized using namespaces for better system organization:

```python
# Node with namespace
class RobotArmNode(Node):
    def __init__(self):
        # Create node with namespace
        super().__init__('arm_controller', namespace='robot1')
        # This creates node name: /robot1/arm_controller
```

## Topics: Asynchronous Communication

### Publisher-Subscriber Pattern

Topics enable asynchronous, many-to-many communication using the publisher-subscriber pattern:

```
┌─────────────┐    ┌─────────────┐
│ Publisher 1 │    │ Publisher 2 │
│   (Data)    │    │   (Data)    │
└──────┬──────┘    └──────┬──────┘
       │                  │
       └────────┬─────────┘
                │
        ┌───────▼────────┐
        │   Topic/Topic  │
        │   (Messages)   │
        └───────┬────────┘
        ┌───────▼────────┐
        │   Topic/Topic  │
        │   (Messages)   │
        └───────┬────────┘
       │                  │
┌──────▼──────┐    ┌─────▼────────┐
│ Subscriber 1│    │ Subscriber 2 │
│ (Processes) │    │ (Processes)  │
└─────────────┘    └─────────────┘
```

### Creating Publishers

```python
from std_msgs.msg import String, Int32
import rclpy
from rclpy.node import Node

class DataPublisher(Node):
    def __init__(self):
        super().__init__('data_publisher')

        # Create publisher with topic name, message type, and queue size
        self.publisher_ = self.create_publisher(
            String,
            'sensor_data',
            10  # Queue size
        )

        # Timer to publish data periodically
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Sensor reading: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: {msg.data}')
        self.i += 1
```

### Creating Subscribers

```python
from std_msgs.msg import String
import rclpy
from rclpy.node import Node

class DataSubscriber(Node):
    def __init__(self):
        super().__init__('data_subscriber')

        # Create subscription to topic with callback function
        self.subscription = self.create_subscription(
            String,
            'sensor_data',
            self.listener_callback,
            10  # Queue size
        )
        # Don't forget to declare that we're not using the subscription
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: {msg.data}')
```

### Complex Message Types

ROS 2 supports complex message types beyond the standard ones:

```python
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header

class NavigationNode(Node):
    def __init__(self):
        super().__init__('navigation_node')

        # Publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )

        # Subscriber for laser scan data
        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            10
        )

    def scan_callback(self, msg):
        # Process laser scan data
        min_distance = min(msg.ranges)
        self.get_logger().info(f'Min obstacle distance: {min_distance}')
```

### Topic QoS Configuration

Quality of Service settings allow fine-tuning topic communication:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

class QoSPublisher(Node):
    def __init__(self):
        super().__init__('qos_publisher')

        # Configure QoS for real-time sensor data
        sensor_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        # Configure QoS for critical commands
        command_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.sensor_pub = self.create_publisher(
            String, 'sensor_data', sensor_qos
        )
        self.command_pub = self.create_publisher(
            Twist, 'cmd_vel', command_qos
        )
```

## Services: Synchronous Communication

### Request-Response Pattern

Services enable synchronous, one-to-one communication where a client sends a request and waits for a response:

```
┌─────────────┐     Request     ┌─────────────┐
│   Client    │ ──────────────→ │   Server    │
│             │                 │             │
│             │ ←────────────── │             │
└─────────────┘    Response     └─────────────┘
```

### Creating Services

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class AddTwoIntsService(Node):
    def __init__(self):
        super().__init__('add_two_ints_service')

        # Create service with service type, name, and callback
        self.srv = self.create_service(
            AddTwoInts,
            'add_two_ints',
            self.add_two_ints_callback
        )

    def add_two_ints_callback(self, request, response):
        # Process the request and set the response
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning {request.a} + {request.b} = {response.sum}')
        return response
```

### Creating Service Clients

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class AddTwoIntsClient(Node):
    def __init__(self):
        super().__init__('add_two_ints_client')

        # Create client for the service
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')

        # Wait for service to be available
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.request = AddTwoInts.Request()

    def send_request(self, a, b):
        self.request.a = a
        self.request.b = b
        # Asynchronously call the service
        self.future = self.cli.call_async(self.request)
        return self.future
```

### Service Implementation Example

```python
from std_srvs.srv import SetBool
import rclpy
from rclpy.node import Node

class MotorControlService(Node):
    def __init__(self):
        super().__init__('motor_control_service')

        # Service to enable/disable motors
        self.motor_enable_srv = self.create_service(
            SetBool,
            'motor_enable',
            self.motor_enable_callback
        )

        self.motors_enabled = False

    def motor_enable_callback(self, request, response):
        self.motors_enabled = request.data
        response.success = True
        response.message = f'Motors {"enabled" if self.motors_enabled else "disabled"}'
        return response
```

## Actions: Advanced Communication

While not the main focus of this section, actions provide goal-based communication:

```python
from rclpy.action import ActionServer
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')

        # Create action server
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback
        )

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1]
            )

            goal_handle.publish_feedback(feedback_msg)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        return result
```

## Communication Pattern Selection

### When to Use Each Pattern

| Pattern | Use Case | Characteristics |
|---------|----------|-----------------|
| Topics | Sensor data, continuous commands | Asynchronous, many-to-many, real-time |
| Services | Query-response, configuration | Synchronous, one-to-one, reliable |
| Actions | Long-running tasks with feedback | Goal-based, with progress feedback |

### Performance Considerations

```python
# Example: Choosing the right pattern for different scenarios

# For sensor data (high frequency, real-time)
sensor_pub = self.create_publisher(LaserScan, 'scan', 1)  # Topic

# For robot configuration (low frequency, reliable)
config_client = self.create_client(SetBool, 'config_service')  # Service

# For navigation (long-running with feedback)
nav_action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')  # Action
```

## Debugging and Monitoring

### Using ROS 2 Command Line Tools

```bash
# List all nodes
ros2 node list

# List all topics
ros2 topic list

# Echo messages on a topic
ros2 topic echo /sensor_data std_msgs/msg/String

# Call a service
ros2 service call /add_two_ints example_interfaces/srv/AddTwoInts "{a: 1, b: 2}"

# Show node graph
rqt_graph
```

### Programmatic Monitoring

```python
class MonitoringNode(Node):
    def __init__(self):
        super().__init__('monitoring_node')

        # Get list of active nodes
        node_names = self.get_node_names()
        self.get_logger().info(f'Active nodes: {node_names}')

        # Get topic names and types
        topic_names_and_types = self.get_topic_names_and_types()
        self.get_logger().info(f'Topics: {topic_names_and_types}')
```

## Best Practices

### Node Design
1. **Single Responsibility**: Each node should have one primary function
2. **Error Handling**: Implement proper error handling and recovery
3. **Resource Management**: Properly clean up resources in destruction

### Topic Design
1. **Message Efficiency**: Use appropriate message types and sizes
2. **Topic Naming**: Use descriptive, consistent naming conventions
3. **QoS Selection**: Choose appropriate QoS settings for each use case

### Service Design
1. **Response Time**: Services should respond in a reasonable time
2. **Idempotency**: Design services to be idempotent when possible
3. **Error Handling**: Return appropriate error codes in responses

## Next Steps

Continue to [Python with rclpy](./python-rclpy.md) to learn how to implement these communication patterns using Python.

## Learning Objectives Review

- Create and manage ROS 2 nodes effectively ✓
- Implement publisher-subscriber communication using topics ✓
- Design and implement request-response communication using services ✓
- Apply appropriate communication patterns to different use cases ✓
- Debug and monitor communication between nodes ✓

## Practical Exercise

Create a simple robot system with:
1. A sensor node that publishes random sensor data
2. A processing node that subscribes to sensor data and processes it
3. A service node that provides configuration commands
4. Use rqt_graph to visualize the communication

## Assessment Questions

1. What is the difference between topics and services in ROS 2?
2. Explain when you would use QoS settings in topic communication.
3. What are the advantages and disadvantages of the publisher-subscriber pattern?
4. How do you handle service calls asynchronously in ROS 2?

## Further Reading

- ROS 2 Topics and Services: https://docs.ros.org/en/humble/Tutorials/Topics/Understanding-ROS2-Topics.html
- ROS 2 Services: https://docs.ros.org/en/humble/Tutorials/Services/Understanding-ROS2-Services.html
- Quality of Service: https://docs.ros.org/en/humble/Concepts/About-Quality-of-Service-Settings.html