---
id: ros2-architecture
title: ROS 2 Architecture
sidebar_position: 2
description: Deep dive into ROS 2 architecture, design principles, and core concepts
keywords: [ros2 architecture, dds, middleware, ros2 concepts, robotics framework]
---

# ROS 2 Architecture

ROS 2 represents a complete redesign of the Robot Operating System with a focus on production deployment, real-time performance, and security. Understanding its architecture is fundamental to effective robotics development.

## Learning Objectives

- Explain the DDS-based architecture of ROS 2
- Understand the role of middleware in ROS 2
- Identify the key components of the ROS 2 framework
- Compare ROS 2 architecture with ROS 1
- Implement basic architectural patterns

## DDS Foundation

### Data Distribution Service (DDS)

ROS 2's architecture is built on DDS (Data Distribution Service), an OMG (Object Management Group) standard for real-time, distributed data exchange. DDS provides:

- **Data-Centricity**: Focus on data rather than communication endpoints
- **Quality of Service (QoS)**: Configurable policies for reliability, durability, etc.
- **Discovery**: Automatic peer discovery in the network
- **Language and Platform Independence**: Standardized interfaces

### DDS vs. ROS 1 Communication

In ROS 1, communication was based on a custom TCP/UDP implementation with a central master node. ROS 2's DDS-based approach provides:

```
ROS 1 Architecture:
┌─────────────┐    ┌─────────────┐
│   Master    │    │   Master    │
│ (Central)   │    │ (Central)   │
└─────┬───────┘    └─────┬───────┘
      │                  │
┌─────▼──────┐    ┌─────▼──────┐
│   Node A   │    │   Node B   │
│            │    │            │
└────────────┘    └────────────┘

ROS 2 Architecture:
┌─────────────────────────────────┐
│           DDS Network           │
│  ┌─────────┐    ┌─────────┐    │
│  │  Node A │    │  Node B │    │
│  │         │    │         │    │
│  └─────────┘    └─────────┘    │
└─────────────────────────────────┘
```

## Core Architecture Components

### 1. Client Libraries

ROS 2 provides client libraries that implement the ROS 2 API:
- **rclcpp**: C++ client library
- **rclpy**: Python client library
- **rcl**: Reference implementation in C
- **rclc**: C library optimized for microcontrollers

```python
# Example: Using rclpy to create a node
import rclpy
from rclpy.node import Node

class ArchitectureNode(Node):
    def __init__(self):
        super().__init__('architecture_node')
        self.get_logger().info('Architecture node initialized')
```

### 2. RMW (ROS Middleware) Layer

The ROS Middleware Abstraction layer provides a common interface to different DDS implementations:

```python
# The middleware selection happens at runtime
# Supported implementations: Fast DDS, Cyclone DDS, RTI Connext DDS
```

### 3. Message and Service Definitions

ROS 2 uses `.msg`, `.srv`, and `.action` files to define message types:

```python
# Example message definition (geometry_msgs/msg/Twist.msg)
# Linear and angular velocities
float64[3] linear
float64[3] angular
```

## Quality of Service (QoS) Profiles

QoS profiles allow fine-tuning communication behavior:

### Reliability Policy
- **Reliable**: All messages are delivered (like TCP)
- **Best Effort**: Messages may be lost (like UDP)

### Durability Policy
- **Transient Local**: Late-joining subscribers receive last known values
- **Volatile**: No historical data provided to late joiners

### History Policy
- **Keep Last**: Store only the most recent N messages
- **Keep All**: Store all messages (limited by memory)

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Example: Configuring QoS for sensor data
sensor_qos = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST
)

# Example: Configuring QoS for critical commands
command_qos = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST
)
```

## Node Architecture

### Node Lifecycle

ROS 2 nodes have a well-defined lifecycle with multiple states:

```
┌─────────────┐
│   Unconfigured  │
└──────┬──────┘
       │ create()
       ▼
┌─────────────┐
│  Inactive   │ ←─────────────────┐
└──────┬──────┘                   │
       │ activate()               │
       ▼                          │
┌─────────────┐    ─ ─ ─ ─ ─ ─ ─ ─│─
│   Active    │ ←→ │  Executing  ││
└──────┬──────┘    ─ ─ ─ ─ ─ ─ ─ ─│─
       │ deactivate()             │
       │                          │
       └──────────────────────────┘
```

### Node Composition

ROS 2 supports node composition to improve performance:

```python
# Example: Composing nodes within the same process
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

class ComposedNode(Node):
    def __init__(self):
        super().__init__('composed_node')
        # Multiple functional components in one node
        self.sensor_processor = SensorProcessor()
        self.motion_controller = MotionController()
        self.behavior_manager = BehaviorManager()
```

## Parameter System

ROS 2 includes a robust parameter system:

```python
# Example: Parameter declaration and usage
class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values and descriptions
        self.declare_parameter('robot_name', 'default_robot')
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('safety_distance', 0.5)

        # Access parameters
        self.robot_name = self.get_parameter('robot_name').value
        self.max_velocity = self.get_parameter('max_velocity').value
```

## Launch System Architecture

ROS 2's launch system provides sophisticated process management:

```python
# Example launch file structure (Python-based)
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_package',
            executable='my_node',
            name='my_node_name',
            parameters=[
                {'param1': 'value1'},
                {'param2': 42}
            ]
        )
    ])
```

## Security Architecture

ROS 2 includes security features based on DDS Security specification:

### Authentication
- Identity verification of nodes
- Certificate-based authentication

### Encryption
- Data encryption in transit
- Secure communication channels

### Access Control
- Permission-based access to topics/services
- Role-based security policies

## Performance Considerations

### Memory Management
- Zero-copy transport for high-performance applications
- Memory-pooled message allocation

### Real-time Support
- Support for real-time operating systems
- Deterministic message delivery

### Network Optimization
- Multicast support for efficient data distribution
- Bandwidth optimization through QoS configuration

## Best Practices

### Design Patterns
1. **Single Responsibility**: Each node should have one primary function
2. **Loose Coupling**: Minimize dependencies between nodes
3. **High Cohesion**: Group related functionality within nodes
4. **Interface Segregation**: Use specific interfaces rather than general ones

### Performance Optimization
- Use appropriate QoS settings for each communication
- Implement node composition for high-frequency interactions
- Consider message size and frequency trade-offs

## Next Steps

Continue to [Nodes, Topics, Services](./nodes-topics-services.md) to explore the communication patterns that make up the ROS 2 ecosystem.

## Learning Objectives Review

- Explain the DDS-based architecture of ROS 2 ✓
- Understand the role of middleware in ROS 2 ✓
- Identify the key components of the ROS 2 framework ✓
- Compare ROS 2 architecture with ROS 1 ✓
- Implement basic architectural patterns ✓

## Practical Exercise

Create a simple ROS 2 node that demonstrates the basic architecture concepts:
1. Create a node with proper logging
2. Implement parameter declaration and usage
3. Add a timer callback to demonstrate execution
4. Run the node and verify it appears in the ROS graph

## Assessment Questions

1. What is DDS and why is it important for ROS 2 architecture?
2. Explain the difference between Reliable and Best Effort QoS policies.
3. What are the advantages of node composition over separate nodes?
4. How does the ROS 2 parameter system differ from ROS 1?

## Further Reading

- ROS 2 Design Documents: https://design.ros2.org/
- DDS Specification: https://www.omg.org/spec/DDS/
- ROS 2 QoS Implementation: https://github.com/ros2/design/blob/master/articles/qos_reliability_history_durability.md