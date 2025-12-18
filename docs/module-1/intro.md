---
id: intro
title: Module 1 Overview - ROS 2 Fundamentals
sidebar_position: 1
description: Introduction to ROS 2 architecture, nodes, topics, and services
keywords: [ros2, robotics, nodes, topics, services, architecture]
---

# Module 1: ROS 2 Fundamentals

Welcome to Module 1 of the Physical AI & Humanoid Robotics textbook! This module provides a comprehensive introduction to ROS 2 (Robot Operating System 2), the foundational framework for modern robotics development.

## Learning Objectives

By the end of this module, you will be able to:
- Understand the architecture and design principles of ROS 2
- Create and manage ROS 2 nodes for robot functionality
- Implement communication patterns using topics and services
- Develop Python-based ROS 2 applications using rclpy
- Model robots using URDF (Unified Robot Description Format)
- Configure and launch complex robot systems

## Module Structure

This module consists of the following chapters:
1. [ROS 2 Architecture](./ros2-architecture.md) - Core concepts and design
2. [Nodes, Topics, Services](./nodes-topics-services.md) - Communication patterns
3. [Python with rclpy](./python-rclpy.md) - Python development
4. [URDF for Humanoids](./urdf-humanoids.md) - Robot description
5. [Launch Files](./launch-files.md) - System configuration
6. [Module 1 Assessment](./assessment.md) - Comprehensive evaluation

## Prerequisites

Before starting this module, you should have:
- Basic Python programming knowledge
- Understanding of object-oriented programming concepts
- Familiarity with command-line tools
- Basic understanding of robotics concepts (covered in introductory content)

## ROS 2 Overview

ROS 2 is a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.

### Key Features of ROS 2

1. **Distributed Computing**: ROS 2 enables communication between processes running on different machines
2. **Language Independence**: Support for multiple programming languages (C++, Python, etc.)
3. **Package Management**: Standardized way to organize and distribute code
4. **Simulation Integration**: Seamless integration with simulation environments
5. **Real-time Support**: Improved real-time capabilities compared to ROS 1
6. **Security**: Built-in security features for production deployment

### ROS 2 vs ROS 1

| Feature | ROS 1 | ROS 2 |
|---------|-------|-------|
| Communication | Custom TCP/UDP | DDS-based |
| Real-time | Limited | Enhanced |
| Multi-robot | Complex | Improved |
| Security | None | Built-in |
| Deployment | Research-focused | Production-ready |
| Architecture | Master-slave | Peer-to-peer |

## Core Concepts

### Nodes
A node is an executable that uses ROS 2 to communicate with other nodes. Nodes are organized into packages to be shared and used by other developers.

### Topics
Topics enable asynchronous message passing between nodes. Multiple nodes can publish to or subscribe from the same topic.

### Services
Services enable synchronous request/response communication between nodes.

### Actions
Actions provide a way to send goals to a server and receive feedback and results asynchronously.

## Development Environment

This module uses:
- **Operating System**: Ubuntu 22.04 LTS (recommended)
- **ROS Distribution**: ROS 2 Humble Hawksbill (LTS)
- **Programming Language**: Python 3.10+ with rclpy
- **Development Tools**: Visual Studio Code, colcon build system

## Getting Started

Each chapter in this module includes:
- Theoretical concepts with practical examples
- Python code examples using rclpy
- Step-by-step tutorials
- Exercises to reinforce learning
- Assessment questions

## Code Example: Simple ROS 2 Node

```python
from std_msgs.msg import String
import rclpy
from rclpy.node import Node

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
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

## Learning Objectives

By the end of this module, you will be able to:
- Understand the architecture and design principles of ROS 2
- Create and manage ROS 2 nodes for robot functionality
- Implement communication patterns using topics and services
- Develop Python-based ROS 2 applications using rclpy
- Model robots using URDF (Unified Robot Description Format)
- Configure and launch complex robot systems

## Practical Exercise

Set up your development environment with ROS 2 Humble Hawksbill and verify the installation by running the basic publisher/subscriber tutorial.

## Assessment Questions

1. What are the main differences between ROS 1 and ROS 2?
2. Explain the concept of a ROS 2 node and its role in the system.
3. What are the three main communication patterns in ROS 2?
4. Why is DDS important for ROS 2's architecture?

## Further Reading

- ROS 2 Documentation: https://docs.ros.org/en/humble/
- ROS 2 Tutorials: https://docs.ros.org/en/humble/Tutorials.html
- DDS Specification: https://www.omg.org/spec/DDS/

## Next Steps

Begin with [ROS 2 Architecture](./ros2-architecture.md) to understand the fundamental concepts that underpin all ROS 2 systems.

## Learning Objectives Review

- Understand the architecture and design principles of ROS 2 ✓
- Create and manage ROS 2 nodes for robot functionality ✓
- Implement communication patterns using topics and services ✓
- Develop Python-based ROS 2 applications using rclpy ✓
- Model robots using URDF (Unified Robot Description Format) ✓
- Configure and launch complex robot systems ✓

## Practical Exercise

Set up your development environment with ROS 2 Humble Hawksbill and verify the installation by running the basic publisher/subscriber tutorial.

## Assessment Questions

1. What are the main differences between ROS 1 and ROS 2?
2. Explain the concept of a ROS 2 node and its role in the system.
3. What are the three main communication patterns in ROS 2?
4. Why is DDS important for ROS 2's architecture?

## Further Reading

- ROS 2 Documentation: https://docs.ros.org/en/humble/
- ROS 2 Tutorials: https://docs.ros.org/en/humble/Tutorials.html
- DDS Specification: https://www.omg.org/spec/DDS/