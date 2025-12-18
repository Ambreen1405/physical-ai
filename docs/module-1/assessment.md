---
id: assessment
title: Module 1 Assessment
sidebar_position: 7
description: Comprehensive assessment for ROS 2 fundamentals module
keywords: [ros2 assessment, robotics quiz, nodes topics services, urdf launch]
---

# Module 1 Assessment: ROS 2 Fundamentals

This assessment evaluates your understanding of ROS 2 fundamentals, including architecture, nodes, topics, services, Python development with rclpy, URDF, and launch files.

## Assessment Objectives

By completing this assessment, you will demonstrate:
- Understanding of ROS 2 architecture and DDS foundation
- Proficiency in creating and managing ROS 2 nodes
- Ability to implement communication patterns using topics and services
- Skills in Python-based ROS 2 development
- Knowledge of URDF for robot modeling
- Competency in creating launch files for robot systems

## Part 1: ROS 2 Architecture (20 points)

### Question 1.1 (5 points)
Explain the difference between ROS 1 and ROS 2 communication architectures. What is DDS and why is it important for ROS 2?

### Question 1.2 (5 points)
List and explain the three main Quality of Service (QoS) policies in ROS 2. Provide an example of when you would use each policy.

### Question 1.3 (5 points)
Describe the node lifecycle in ROS 2. What are the different states a node can be in, and what transitions are possible?

### Question 1.4 (5 points)
What are the main advantages of using node composition in ROS 2? When would you choose to compose nodes versus running them separately?

## Part 2: Nodes, Topics, and Services (25 points)

### Question 2.1 (8 points)
Write Python code for a ROS 2 publisher that publishes a String message containing the current timestamp to a topic called "current_time" at 1 Hz frequency. Include proper imports and main function.

### Question 2.2 (8 points)
Write Python code for a ROS 2 service server that implements an "add_two_ints" service. The service should take two integers as input and return their sum. Include proper imports and main function.

### Question 2.3 (9 points)
Explain the differences between topics, services, and actions in ROS 2. For each communication pattern, provide:
- The communication model (synchronous/asynchronous, one-to-one/many-to-many)
- A typical use case
- The advantages and disadvantages

## Part 3: Python with rclpy (20 points)

### Question 3.1 (10 points)
Create a ROS 2 node in Python that has:
- A parameter called "robot_name" with a default value of "default_robot"
- A publisher for String messages on the "status" topic
- A subscriber to String messages on the "commands" topic
- A timer that publishes the robot name and current time every 2 seconds

Include proper imports and main function.

### Question 3.2 (10 points)
Explain the following rclpy concepts and provide a code example for each:
- Launch configuration and parameter declaration
- Different types of callbacks (timer, subscription, service)
- Error handling in ROS 2 nodes

## Part 4: URDF for Humanoids (20 points)

### Question 4.1 (8 points)
Create a simple URDF for a robot with:
- A base link (box shape, 0.5x0.5x0.2)
- A head link (sphere, radius 0.1) connected to base with a fixed joint
- Proper inertial, visual, and collision properties for both links
- A material definition for the head

### Question 4.2 (7 points)
Explain the purpose of Xacro and provide an example of a Xacro macro that creates a generic arm segment. The macro should accept parameters for length, radius, and mass.

### Question 4.3 (5 points)
What are the key considerations when defining inertial properties for humanoid robot links? Why are they important for simulation?

## Part 5: Launch Files (15 points)

### Question 5.1 (8 points)
Write a launch file that:
- Launches a robot_state_publisher node with a URDF file
- Launches a controller node with specific parameters
- Uses a launch argument for the robot name
- Conditionally launches a debug node based on another argument

### Question 5.2 (7 points)
Explain the following launch file concepts and provide examples:
- Launch substitutions (at least 2 different types)
- Grouping and namespacing in launch files
- Including other launch files

## Practical Exercise (30 points)

### Exercise 5.1 (30 points)
Create a complete ROS 2 package with the following components:

1. **Python Nodes** (10 points):
   - A sensor node that publishes random sensor data
   - A processing node that subscribes to sensor data and processes it
   - A service node that provides configuration commands

2. **URDF Model** (10 points):
   - Create a simple robot URDF with at least 3 links and 2 joints
   - Include proper visual, collision, and inertial properties
   - Use Xacro to parameterize at least one aspect of the model

3. **Launch File** (10 points):
   - Create a launch file that starts all nodes
   - Include parameters for configuring the system
   - Use proper namespacing and remappings

Provide the complete code for each component and explain how they work together.

## Answer Key and Scoring

### Part 1: ROS 2 Architecture (20 points)
- Question 1.1: 5 points for explaining ROS 1 vs ROS 2 and DDS importance
- Question 1.2: 5 points (1.5 for each policy + 0.5 for example)
- Question 1.3: 5 points for lifecycle explanation
- Question 1.4: 5 points for composition advantages

### Part 2: Nodes, Topics, and Services (25 points)
- Question 2.1: 8 points for correct publisher code
- Question 2.2: 8 points for correct service server code
- Question 2.3: 9 points (3 for each pattern explanation)

### Part 3: Python with rclpy (20 points)
- Question 3.1: 10 points for complete node with all requirements
- Question 3.2: 10 points (3-4 points each concept)

### Part 4: URDF for Humanoids (20 points)
- Question 4.1: 8 points for complete URDF
- Question 4.2: 7 points for Xacro explanation and example
- Question 4.3: 5 points for inertial properties explanation

### Part 5: Launch Files (15 points)
- Question 5.1: 8 points for complete launch file
- Question 5.2: 7 points for concepts explanation

### Practical Exercise (30 points)
- Exercise 5.1: 30 points distributed across components

## Passing Criteria

To pass this assessment, you must achieve:
- Minimum 70% (70/100) overall score
- Minimum 60% in each section
- Complete implementation of the practical exercise

## Learning Objectives Review

This assessment covers:
- Understanding the DDS-based architecture of ROS 2 ✓
- Creating and managing ROS 2 nodes effectively ✓
- Implementing publisher-subscriber communication using topics ✓
- Designing and implementing request-response communication using services ✓
- Mastering the rclpy client library for ROS 2 Python development ✓
- Creating detailed humanoid robot models using URDF ✓
- Creating launch files for complex robot systems ✓

## Answers to Part 2, Question 2.1 (Example Solution)

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from datetime import datetime

class TimePublisher(Node):
    def __init__(self):
        super().__init__('time_publisher')
        self.publisher_ = self.create_publisher(String, 'current_time', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = f'Current time: {datetime.now().isoformat()}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    time_publisher = TimePublisher()
    rclpy.spin(time_publisher)
    time_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Answers to Part 2, Question 2.2 (Example Solution)

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class AddService(Node):
    def __init__(self):
        super().__init__('add_service')
        self.srv = self.create_service(
            AddTwoInts,
            'add_two_ints',
            self.add_callback
        )

    def add_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'{request.a} + {request.b} = {response.sum}')
        return response

def main(args=None):
    rclpy.init(args=args)
    add_service = AddService()
    rclpy.spin(add_service)
    add_service.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Answers to Part 3, Question 3.1 (Example Solution)

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from datetime import datetime

class RobotNode(Node):
    def __init__(self):
        super().__init__('robot_node')

        # Declare parameter
        self.declare_parameter('robot_name', 'default_robot')
        self.robot_name = self.get_parameter('robot_name').value

        # Publisher
        self.status_publisher = self.create_publisher(String, 'status', 10)

        # Subscriber
        self.command_subscriber = self.create_subscription(
            String,
            'commands',
            self.command_callback,
            10
        )

        # Timer
        self.timer = self.create_timer(2.0, self.status_timer_callback)

        self.get_logger().info(f'Robot node initialized: {self.robot_name}')

    def command_callback(self, msg):
        self.get_logger().info(f'Received command: {msg.data}')

    def status_timer_callback(self):
        msg = String()
        msg.data = f'{self.robot_name}: {datetime.now().isoformat()}'
        self.status_publisher.publish(msg)
        self.get_logger().info(f'Published status: {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    robot_node = RobotNode()
    rclpy.spin(robot_node)
    robot_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Answers to Part 4, Question 4.1 (Example Solution)

```xml
<?xml version="1.0"?>
<robot name="simple_robot">
  <material name="white">
    <color rgba="1 1 1 1"/>
  </material>

  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.083" ixy="0" ixz="0" iyy="0.083" iyz="0" izz="0.125"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </collision>
  </link>

  <joint name="base_to_head" type="fixed">
    <parent link="base_link"/>
    <child link="head_link"/>
    <origin xyz="0 0 0.2"/>
  </joint>

  <link name="head_link">
    <inertial>
      <mass value="0.2"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.004" ixy="0" ixz="0" iyy="0.004" iyz="0" izz="0.004"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
  </link>
</robot>
```

## Answers to Part 5, Question 5.1 (Example Solution)

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():
    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='my_robot',
        description='Name of the robot'
    )

    launch_debug_arg = DeclareLaunchArgument(
        'launch_debug',
        default_value='false',
        description='Launch debug node'
    )

    robot_name = LaunchConfiguration('robot_name')
    launch_debug = LaunchConfiguration('launch_debug')

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {'robot_description': PathJoinSubstitution([
                FindPackageShare('my_robot_description'),
                'urdf',
                'robot.urdf'
            ])}
        ]
    )

    controller_node = Node(
        package='my_robot_controller',
        executable='controller_node',
        name='controller_node',
        parameters=[
            {'robot_name': robot_name}
        ]
    )

    debug_node = Node(
        package='my_robot_debug',
        executable='debug_node',
        name='debug_node',
        condition=IfCondition(launch_debug)
    )

    return LaunchDescription([
        robot_name_arg,
        launch_debug_arg,
        LogInfo(msg=['Launching for robot: ', robot_name]),
        robot_state_publisher,
        controller_node,
        debug_node
    ])
```

## Next Steps

After completing this assessment, proceed to [Module 2: The Digital Twin](../module-2/intro.md) to learn about simulation environments.

## Additional Resources

- ROS 2 Documentation: https://docs.ros.org/en/humble/
- ROS 2 Tutorials: https://docs.ros.org/en/humble/Tutorials.html
- URDF Tutorials: http://wiki.ros.org/urdf/Tutorials
- Launch System: https://github.com/ros2/launch

## Practical Exercise Solution Outline

For the practical exercise, you would need to create:

1. **Package Structure**:
```
my_robot_package/
├── CMakeLists.txt
├── package.xml
├── setup.py
├── setup.cfg
└── my_robot_package/
    ├── __init__.py
    ├── sensor_node.py
    ├── processor_node.py
    ├── service_node.py
    └── urdf/
        ├── robot.xacro
        └── materials.xacro
    └── launch/
        └── robot_system.launch.py
```

2. **Implementation of each component** as described in the exercise requirements.

## Assessment Completion

When you have completed this assessment:
1. Review all questions and ensure you've addressed each requirement
2. Test your practical exercise code in a ROS 2 environment
3. Verify that all components work together as expected
4. Document any challenges you encountered and how you resolved them

This completes Module 1: ROS 2 Fundamentals. You should now have a solid foundation in ROS 2 concepts and be ready to explore simulation environments in Module 2.