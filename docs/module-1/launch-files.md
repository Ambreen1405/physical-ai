---
id: launch-files
title: Launch Files
sidebar_position: 6
description: Comprehensive guide to ROS 2 launch files for configuring and launching robot systems
keywords: [ros2 launch, launch files, robot configuration, system launch, launch parameters]
---

# Launch Files

Launch files in ROS 2 provide a powerful way to configure and launch complex robot systems with multiple nodes, parameters, and configurations. This chapter covers everything you need to know about creating and using launch files for robot systems.

## Learning Objectives

- Understand the structure and syntax of ROS 2 launch files
- Create launch files for complex robot systems
- Use parameters, arguments, and conditions in launch files
- Implement advanced launch patterns and best practices
- Debug and troubleshoot launch file configurations

## Introduction to Launch Files

### What are Launch Files?

Launch files in ROS 2 are Python scripts that define how to launch multiple nodes and configure the ROS system. They replace the XML-based launch files from ROS 1 and provide more flexibility and power through Python.

### Basic Launch File Structure

```python
# basic_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='demo_nodes_cpp',
            executable='talker',
            name='talker_node'
        ),
        Node(
            package='demo_nodes_cpp',
            executable='listener',
            name='listener_node'
        )
    ])
```

### Launch File Execution

```bash
# Launch a launch file
ros2 launch my_package basic_launch.py

# Launch with arguments
ros2 launch my_package advanced_launch.py robot_name:=robot1
```

## Basic Launch File Components

### LaunchDescription

The `LaunchDescription` is the container for all launch actions:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Create launch configuration
    log_message = LaunchConfiguration('log_message', default='Hello World')

    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'log_message',
            default_value='Hello World',
            description='Message to log'
        ),

        # Log the message
        LogInfo(msg=log_message),

        # Launch nodes
        Node(
            package='demo_nodes_cpp',
            executable='talker',
            name='talker_node'
        )
    ])
```

### Launch Arguments

Launch arguments allow parameterizing launch files:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='default_robot',
        description='Name of the robot'
    )

    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Robot namespace'
    )

    # Use launch configurations
    robot_name = LaunchConfiguration('robot_name')
    namespace = LaunchConfiguration('namespace')

    return LaunchDescription([
        robot_name_arg,
        namespace_arg,

        Node(
            package='my_robot_package',
            executable='robot_controller',
            name='controller',
            namespace=namespace,
            parameters=[
                {'robot_name': robot_name}
            ]
        )
    ])
```

## Node Launching with Parameters

### Basic Node Launching

```python
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
                {'param2': 42},
                {'param3': True}
            ],
            remappings=[
                ('original_topic', 'remapped_topic'),
                ('/original/service', '/remapped/service')
            ],
            arguments=['--arg1', 'value1'],
            output='screen'  # Show output in terminal
        )
    ])
```

### Parameter Files

Launch files can load parameters from YAML files:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile

def generate_launch_description():
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value='/path/to/params.yaml',
        description='Path to parameters file'
    )

    params_file = LaunchConfiguration('params_file')

    return LaunchDescription([
        params_file_arg,

        Node(
            package='my_package',
            executable='my_node',
            name='my_node_with_params',
            parameters=[ParameterFile(params_file)]
        )
    ])
```

Example parameters file (params.yaml):
```yaml
my_node_with_params:
  ros__parameters:
    robot_name: "my_robot"
    max_velocity: 1.0
    safety_distance: 0.5
    topics:
      cmd_vel: "/cmd_vel"
      laser_scan: "/scan"
```

## Advanced Launch Patterns

### Conditional Launching

```python
from launch import LaunchDescription, LaunchCondition
from launch.actions import DeclareLaunchArgument, IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )

    enable_debug_arg = DeclareLaunchArgument(
        'enable_debug',
        default_value='false',
        description='Enable debug nodes'
    )

    # Get configurations
    use_sim_time = LaunchConfiguration('use_sim_time')
    enable_debug = LaunchConfiguration('enable_debug')

    return LaunchDescription([
        use_sim_time_arg,
        enable_debug_arg,

        # Launch nodes conditionally
        Node(
            package='my_package',
            executable='robot_driver',
            name='robot_driver',
            parameters=[
                {'use_sim_time': use_sim_time}
            ]
        ),

        # Debug node only if enabled
        Node(
            package='my_package',
            executable='debug_node',
            name='debug_node',
            condition=IfCondition(enable_debug)
        )
    ])
```

### Grouping and Namespacing

```python
from launch import LaunchDescription
from launch.actions import GroupAction
from launch_ros.actions import Node, PushRosNamespace

def generate_launch_description():
    return LaunchDescription([
        # Group nodes under a namespace
        GroupAction(
            actions=[
                PushRosNamespace('robot1'),

                Node(
                    package='navigation2',
                    executable='bt_navigator',
                    name='bt_navigator'
                ),

                Node(
                    package='navigation2',
                    executable='controller_server',
                    name='controller_server'
                ),

                Node(
                    package='navigation2',
                    executable='planner_server',
                    name='planner_server'
                )
            ]
        )
    ])
```

## Launch Substitutions

### Common Substitutions

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare arguments
    robot_model_arg = DeclareLaunchArgument(
        'robot_model',
        default_value='my_robot',
        description='Robot model name'
    )

    # Use substitutions
    robot_model = LaunchConfiguration('robot_model')
    config_path = PathJoinSubstitution([
        FindPackageShare('my_robot_description'),
        'config',
        [robot_model, '.yaml']
    ])

    return LaunchDescription([
        robot_model_arg,

        # Set environment variable
        SetEnvironmentVariable(
            name='MY_ROBOT_MODEL',
            value=robot_model
        ),

        Node(
            package='my_robot_package',
            executable='robot_bringup',
            name=[robot_model, '_bringup'],  # Concatenate strings
            parameters=[config_path]
        )
    ])
```

## Complex Robot System Launch

### Complete Humanoid Robot Launch

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )

    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='humanoid_robot',
        description='Name of the robot'
    )

    enable_vision_arg = DeclareLaunchArgument(
        'enable_vision',
        default_value='true',
        description='Enable vision processing nodes'
    )

    # Configurations
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_name = LaunchConfiguration('robot_name')
    enable_vision = LaunchConfiguration('enable_vision')

    # Package locations
    pkg_robot_description = FindPackageShare('humanoid_description')
    pkg_robot_bringup = FindPackageShare('humanoid_bringup')

    return LaunchDescription([
        use_sim_time_arg,
        robot_name_arg,
        enable_vision_arg,

        # Robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'robot_description': PathJoinSubstitution([
                    pkg_robot_description,
                    'urdf',
                    'humanoid.urdf.xacro'
                ])}
            ],
            remappings=[
                ('/joint_states', [robot_name, '/joint_states'])
            ]
        ),

        # Joint state publisher (GUI for testing)
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            parameters=[
                {'use_sim_time': use_sim_time}
            ]
        ),

        # Robot controller
        Node(
            package='humanoid_controller',
            executable='controller_node',
            name='controller_node',
            parameters=[
                PathJoinSubstitution([
                    pkg_robot_bringup,
                    'config',
                    'controller.yaml'
                ]),
                {'use_sim_time': use_sim_time},
                {'robot_name': robot_name}
            ],
            output='screen'
        ),

        # Vision processing (conditional)
        Node(
            package='humanoid_vision',
            executable='vision_node',
            name='vision_node',
            parameters=[
                PathJoinSubstitution([
                    pkg_robot_bringup,
                    'config',
                    'vision.yaml'
                ]),
                {'use_sim_time': use_sim_time}
            ],
            condition=IfCondition(enable_vision),
            output='screen'
        ),

        # Launch additional files
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    pkg_robot_bringup,
                    'launch',
                    'navigation.launch.py'
                ])
            ),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'robot_name': robot_name
            }.items()
        )
    ])
```

## Launch Actions and Utilities

### Timer Actions

```python
from launch import LaunchDescription, TimerAction
from launch.actions import LogInfo
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Launch initial nodes immediately
        Node(
            package='my_package',
            executable='initial_node',
            name='initial_node'
        ),

        # Launch other nodes after delay
        TimerAction(
            period=5.0,  # Wait 5 seconds
            actions=[
                Node(
                    package='my_package',
                    executable='delayed_node',
                    name='delayed_node'
                ),
                LogInfo(msg='Delayed node launched after 5 seconds')
            ]
        )
    ])
```

### Execute Process

```python
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    return LaunchDescription([
        # Execute shell commands
        ExecuteProcess(
            cmd=['ros2', 'param', 'set', '/my_node', 'param_name', 'param_value'],
            output='screen'
        ),

        # Execute external programs
        ExecuteProcess(
            cmd=['rviz2', '-d', PathJoinSubstitution([
                FindPackageShare('my_pkg'),
                'rviz',
                'config.rviz'
            ])],
            output='screen'
        )
    ])
```

## Parameter Configuration

### Complex Parameter Setup

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    config_arg = DeclareLaunchArgument(
        'config',
        default_value='default',
        description='Configuration profile to use'
    )

    config = LaunchConfiguration('config')

    # Determine config file based on argument
    config_file = PathJoinSubstitution([
        FindPackageShare('my_robot_config'),
        'config',
        [config, '.yaml']
    ])

    return LaunchDescription([
        config_arg,

        Node(
            package='my_robot_navigation',
            executable='nav2_bringup',
            name='nav2_bringup',
            parameters=[
                config_file,
                {'use_sim_time': True},
                {'autostart': True},
                {'map_subscribe_transient_local': True}
            ],
            output='screen'
        )
    ])
```

## Launch File Best Practices

### Modular Launch Files

```python
# main_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Main launch file that includes others
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                FindPackageShare('my_robot_description'),
                '/launch',
                '/robot_state_publisher.launch.py'
            ])
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                FindPackageShare('my_robot_control'),
                '/launch',
                '/controller.launch.py'
            ])
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                FindPackageShare('my_robot_navigation'),
                '/launch',
                '/navigation.launch.py'
            ])
        )
    ])
```

### Error Handling and Validation

```python
from launch import LaunchDescription, LaunchContext
from launch.actions import LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch.launch_description_entity import LaunchDescriptionEntity
from launch.some_substitutions_type import SomeSubstitutionsType
from launch.utilities import perform_substitutions

def validate_config(context: LaunchContext):
    """Custom validation function"""
    config_file = LaunchConfiguration('config_file').perform(context)

    # Perform validation
    if not config_file.endswith('.yaml'):
        context.launch_logger.error(f'Config file must be YAML: {config_file}')
        return []

    return [LogInfo(msg=f'Validated config file: {config_file}')]

def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=validate_config),

        # Other launch actions...
    ])
```

## Debugging Launch Files

### Debug Information

```python
from launch import LaunchDescription
from launch.actions import LogInfo, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, EnvironmentVariable

def generate_launch_description():
    return LaunchDescription([
        # Log system information
        LogInfo(
            msg=['ROS_DOMAIN_ID: ', EnvironmentVariable(name='ROS_DOMAIN_ID', default_value='0')]
        ),

        # Log launch arguments
        DeclareLaunchArgument(
            'debug_level',
            default_value='info',
            description='Debug level'
        ),

        LogInfo(
            msg=['Debug level: ', LaunchConfiguration('debug_level')]
        )
    ])
```

## Advanced Launch Techniques

### Custom Launch Actions

```python
from launch import LaunchDescription
from launch.actions import Action
from launch.some_actions_type import SomeActionsType
from launch.launch_context import LaunchContext

class CustomValidationAction(Action):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def execute(self, context: LaunchContext) -> SomeActionsType:
        # Perform custom validation
        context.launch_logger.info('Performing custom validation...')

        # Return additional actions if needed
        return []
```

### Launch File Composition

```python
# composite_launch.py
from launch import LaunchDescription
from launch.actions import GroupAction, SetEnvironmentVariable
from launch_ros.actions import Node, PushRosNamespace

def generate_launch_description():
    return LaunchDescription([
        # Set environment for all nodes
        SetEnvironmentVariable(
            name='CUSTOM_ENV',
            value='robot_system'
        ),

        # Robot 1 group
        GroupAction(
            actions=[
                PushRosNamespace('robot1'),
                Node(
                    package='my_package',
                    executable='robot_node',
                    name='robot_node'
                )
            ]
        ),

        # Robot 2 group
        GroupAction(
            actions=[
                PushRosNamespace('robot2'),
                Node(
                    package='my_package',
                    executable='robot_node',
                    name='robot_node'
                )
            ]
        )
    ])
```

## Common Launch File Patterns

### Sensor Processing Pipeline

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'sensor_namespace',
            default_value='laser',
            description='Namespace for sensor nodes'
        ),

        # Sensor driver
        Node(
            package='velodyne_driver',
            executable='velodyne_node',
            name='velodyne_driver',
            namespace=LaunchConfiguration('sensor_namespace'),
            parameters=[
                {'device_ip': '192.168.1.201'},
                {'frame_id': 'laser_link'}
            ]
        ),

        # Point cloud processing
        Node(
            package='point_cloud_processing',
            executable='filter_node',
            name='pointcloud_filter',
            namespace=LaunchConfiguration('sensor_namespace'),
            parameters=[
                {'voxel_leaf_size': 0.1}
            ],
            remappings=[
                ('input', 'velodyne_points'),
                ('output', 'filtered_points')
            ]
        )
    ])
```

## Launch File Testing

### Testing Launch Files

```python
# test_launch.py
import unittest
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_testing.actions import ReadyToTest

def generate_test_description():
    return LaunchDescription([
        # Launch nodes for testing
        ExecuteProcess(
            cmd=['ros2', 'run', 'my_package', 'test_node'],
            output='screen'
        ),

        # Ready to test
        ReadyToTest()
    ])

class TestLaunchFile(unittest.TestCase):
    def test_node_launches(self):
        # Test implementation
        pass
```

## Next Steps

Continue to [Module 1 Assessment](./assessment.md) to test your knowledge of ROS 2 fundamentals.

## Learning Objectives Review

- Understand the structure and syntax of ROS 2 launch files ✓
- Create launch files for complex robot systems ✓
- Use parameters, arguments, and conditions in launch files ✓
- Implement advanced launch patterns and best practices ✓
- Debug and troubleshoot launch file configurations ✓

## Practical Exercise

Create a launch file for a simple robot system that includes:
1. Robot state publisher with URDF
2. Joint state publisher
3. A controller node with parameters
4. Conditional launching based on arguments
5. Proper namespacing and remappings

## Assessment Questions

1. What is the difference between LaunchConfiguration and DeclareLaunchArgument?
2. How do you conditionally launch nodes based on arguments?
3. Explain the purpose of PathJoinSubstitution in launch files.
4. How can you include other launch files within a launch file?

## Further Reading

- ROS 2 Launch Documentation: https://docs.ros.org/en/humble/p/launch/
- Launch System Design: https://github.com/ros2/launch
- ROS 2 Launch Tutorials: https://docs.ros.org/en/humble/Tutorials/Intermediate/Launch/Creating-Launch-Files.html
- Launch System API: https://launch.ros.org/api/launch/