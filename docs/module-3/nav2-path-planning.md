---
id: nav2-path-planning
title: Nav2 Path Planning
sidebar_position: 6
description: Comprehensive guide to path planning using Nav2 for humanoid robotics applications
keywords: [nav2, path planning, navigation, robotics, pathfinding, humanoid navigation, navigation2]
---

# Nav2 Path Planning: Bipedal Navigation Systems

Navigation is a fundamental capability for humanoid robots, enabling them to move safely and efficiently through environments. This chapter covers the Nav2 framework, which provides a comprehensive navigation system for robots, with specific focus on humanoid applications.

## Learning Objectives

- Understand Nav2 architecture and components
- Configure Nav2 for humanoid robotics applications
- Implement path planning algorithms for bipedal locomotion
- Integrate perception systems with navigation
- Optimize navigation for dynamic environments
- Troubleshoot common navigation issues

## Introduction to Nav2

### What is Nav2?

Nav2 (Navigation 2) is the next-generation navigation framework for ROS 2, designed to provide safe and reliable path planning and navigation for robots. It builds upon the lessons learned from ROS 1's navigation stack with significant improvements.

### Key Features of Nav2

- **Modular Architecture**: Pluggable components for flexibility
- **Behavior Trees**: Declarative behavior specification
- **Advanced Planning**: Sophisticated path planning algorithms
- **Recovery Behaviors**: Robust recovery from navigation failures
- **Simulation Integration**: Seamless simulation-to-real transfer
- **Performance**: Optimized for real-time applications

### Nav2 vs Legacy Navigation Stack

| Aspect | Nav2 | Legacy Navigation |
|--------|------|-------------------|
| Architecture | Component-based with BT | Monolithic nodes |
| Behavior Logic | Behavior Trees | Hardcoded state machines |
| Flexibility | Highly configurable | Limited customization |
| Recovery | Sophisticated recovery | Basic recovery |
| Performance | Optimized for ROS 2 | Designed for ROS 1 |
| Simulation | Integrated simulation | Separate tools |

## Nav2 Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        Nav2 Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   Planner       │  │   Controller    │  │   Recovery      │  │
│  │   Server        │  │   Server        │  │   Server        │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│         │                       │                       │       │
│         ▼                       ▼                       ▼       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   Global        │  │   Local         │  │   Recovery      │  │
│  │   Planner       │  │   Planner       │  │   Behaviors     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│         │                       │                       │       │
│         └───────────────────────┼───────────────────────┘       │
│                                 ▼                               │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                Behavior Tree Executor                       │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                 │                               │
│         ┌───────────────────────┼───────────────────────┐       │
│         ▼                       ▼                       ▼       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   Costmap       │  │   Transform     │  │   Sensors       │  │
│  │   Server        │  │   Server        │  │   Server        │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Behavior Tree Integration

Nav2 uses Behavior Trees (BT) to orchestrate navigation tasks:

```xml
<root main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree">
    <PipelineSequence name="NavigateWithReplanning">
      <RateController hz="1.0">
        <ComputePathToPose goal="{goal}" path="{path}" planner_id="GridBased"/>
      </RateController>
      <FollowPath path="{path}" controller_id="FollowPath"/>
    </PipelineSequence>
  </BehaviorTree>
</root>
```

## Nav2 Installation and Setup

### Installing Nav2

```bash
# Install Nav2 packages
sudo apt update
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup

# Install additional tools
sudo apt install ros-humble-nav2-rviz-plugins
sudo apt install ros-humble-nav2-system-tests
sudo apt install ros-humble-nav2-map-server
sudo apt install ros-humble-nav2-lifecycle-manager
sudo apt install ros-humble-nav2-planners
sudo apt install ros-humble-nav2-controller
sudo apt install ros-humble-nav2-behaviors
```

### Basic Nav2 Launch

```python
# launch/basic_nav2_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    map_yaml_file = LaunchConfiguration('map', default='')
    params_file = LaunchConfiguration('params_file')
    autostart = LaunchConfiguration('autostart', default='true')
    rviz_config_file = LaunchConfiguration('rviz_config', default='nav2_default_view.rviz')

    # Package locations
    pkg_nav2_bringup = FindPackageShare('nav2_bringup')
    pkg_nav2_bt_navigator = FindPackageShare('nav2_bt_navigator')

    # Launch files
    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                pkg_nav2_bringup,
                'launch',
                'localization.launch.py'
            ])
        ]),
        launch_arguments={
            'map': map_yaml_file,
            'use_sim_time': use_sim_time,
            'params_file': params_file
        }.items()
    )

    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                pkg_nav2_bringup,
                'launch',
                'navigation.launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': params_file,
            'autostart': autostart
        }.items()
    )

    # RViz node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', PathJoinSubstitution([pkg_nav2_bringup, 'rviz', rviz_config_file])],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'map',
            default_value='',
            description='Full path to map file to load'
        ),
        DeclareLaunchArgument(
            'params_file',
            default_value=PathJoinSubstitution([
                FindPackageShare('my_robot_navigation'),
                'config',
                'nav2_params.yaml'
            ]),
            description='Full path to the ROS2 parameters file to use for all launched nodes'
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'
        ),
        DeclareLaunchArgument(
            'autostart',
            default_value='true',
            description='Automatically startup the nav2 stack'
        ),
        DeclareLaunchArgument(
            'rviz_config',
            default_value='nav2_default_view.rviz',
            description='Full path to the RViz config file to use'
        ),
        localization_launch,
        navigation_launch,
        rviz_node
    ])
```

## Configuration Files

### Nav2 Parameters Configuration

```yaml
# config/nav2_params.yaml
amcl:
  ros__parameters:
    use_sim_time: False
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_footprint"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: false
    global_frame_id: "map"
    lambda_short: 0.1
    likelihood_max_dist: 2.0
    max_beams: 60
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.5
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    save_pose_rate: 0.5
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.25
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05
    scan_topic: scan

amcl_map_client:
  ros__parameters:
    use_sim_time: False

amcl_rclcpp_node:
  ros__parameters:
    use_sim_time: False

bt_navigator:
  ros__parameters:
    use_sim_time: False
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    bt_loop_duration: 10
    default_server_timeout: 20
    enable_groot_monitoring: True
    enable_groot_recording: False
    groot_recording_options: {}
    default_nav_through_poses_bt_xml: /opt/ros/humble/share/nav2_bt_navigator/behavior_trees/navigate_w_replanning_and_recovery.xml
    default_navigate_to_pose_bt_xml: /opt/ros/humble/share/nav2_bt_navigator/behavior_trees/navigate_w_replanning_and_recovery.xml
    plugin_lib_names:
    - nav2_compute_path_to_pose_action_bt_node
    - nav2_compute_path_through_poses_action_bt_node
    - nav2_smooth_path_action_bt_node
    - nav2_follow_path_action_bt_node
    - nav2_spin_action_bt_node
    - nav2_wait_action_bt_node
    - nav2_back_up_action_bt_node
    - nav2_drive_on_heading_bt_node
    - nav2_clear_costmap_service_bt_node
    - nav2_is_stuck_condition_bt_node
    - nav2_goal_reached_condition_bt_node
    - nav2_goal_updated_condition_bt_node
    - nav2_globally_cleared_condition_bt_node
    - nav2_locally_cleared_condition_bt_node
    - nav2_map_accessed_condition_bt_node
    - nav2_initial_pose_received_condition_bt_node
    - nav2_reinitialize_global_localization_service_bt_node
    - nav2_rate_controller_bt_node
    - nav2_distance_controller_bt_node
    - nav2_speed_controller_bt_node
    - nav2_truncate_path_action_bt_node
    - nav2_truncate_path_local_action_bt_node
    - nav2_goal_updater_node_bt_node
    - nav2_recovery_node_bt_node
    - nav2_pipeline_sequence_bt_node
    - nav2_round_robin_node_bt_node
    - nav2_transform_available_condition_bt_node
    - nav2_time_expired_condition_bt_node
    - nav2_path_expiring_timer_condition
    - nav2_distance_traveled_condition_bt_node
    - nav2_single_trigger_bt_node
    - nav2_is_battery_low_condition_bt_node
    - nav2_navigate_through_poses_action_bt_node
    - nav2_navigate_to_pose_action_bt_node
    - nav2_remove_passed_goals_action_bt_node
    - nav2_planner_selector_bt_node
    - nav2_controller_selector_bt_node
    - nav2_goal_checker_selector_bt_node
    - nav2_controller_cancel_bt_node
    - nav2_path_longer_on_approach_bt_node
    - nav2_wait_cancel_bt_node
    - nav2_spin_cancel_bt_node
    - nav2_back_up_cancel_bt_node
    - nav2_drive_on_heading_cancel_bt_node

bt_navigator_rclcpp_node:
  ros__parameters:
    use_sim_time: False

controller_server:
  ros__parameters:
    use_sim_time: False
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # Progress checker parameters
    progress_checker:
      plugin: "nav2_controller::SimpleProgressChecker"
      required_movement_radius: 0.5
      movement_time_allowance: 10.0

    # Goal checker parameters
    goal_checker:
      plugin: "nav2_controller::SimpleGoalChecker"
      xy_goal_tolerance: 0.25
      yaw_goal_tolerance: 0.25
      stateful: True

    # Controller parameters
    FollowPath:
      plugin: "nav2_rotation_shim_controller::RotationShimController"
      # Base local planner parameters
      approximate_linear_controller: "dwb_core::DWBLocalPlanner"
      stopping_feeback: "stopping"
      rotating_feeback: "rotating_cmd"
      moving_feeback: "moving_cmd"
      approximate_angular_controller: "dwb_core::DWBLocalPlanner"
      exact_angular_controller: "nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController"

dwb_core:
  ros__parameters:
    use_sim_time: False
    debug_trajectory_details: False
    min_vel_x: 0.0
    min_vel_y: 0.0
    max_vel_x: 0.5
    max_vel_y: 0.0
    max_vel_theta: 1.0
    min_speed_xy: 0.0
    max_speed_xy: 0.5
    min_speed_theta: 0.0
    acc_lim_x: 2.5
    acc_lim_y: 0.0
    acc_lim_theta: 3.2
    decel_lim_x: -2.5
    decel_lim_y: 0.0
    decel_lim_theta: -3.2
    vx_samples: 20
    vy_samples: 5
    vtheta_samples: 20
    sim_time: 1.7
    linear_granularity: 0.05
    angular_granularity: 0.025
    transform_tolerance: 0.2
    xy_goal_tolerance: 0.25
    trans_stopped_velocity: 0.25
    short_circuit_trajectory_evaluation: True
    stateful: True
    critics: ["RotateToGoal", "Oscillation", "BaseObstacle", "GoalAlign", "PathAlign", "PathDist", "GoalDist"]
    BaseObstacle.scale: 0.02
    PathAlign.scale: 32.0
    PathAlign.forward_point_distance: 0.1
    GoalAlign.scale: 24.0
    GoalAlign.forward_point_distance: 0.1
    PathDist.scale: 32.0
    GoalDist.scale: 24.0
    RotateToGoal.scale: 32.0
    RotateToGoal.slowing_factor: 5.0
    RotateToGoal.lookahead_time: -1.0

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_link
      use_sim_time: False
      rolling_window: true
      width: 3
      height: 3
      resolution: 0.05
      robot_radius: 0.22
      plugins: ["voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: True
        publish_voxel_map: False
        origin_z: 0.0
        z_resolution: 0.2
        z_voxels: 10
        max_obstacle_height: 2.0
        mark_threshold: 0
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        map_subscribe_transient_local: True
      always_send_full_costmap: True
  local_costmap_client:
    ros__parameters:
      use_sim_time: False
  local_costmap_rclcpp_node:
    ros__parameters:
      use_sim_time: False

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: map
      robot_base_frame: base_link
      use_sim_time: False
      robot_radius: 0.22
      resolution: 0.05
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      always_send_full_costmap: True
  global_costmap_client:
    ros__parameters:
      use_sim_time: False
  global_costmap_rclcpp_node:
    ros__parameters:
      use_sim_time: False

map_server:
  ros__parameters:
    use_sim_time: False
    yaml_filename: ""

map_saver:
  ros__parameters:
    use_sim_time: False
    save_map_timeout: 5.0
    free_thresh_default: 0.25
    occupied_thresh_default: 0.65
    map_subscribe_transient_local: True

planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    use_sim_time: False
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner::NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true

smoother_server:
  ros__parameters:
    use_sim_time: False
    smoother_plugins: ["simple_smoother"]
    simple_smoother:
      plugin: "nav2_smoother::SimpleSmoother"
      tolerance: 1.0e-10
      max_its: 1000
      weight_smooth: 0.9
      weight_data: 0.1

behavior_server:
  ros__parameters:
    costmap_topic: local_costmap/costmap_raw
    footprint_topic: local_costmap/published_footprint
    cycle_frequency: 10.0
    behavior_plugins: ["spin", "backup", "drive_on_heading", "wait"]
    spin:
      plugin: "nav2_behaviors::Spin"
      spin_dist: 1.57
    backup:
      plugin: "nav2_behaviors::BackUp"
      backup_dist: 0.15
      backup_speed: 0.025
    drive_on_heading:
      plugin: "nav2_behaviors::DriveOnHeading"
      drive_on_heading_angle_tol: 0.785
      drive_on_heading_forward_dist: 0.5
      drive_on_heading_max_drive_dist: 1.0
    wait:
      plugin: "nav2_behaviors::Wait"
      wait_duration: 1.0

waypoint_follower:
  ros__parameters:
    loop_rate: 20
    stop_on_failure: false
    waypoint_task_executor_plugin: "wait_at_waypoint"
    wait_at_waypoint:
      plugin: "nav2_waypoint_follower::WaitAtWaypoint"
      enabled: true
      wait_time: 1
```

## Path Planning Algorithms

### Global Path Planning

```python
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import LaserScan
from nav2_msgs.action import ComputePathToPose
from rclpy.action import ActionClient
import numpy as np

class Nav2GlobalPlanner(Node):
    def __init__(self):
        super().__init__('nav2_global_planner')

        # Action client for path computation
        self.path_client = ActionClient(
            self,
            ComputePathToPose,
            'compute_path_to_pose'
        )

        # Publishers
        self.global_path_pub = self.create_publisher(Path, '/plan', 10)

        # Subscribers
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/move_base_simple/goal',
            self.goal_callback,
            10
        )

        # Internal state
        self.current_path = None
        self.waypoints = []
        self.path_valid = False

        self.get_logger().info('Nav2 Global Planner initialized')

    def goal_callback(self, goal_msg):
        """Receive navigation goal and compute path"""
        self.get_logger().info(f'Received navigation goal: {goal_msg.pose.position.x}, {goal_msg.pose.position.y}')

        # Wait for action server
        if not self.path_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error('Path planner action server not available')
            return

        # Create goal message
        goal = ComputePathToPose.Goal()
        goal.goal = goal_msg.pose
        goal.planner_id = "GridBased"  # Use Navfn planner

        # Send goal
        future = self.path_client.send_goal_async(goal)
        future.add_done_callback(self.path_result_callback)

    def path_result_callback(self, future):
        """Handle path computation result"""
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().info('Goal rejected')
                return

            self.get_logger().info('Goal accepted, getting result...')

            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self.get_path_result)

        except Exception as e:
            self.get_logger().error(f'Exception in path result callback: {e}')

    def get_path_result(self, future):
        """Process path result"""
        try:
            result = future.result().result

            if result.error_code == 0:  # SUCCESS
                self.current_path = result.path
                self.path_valid = True

                # Publish path
                self.global_path_pub.publish(self.current_path)

                self.get_logger().info(f'Computed path with {len(self.current_path.poses)} waypoints')

                # Trigger local planner with path
                self.execute_navigation_path()
            else:
                self.get_logger().error(f'Path computation failed with error code: {result.error_code}')
                self.path_valid = False

        except Exception as e:
            self.get_logger().error(f'Exception in get path result: {e}')

    def execute_navigation_path(self):
        """Execute navigation along computed path"""
        # This would trigger the BT navigator to follow the path
        # In practice, this integrates with the Nav2 BT navigator
        pass

    def get_path_to_waypoint(self, start, goal):
        """Get path between start and goal positions"""
        if not self.path_client.wait_for_server(timeout_sec=1.0):
            return None

        goal_msg = ComputePathToPose.Goal()
        goal_msg.goal.pose.position.x = goal[0]
        goal_msg.goal.pose.position.y = goal[1]
        goal_msg.goal.pose.position.z = 0.0

        # Set orientation (simple approach)
        goal_msg.goal.pose.orientation.w = 1.0

        future = self.path_client.send_goal_async(goal_msg)
        future.add_done_callback(lambda f: self.waypoint_path_callback(f, start, goal))

    def waypoint_path_callback(self, future, start, goal):
        """Handle waypoint path result"""
        try:
            goal_handle = future.result()
            if goal_handle.accepted:
                result_future = goal_handle.get_result_async()
                result_future.add_done_callback(
                    lambda f: self.process_waypoint_result(f, start, goal)
                )
        except Exception as e:
            self.get_logger().error(f'Waypoint path error: {e}')

    def process_waypoint_result(self, future, start, goal):
        """Process waypoint navigation result"""
        try:
            result = future.result().result
            if result.error_code == 0:
                path = result.path
                self.get_logger().info(f'Waypoint path from {start} to {goal} computed')
                # Process path as needed
            else:
                self.get_logger().error(f'Waypoint path computation failed: {result.error_code}')
        except Exception as e:
            self.get_logger().error(f'Waypoint result error: {e}')
```

### Local Path Planning and Trajectory Generation

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan, PointCloud2
from nav_msgs.msg import Path, OccupancyGrid
from visualization_msgs.msg import MarkerArray
import numpy as np
import math
from scipy.spatial import KDTree

class Nav2LocalPlanner(Node):
    def __init__(self):
        super().__init__('nav2_local_planner')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.local_plan_pub = self.create_publisher(Path, '/local_plan', 10)
        self.velocity_marker_pub = self.create_publisher(MarkerArray, '/velocity_markers', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        self.global_plan_sub = self.create_subscription(
            Path,
            '/plan',
            self.global_plan_callback,
            10
        )
        self.local_costmap_sub = self.create_subscription(
            OccupancyGrid,
            '/local_costmap/costmap',
            self.costmap_callback,
            10
        )

        # Parameters
        self.robot_radius = 0.22  # meters
        self.max_linear_vel = 0.5  # m/s
        self.max_angular_vel = 1.0  # rad/s
        self.min_linear_vel = 0.05  # m/s
        self.min_angular_vel = 0.05  # rad/s
        self.controller_frequency = 20.0  # Hz
        self.lookahead_distance = 0.5  # meters

        # Internal state
        self.current_pose = None
        self.current_velocity = Twist()
        self.global_plan = None
        self.local_plan = Path()
        self.scan_data = None
        self.costmap = None
        self.current_goal_idx = 0

        # Timer for control loop
        self.control_timer = self.create_timer(
            1.0 / self.controller_frequency,
            self.control_loop
        )

        self.get_logger().info('Nav2 Local Planner initialized')

    def odom_callback(self, msg):
        """Update robot pose and velocity from odometry"""
        self.current_pose = msg.pose.pose
        self.current_velocity = msg.twist.twist

    def scan_callback(self, msg):
        """Process laser scan data for obstacle detection"""
        self.scan_data = msg

    def global_plan_callback(self, msg):
        """Receive global path and update local planner"""
        self.global_plan = msg
        self.current_goal_idx = 0
        self.get_logger().info(f'Received global plan with {len(msg.poses)} waypoints')

    def costmap_callback(self, msg):
        """Update local costmap"""
        self.costmap = msg

    def control_loop(self):
        """Main control loop for local navigation"""
        if (self.current_pose is None or
            self.global_plan is None or
            len(self.global_plan.poses) == 0):
            return

        # Update local plan
        self.update_local_plan()

        # Generate velocity command
        cmd_vel = self.generate_velocity_command()

        # Publish command
        self.cmd_vel_pub.publish(cmd_vel)

        # Publish local plan for visualization
        self.local_plan_pub.publish(self.local_plan)

    def update_local_plan(self):
        """Update the local navigation plan based on global plan"""
        if self.global_plan is None or len(self.global_plan.poses) == 0:
            return

        # Find closest point on global path
        closest_idx = self.find_closest_waypoint()
        if closest_idx is None:
            return

        # Create local plan with lookahead
        local_waypoints = []
        start_idx = max(0, closest_idx)
        end_idx = min(len(self.global_plan.poses),
                      closest_idx + int(self.lookahead_distance / 0.1))  # 0.1m spacing assumption

        for i in range(start_idx, end_idx):
            local_waypoints.append(self.global_plan.poses[i])

        # Create local path message
        self.local_plan = Path()
        self.local_plan.header.frame_id = 'map'
        self.local_plan.header.stamp = self.get_clock().now().to_msg()
        self.local_plan.poses = local_waypoints

    def find_closest_waypoint(self):
        """Find the closest waypoint on the global path"""
        if (self.current_pose is None or
            self.global_plan is None or
            len(self.global_plan.poses) == 0):
            return None

        current_pos = np.array([
            self.current_pose.position.x,
            self.current_pose.position.y
        ])

        min_dist = float('inf')
        closest_idx = 0

        for i, pose in enumerate(self.global_plan.poses):
            waypoint_pos = np.array([
                pose.pose.position.x,
                pose.pose.position.y
            ])

            dist = np.linalg.norm(current_pos - waypoint_pos)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # Update goal index if we've moved past the current target
        if min_dist < 0.5:  # 50cm threshold
            self.current_goal_idx = closest_idx

        return closest_idx

    def generate_velocity_command(self):
        """Generate velocity command based on local plan and obstacles"""
        cmd_vel = Twist()

        if (self.local_plan is None or
            len(self.local_plan.poses) == 0 or
            self.current_pose is None):
            return cmd_vel

        # Get target waypoint
        target_idx = min(self.current_goal_idx + 1, len(self.local_plan.poses) - 1)
        if target_idx >= len(self.local_plan.poses):
            target_idx = len(self.local_plan.poses) - 1

        target_pose = self.local_plan.poses[target_idx].pose
        current_pos = np.array([
            self.current_pose.position.x,
            self.current_pose.position.y
        ])
        target_pos = np.array([
            target_pose.position.x,
            target_pose.position.y
        ])

        # Calculate desired direction
        direction = target_pos - current_pos
        distance = np.linalg.norm(direction)

        if distance < 0.1:  # Very close to target, slow down
            cmd_vel.linear.x = max(0.05, min(self.max_linear_vel * 0.3, self.current_velocity.linear.x))
        else:
            # Calculate linear velocity based on distance
            cmd_vel.linear.x = min(self.max_linear_vel, distance * 0.5)  # Proportional control
            cmd_vel.linear.x = max(self.min_linear_vel, cmd_vel.linear.x)

        # Calculate angular velocity for orientation
        desired_yaw = math.atan2(direction[1], direction[0])

        # Current robot orientation
        current_yaw = self.quaternion_to_yaw(self.current_pose.orientation)

        # Calculate angle difference
        angle_diff = desired_yaw - current_yaw
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))  # Normalize to [-π, π]

        # Proportional control for angular velocity
        cmd_vel.angular.z = max(-self.max_angular_vel,
                               min(self.max_angular_vel, angle_diff * 1.0))
        cmd_vel.angular.z = max(-self.min_angular_vel,
                               min(self.min_angular_vel, cmd_vel.angular.z))

        # Check for obstacles using laser scan
        if self.scan_data is not None:
            obstacle_detected = self.check_obstacles(cmd_vel)
            if obstacle_detected:
                # Slow down or stop if obstacle detected
                cmd_vel.linear.x *= 0.3
                cmd_vel.angular.z *= 0.5

        return cmd_vel

    def check_obstacles(self, cmd_vel):
        """Check for obstacles in the planned path"""
        if self.scan_data is None:
            return False

        # Check forward direction (simplified)
        min_angle = -math.pi / 6  # -30 degrees
        max_angle = math.pi / 6   # 30 degrees

        angle_increment = self.scan_data.angle_increment
        min_idx = int((min_angle - self.scan_data.angle_min) / angle_increment)
        max_idx = int((max_angle - self.scan_data.angle_min) / angle_increment)

        min_idx = max(0, min_idx)
        max_idx = min(len(self.scan_data.ranges), max_idx)

        # Check for obstacles within robot radius + safety margin
        safety_margin = 0.3  # meters
        min_distance = self.robot_radius + safety_margin

        for i in range(min_idx, max_idx):
            if i < len(self.scan_data.ranges):
                range_val = self.scan_data.ranges[i]
                if 0 < range_val < min_distance:
                    return True

        return False

    def quaternion_to_yaw(self, quat):
        """Convert quaternion to yaw angle"""
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        return math.atan2(siny_cosp, cosy_cosp)

class Nav2BipedalController(Nav2LocalPlanner):
    """
    Specialized controller for bipedal humanoid robots that takes into account
    balance, step constraints, and legged locomotion characteristics
    """
    def __init__(self):
        super().__init__()

        # Bipedal-specific parameters
        self.step_height = 0.05  # meters
        self.step_length = 0.3   # meters
        self.step_frequency = 1.0  # Hz
        self.balance_margin = 0.1  # meters
        self.max_step_deviation = 0.1  # meters

        # Bipedal state
        self.left_foot_pose = None
        self.right_foot_pose = None
        self.support_foot = 'left'  # Current support foot
        self.step_phase = 0.0  # Current phase in step cycle (0.0 to 1.0)

        self.get_logger().info('Nav2 Bipedal Controller initialized')

    def generate_bipedal_velocity_command(self):
        """Generate velocity command suitable for bipedal locomotion"""
        # First, get the base velocity command
        base_cmd = self.generate_velocity_command()

        # Adjust for bipedal constraints
        bipedal_cmd = self.adapt_to_bipedal_constraints(base_cmd)

        return bipedal_cmd

    def adapt_to_bipedal_constraints(self, base_cmd):
        """Adapt base velocity command to bipedal robot constraints"""
        # Limit acceleration to prevent falls
        max_linear_acc = 0.3  # m/s^2
        max_angular_acc = 0.5  # rad/s^2

        # Calculate desired accelerations
        desired_linear_acc = (base_cmd.linear.x - self.current_velocity.linear.x) * self.controller_frequency
        desired_angular_acc = (base_cmd.angular.z - self.current_velocity.angular.z) * self.controller_frequency

        # Limit accelerations
        limited_linear_acc = max(-max_linear_acc, min(max_linear_acc, desired_linear_acc))
        limited_angular_acc = max(-max_angular_acc, min(max_angular_acc, desired_angular_acc))

        # Calculate limited velocities
        limited_linear_vel = self.current_velocity.linear.x + (limited_linear_acc / self.controller_frequency)
        limited_angular_vel = self.current_velocity.angular.z + (limited_angular_acc / self.controller_frequency)

        # Apply velocity limits
        result_cmd = Twist()
        result_cmd.linear.x = max(-self.max_linear_vel, min(self.max_linear_vel, limited_linear_vel))
        result_cmd.angular.z = max(-self.max_angular_vel, min(self.max_angular_vel, limited_angular_vel))

        # Ensure minimum velocities for stable walking
        if abs(result_cmd.linear.x) < self.min_linear_vel and abs(base_cmd.linear.x) > 0.01:
            result_cmd.linear.x = self.min_linear_vel if base_cmd.linear.x > 0 else -self.min_linear_vel

        if abs(result_cmd.angular.z) < self.min_angular_vel and abs(base_cmd.angular.z) > 0.01:
            result_cmd.angular.z = self.min_angular_vel if base_cmd.angular.z > 0 else -self.min_angular_vel

        return result_cmd

    def plan_bipedal_steps(self, target_velocity):
        """Plan bipedal stepping pattern to achieve target velocity"""
        # Calculate step timing based on desired velocity
        if abs(target_velocity.linear.x) > 0.01:
            # Adjust step frequency based on desired speed
            step_freq = max(0.5, min(2.0, 0.5 + abs(target_velocity.linear.x) * 1.0))
        else:
            step_freq = 0.5  # Default standing frequency

        # Plan step sequence
        step_sequence = []
        current_time = 0.0
        sequence_duration = 2.0  # Plan 2 seconds ahead

        while current_time < sequence_duration:
            step_info = self.calculate_next_step(current_time, target_velocity)
            step_sequence.append(step_info)
            current_time += 1.0 / step_freq

        return step_sequence

    def calculate_next_step(self, time, target_velocity):
        """Calculate the next step based on current state and target velocity"""
        # Calculate desired foot placement based on target velocity
        dt = 1.0 / self.step_frequency  # Time per step
        desired_travel = target_velocity.linear.x * dt

        # Current support foot position (simplified)
        if self.support_foot == 'left':
            support_foot_pos = self.left_foot_pose
            swing_foot_pos = self.right_foot_pose
        else:
            support_foot_pos = self.right_foot_pose
            swing_foot_pos = self.left_foot_pose

        if support_foot_pos is None:
            # Default to current robot position if foot poses unknown
            support_foot_pos = self.current_pose

        # Calculate next step position
        next_step_x = support_foot_pos.position.x + desired_travel
        next_step_y = support_foot_pos.position.y  # Stay centered laterally

        # Add some lateral variation for stability
        step_offset = math.sin(time * 2 * math.pi * self.step_frequency) * 0.1
        next_step_y += step_offset

        # Calculate step trajectory (simplified parabolic arc)
        step_trajectory = self.calculate_step_trajectory(
            swing_foot_pos,
            [next_step_x, next_step_y, support_foot_pos.position.z],
            self.step_height
        )

        return {
            'time': time,
            'foot': 'left' if self.support_foot == 'right' else 'right',
            'position': [next_step_x, next_step_y, support_foot_pos.position.z],
            'trajectory': step_trajectory
        }

    def calculate_step_trajectory(self, start_pos, end_pos, step_height):
        """Calculate parabolic step trajectory"""
        # Simplified parabolic trajectory for stepping
        trajectory = []

        start_point = [start_pos.position.x, start_pos.position.y, start_pos.position.z] if start_pos else end_pos
        end_point = end_pos

        # Calculate intermediate points
        num_points = 10
        for i in range(num_points + 1):
            t = i / num_points  # Parameter from 0 to 1

            # Parabolic height profile
            height_factor = 4 * t * (1 - t)  # Parabola from 0 to 1 back to 0
            z_offset = height_factor * step_height

            # Linear interpolation for x,y
            x = start_point[0] + t * (end_point[0] - start_point[0])
            y = start_point[1] + t * (end_point[1] - start_point[1])
            z = start_point[2] + z_offset

            trajectory.append([x, y, z])

        return trajectory
```

## Humanoid-Specific Navigation Considerations

### Bipedal Locomotion Planning

```python
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class BipedalLocomotionPlanner:
    def __init__(self):
        # Bipedal-specific parameters
        self.step_length = 0.3  # meters
        self.step_width = 0.2   # meters (lateral distance between feet)
        self.step_height = 0.05 # meters (foot clearance)
        self.step_period = 1.0  # seconds per step
        self.com_height = 0.8   # Center of mass height
        self.foot_size = [0.25, 0.1]  # Length x Width of foot

        # Balance constraints
        self.max_com_deviation = 0.1  # Maximum CoM deviation from foot center
        self.stability_margin = 0.05  # Safety margin for ZMP

        # Walking pattern parameters
        self.stride_length = 0.3
        self.turning_radius = 1.0

    def plan_walk_pattern(self, path, start_orientation=0.0):
        """Plan bipedal walk pattern along a given path"""
        footsteps = []

        # Convert path to step sequence
        path_length = len(path)
        if path_length < 2:
            return footsteps

        # Calculate total path length
        total_distance = 0
        for i in range(1, path_length):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            total_distance += np.sqrt(dx*dx + dy*dy)

        # Plan footsteps based on stride length
        cumulative_distance = 0
        current_pos = np.array(path[0])
        current_orientation = start_orientation
        left_support = True  # Start with left foot support

        i = 1
        while i < path_length:
            # Calculate direction to next waypoint
            target_pos = np.array(path[i])
            direction = target_pos - current_pos
            distance_to_target = np.linalg.norm(direction)

            if distance_to_target < self.stride_length:
                # Move to next waypoint
                current_pos = target_pos
                if i < path_length - 1:
                    # Calculate orientation for next segment
                    next_direction = np.array(path[i+1]) - target_pos
                    current_orientation = np.arctan2(next_direction[1], next_direction[0])
                i += 1
            else:
                # Take a step towards the target
                step_direction = direction / distance_to_target
                step_end = current_pos + step_direction * self.stride_length

                # Determine foot placement (alternating left/right)
                foot_offset = self.calculate_foot_placement(
                    current_orientation,
                    left_support,
                    step_direction
                )

                foot_pos = step_end + foot_offset

                # Add footstep
                footsteps.append({
                    'position': foot_pos,
                    'orientation': current_orientation,
                    'foot': 'left' if left_support else 'right',
                    'time': cumulative_distance / self.stride_length * self.step_period
                })

                # Update state
                current_pos = step_end
                left_support = not left_support
                cumulative_distance += self.stride_length

        return footsteps

    def calculate_foot_placement(self, orientation, is_left_foot, step_direction):
        """Calculate appropriate foot placement considering balance"""
        # Calculate lateral offset based on support foot
        lateral_offset = self.step_width / 2
        if is_left_foot:
            # Left foot should be to the left of the CoM path
            perpendicular = np.array([-step_direction[1], step_direction[0]])
        else:
            # Right foot should be to the right of the CoM path
            perpendicular = np.array([step_direction[1], -step_direction[0]])

        foot_offset = perpendicular * lateral_offset

        # Apply orientation rotation
        cos_theta = np.cos(orientation)
        sin_theta = np.sin(orientation)
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])

        rotated_offset = rotation_matrix @ foot_offset[:2]

        return np.array([rotated_offset[0], rotated_offset[1], 0])

    def generate_com_trajectory(self, footsteps):
        """Generate Center of Mass trajectory for stable walking"""
        if not footsteps:
            return []

        com_trajectory = []

        # Use inverted pendulum model for CoM motion
        omega = np.sqrt(9.81 / self.com_height)  # Natural frequency

        for i, footstep in enumerate(footsteps):
            # Calculate CoM position relative to support foot
            # This is a simplified model - real implementation would be more complex

            # For each step, interpolate CoM position between foot placements
            if i < len(footsteps) - 1:
                next_footstep = footsteps[i + 1]

                # Generate intermediate CoM positions
                num_interpolations = 10
                for j in range(num_interpolations):
                    t = j / num_interpolations

                    # Interpolate between current and next footstep
                    current_support = footstep['position']
                    next_support = next_footstep['position']

                    # CoM follows approximately between feet
                    com_x = current_support[0] + t * (next_support[0] - current_support[0])
                    com_y = current_support[1] + t * (next_support[1] - current_support[1])

                    # Keep CoM at nominal height
                    com_z = self.com_height

                    # Add slight oscillation for natural walking
                    oscillation = 0.01 * np.sin(omega * t * self.step_period)
                    com_z += oscillation

                    com_trajectory.append([com_x, com_y, com_z])

        return com_trajectory

    def check_balance_feasibility(self, footsteps):
        """Check if the planned footsteps maintain balance"""
        if len(footsteps) < 2:
            return True

        # Check Zero Moment Point (ZMP) feasibility
        for i in range(1, len(footsteps)):
            current_step = footsteps[i-1]
            next_step = footsteps[i]

            # Calculate ZMP between steps
            zmp_x = (current_step['position'][0] + next_step['position'][0]) / 2
            zmp_y = (current_step['position'][1] + next_step['position'][1]) / 2

            # Check if ZMP is within support polygon
            # For simplicity, check if it's between the two feet
            support_polygon = self.calculate_support_polygon(current_step, next_step)

            if not self.point_in_polygon([zmp_x, zmp_y], support_polygon):
                return False, f"ZMP violation at step {i}"

        return True, "Balance feasible"

    def calculate_support_polygon(self, current_step, next_step):
        """Calculate support polygon for double support phase"""
        # Simplified rectangular support polygon
        # In reality, this would consider foot shapes and orientations

        current_foot_center = current_step['position'][:2]
        next_foot_center = next_step['position'][:2]

        # Calculate rectangle that encompasses both feet
        min_x = min(current_foot_center[0], next_foot_center[0]) - self.foot_size[0]/2
        max_x = max(current_foot_center[0], next_foot_center[0]) + self.foot_size[0]/2
        min_y = min(current_foot_center[1], next_foot_center[1]) - self.foot_size[1]/2
        max_y = max(current_foot_center[1], next_foot_center[1]) + self.foot_size[1]/2

        return [
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y]
        ]

    def point_in_polygon(self, point, polygon):
        """Check if a point is inside a polygon using ray casting"""
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def visualize_walk_pattern(self, path, footsteps, com_trajectory):
        """Visualize the planned walk pattern"""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot path
        path_array = np.array(path)
        ax.plot(path_array[:, 0], path_array[:, 1], 'b-', linewidth=2, label='Desired Path')

        # Plot footsteps
        left_footsteps = [f for f in footsteps if f['foot'] == 'left']
        right_footsteps = [f for f in footsteps if f['foot'] == 'right']

        if left_footsteps:
            left_array = np.array([f['position'] for f in left_footsteps])
            ax.scatter(left_array[:, 0], left_array[:, 1], c='red', s=100, label='Left Foot', zorder=5)

        if right_footsteps:
            right_array = np.array([f['position'] for f in right_footsteps])
            ax.scatter(right_array[:, 0], right_array[:, 1], c='blue', s=100, label='Right Foot', zorder=5)

        # Plot CoM trajectory
        if com_trajectory:
            com_array = np.array(com_trajectory)
            ax.plot(com_array[:, 0], com_array[:, 1], 'g--', linewidth=1, label='CoM Trajectory')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Bipedal Walk Pattern Planning')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        plt.tight_layout()
        plt.show()

def main():
    # Example usage of bipedal locomotion planner
    planner = BipedalLocomotionPlanner()

    # Define a simple path
    path = [
        [0, 0],
        [1, 0],
        [2, 0.5],
        [3, 1],
        [4, 1.5],
        [5, 1.5]
    ]

    # Plan footsteps
    footsteps = planner.plan_walk_pattern(path)

    # Generate CoM trajectory
    com_trajectory = planner.generate_com_trajectory(footsteps)

    # Check balance feasibility
    is_balanced, message = planner.check_balance_feasibility(footsteps)
    print(f"Balance check: {message}")

    # Visualize results
    # planner.visualize_walk_pattern(path, footsteps, com_trajectory)

    print(f"Planned {len(footsteps)} footsteps for the path")
    print(f"Generated CoM trajectory with {len(com_trajectory)} points")

if __name__ == "__main__":
    main()
```

## Integration with Perception Systems

### Sensor Integration for Navigation

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2, Image
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Bool
import numpy as np
import cv2
from cv_bridge import CvBridge

class PerceptionIntegratedNavigator(Node):
    def __init__(self):
        super().__init__('perception_integrated_navigator')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Publishers
        self.obstacle_warning_pub = self.create_publisher(Bool, '/obstacle_warning', 10)
        self.navigation_status_pub = self.create_publisher(String, '/navigation_status', 10)
        self.perception_markers_pub = self.create_publisher(MarkerArray, '/perception_markers', 10)

        # Subscribers
        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_rect_raw',
            self.depth_callback,
            10
        )
        self.rgb_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_rect_color',
            self.rgb_callback,
            10
        )
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # Perception-integrated navigation components
        self.obstacle_detector = ObstacleDetector()
        self.terrain_classifier = TerrainClassifier()
        self.dynamic_object_tracker = DynamicObjectTracker()

        # Navigation state
        self.current_pose = None
        self.current_velocity = None
        self.imu_data = None
        self.depth_image = None
        self.rgb_image = None
        self.laser_data = None

        # Navigation safety parameters
        self.safety_distance = 0.5  # meters
        self.dynamic_object_buffer = 0.8  # meters
        self.rough_terrain_penalty = 2.0  # multiplier for cost

        # Timer for perception processing
        self.perception_timer = self.create_timer(0.1, self.perception_processing_loop)

        self.get_logger().info('Perception-Integrated Navigator initialized')

    def laser_callback(self, msg):
        """Process laser scan data"""
        self.laser_data = msg
        obstacles = self.obstacle_detector.detect_from_laser(msg)
        self.update_costmap_with_obstacles(obstacles)

    def depth_callback(self, msg):
        """Process depth image data"""
        try:
            self.depth_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # Extract 3D obstacles from depth
            obstacles_3d = self.obstacle_detector.detect_from_depth(self.depth_image)
            self.update_costmap_with_3d_obstacles(obstacles_3d)

        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {e}')

    def rgb_callback(self, msg):
        """Process RGB image data for semantic understanding"""
        try:
            self.rgb_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Classify terrain and detect semantic obstacles
            terrain_classification = self.terrain_classifier.classify_from_image(self.rgb_image)
            semantic_obstacles = self.obstacle_detector.detect_semantic_obstacles(self.rgb_image)

            # Update navigation with semantic information
            self.update_navigation_with_semantics(terrain_classification, semantic_obstacles)

        except Exception as e:
            self.get_logger().error(f'Error processing RGB image: {e}')

    def imu_callback(self, msg):
        """Process IMU data for balance and orientation"""
        self.imu_data = msg

        # Check for tilt that might affect navigation
        roll, pitch, yaw = self.quaternion_to_euler(msg.orientation)

        # If robot is tilted beyond safe limits, pause navigation
        max_tilt = 0.3  # radians (~17 degrees)
        if abs(roll) > max_tilt or abs(pitch) > max_tilt:
            self.pause_navigation("Robot tilt unsafe")

    def odom_callback(self, msg):
        """Update pose and velocity for navigation"""
        self.current_pose = msg.pose.pose
        self.current_velocity = msg.twist.twist

    def perception_processing_loop(self):
        """Main loop for processing perception data and updating navigation"""
        if self.current_pose is None:
            return

        # Process all sensor data
        self.process_fusion_data()

        # Update navigation plan based on perception
        self.update_navigation_plan()

        # Check safety conditions
        self.check_navigation_safety()

    def process_fusion_data(self):
        """Fuse data from multiple sensors"""
        if (self.laser_data is not None and
            self.depth_image is not None and
            self.rgb_image is not None):

            # Fuse obstacle detections
            laser_obstacles = self.obstacle_detector.detect_from_laser(self.laser_data)
            depth_obstacles = self.obstacle_detector.detect_from_depth(self.depth_image)
            visual_obstacles = self.obstacle_detector.detect_semantic_obstacles(self.rgb_image)

            # Combine obstacle detections with confidence weighting
            fused_obstacles = self.fuse_obstacle_detections(
                laser_obstacles, depth_obstacles, visual_obstacles
            )

            # Update costmap with fused obstacles
            self.update_costmap_with_fused_obstacles(fused_obstacles)

    def fuse_obstacle_detections(self, laser_obs, depth_obs, visual_obs):
        """Fuse obstacle detections from multiple sensors"""
        fused_obstacles = []

        # Confidence weights for different sensors
        laser_weight = 0.8
        depth_weight = 0.9
        visual_weight = 0.7

        # Create obstacle dictionary with weighted confidences
        obstacle_dict = {}

        # Add laser obstacles
        for obs in laser_obs:
            pos_key = (round(obs['x'], 2), round(obs['y'], 2))
            if pos_key not in obstacle_dict:
                obstacle_dict[pos_key] = {'confidence': 0, 'type': 'unknown'}
            obstacle_dict[pos_key]['confidence'] += laser_obs['confidence'] * laser_weight
            if 'laser' not in obstacle_dict[pos_key]['type']:
                obstacle_dict[pos_key]['type'] += '_laser'

        # Add depth obstacles
        for obs in depth_obs:
            pos_key = (round(obs['x'], 2), round(obs['y'], 2))
            if pos_key not in obstacle_dict:
                obstacle_dict[pos_key] = {'confidence': 0, 'type': 'unknown'}
            obstacle_dict[pos_key]['confidence'] += obs['confidence'] * depth_weight
            if 'depth' not in obstacle_dict[pos_key]['type']:
                obstacle_dict[pos_key]['type'] += '_depth'

        # Add visual obstacles
        for obs in visual_obs:
            pos_key = (round(obs['x'], 2), round(obs['y'], 2))
            if pos_key not in obstacle_dict:
                obstacle_dict[pos_key] = {'confidence': 0, 'type': obs.get('class', 'unknown')}
            obstacle_dict[pos_key]['confidence'] += obs['confidence'] * visual_weight

        # Create fused obstacle list
        for pos_key, data in obstacle_dict.items():
            if data['confidence'] > 0.5:  # Threshold for valid obstacle
                fused_obstacles.append({
                    'x': pos_key[0],
                    'y': pos_key[1],
                    'confidence': data['confidence'],
                    'type': data['type']
                })

        return fused_obstacles

    def update_costmap_with_fused_obstacles(self, obstacles):
        """Update costmap with fused obstacle data"""
        # This would update the Nav2 costmap with obstacle information
        # In practice, this integrates with the Nav2 costmap server

        # For now, just log the obstacles
        for obs in obstacles:
            self.get_logger().debug(f"Fused obstacle: {obs}")

    def update_navigation_with_semantics(self, terrain_class, semantic_obstacles):
        """Update navigation based on semantic understanding"""
        # Adjust navigation parameters based on terrain classification
        if terrain_class['roughness'] > 0.7:  # Very rough terrain
            # Reduce speed and increase safety margins
            self.reduce_navigation_speed(0.5)
            self.increase_safety_margin(0.3)
        elif terrain_class['slope'] > 0.3:  # Steep slope
            # Use different navigation strategy for slopes
            self.adapt_to_slope_navigation(terrain_class['slope'])

        # Handle semantic obstacles differently
        for obs in semantic_obstacles:
            if obs['class'] == 'person':
                # Maintain larger safety buffer for people
                self.increase_dynamic_buffer(1.0)
            elif obs['class'] == 'vehicle':
                # Be especially cautious around vehicles
                self.activate_caution_mode()
            elif obs['class'] == 'fragile_object':
                # Navigate carefully around fragile objects
                self.enable_delicate_navigation()

    def check_navigation_safety(self):
        """Check if navigation is currently safe"""
        safety_issues = []

        # Check for immediate obstacles
        if self.laser_data:
            min_range = min(self.laser_data.ranges) if self.laser_data.ranges else float('inf')
            if min_range < self.safety_distance:
                safety_issues.append(f"Obstacle too close: {min_range:.2f}m < {self.safety_distance:.2f}m")

        # Check for dynamic obstacles
        if self.dynamic_object_tracker.has_objects_approaching():
            safety_issues.append("Approaching dynamic object detected")

        # Check IMU for unsafe tilt
        if (self.imu_data and
            (abs(self.quaternion_to_euler(self.imu_data.orientation)[0]) > 0.4 or
             abs(self.quaternion_to_euler(self.imu_data.orientation)[1]) > 0.4)):
            safety_issues.append("Unsafe robot orientation detected")

        # Publish safety status
        if safety_issues:
            self.get_logger().warn(f"Navigation safety issues: {', '.join(safety_issues)}")
            self.publish_safety_warning(safety_issues)
        else:
            self.get_logger().info("Navigation is safe")

    def publish_safety_warning(self, issues):
        """Publish safety warnings"""
        warning_msg = Bool()
        warning_msg.data = True
        self.obstacle_warning_pub.publish(warning_msg)

        status_msg = String()
        status_msg.data = f"Safety issues: {', '.join(issues)}"
        self.navigation_status_pub.publish(status_msg)

    def quaternion_to_euler(self, quat):
        """Convert quaternion to Euler angles"""
        import math
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        sinr_cosp = 2 * (quat.w * quat.x + quat.y * quat.z)
        cosr_cosp = 1 - 2 * (quat.x * quat.x + quat.y * quat.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (quat.w * quat.y - quat.z * quat.x)
        pitch = math.asin(sinp)

        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def reduce_navigation_speed(self, factor):
        """Reduce navigation speed by factor"""
        # This would integrate with the Nav2 controller to reduce speeds
        pass

    def increase_safety_margin(self, additional_margin):
        """Increase safety margin by additional amount"""
        self.safety_distance += additional_margin

    def adapt_to_slope_navigation(self, slope_angle):
        """Adapt navigation for sloped terrain"""
        # Adjust footstep planning for slopes
        # Modify balance control parameters
        pass

    def increase_dynamic_buffer(self, additional_buffer):
        """Increase buffer for dynamic objects"""
        self.dynamic_object_buffer += additional_buffer

    def activate_caution_mode(self):
        """Activate cautious navigation mode"""
        self.reduce_navigation_speed(0.3)
        self.increase_safety_margin(0.5)

    def enable_delicate_navigation(self):
        """Enable delicate navigation around fragile objects"""
        self.reduce_navigation_speed(0.2)
        self.increase_safety_margin(0.8)

class ObstacleDetector:
    def __init__(self):
        # Initialize obstacle detection models
        pass

    def detect_from_laser(self, laser_msg):
        """Detect obstacles from laser scan"""
        obstacles = []

        for i, range_val in enumerate(laser_msg.ranges):
            if 0 < range_val < 3.0:  # Valid range measurement
                angle = laser_msg.angle_min + i * laser_msg.angle_increment
                x = range_val * np.cos(angle)
                y = range_val * np.sin(angle)

                obstacles.append({
                    'x': x,
                    'y': y,
                    'range': range_val,
                    'angle': angle,
                    'confidence': 0.9  # High confidence for laser
                })

        return obstacles

    def detect_from_depth(self, depth_image):
        """Detect 3D obstacles from depth image"""
        obstacles = []

        # Simple threshold-based obstacle detection
        valid_depths = depth_image[depth_image > 0]
        if len(valid_depths) > 0:
            min_depth = np.min(valid_depths)
            if min_depth < 2.0:  # Something is close
                # Convert pixel coordinates to world coordinates
                # This is simplified - real implementation would use camera parameters
                height, width = depth_image.shape
                for u in range(0, width, 10):  # Sample every 10 pixels
                    for v in range(0, height, 10):
                        depth_val = depth_image[v, u]
                        if 0 < depth_val < 2.0:
                            # Convert to world coordinates (simplified)
                            x = (u - width/2) * depth_val * 0.001  # Rough conversion
                            y = (v - height/2) * depth_val * 0.001

                            obstacles.append({
                                'x': x,
                                'y': y,
                                'z': depth_val,
                                'confidence': 0.7
                            })

        return obstacles

    def detect_semantic_obstacles(self, rgb_image):
        """Detect semantic obstacles from RGB image"""
        # This would use a trained CNN for semantic segmentation
        # For now, return empty list
        return []

class TerrainClassifier:
    def __init__(self):
        # Initialize terrain classification models
        pass

    def classify_from_image(self, rgb_image):
        """Classify terrain type from RGB image"""
        # Analyze image texture, color, and structure
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

        # Calculate texture metrics
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Estimate roughness (higher variance = rougher texture)
        roughness = min(1.0, laplacian_var / 1000.0)

        # Estimate slope from apparent texture compression
        # This is a very simplified approach
        slope = 0.1  # Default assumption

        return {
            'roughness': roughness,
            'slope': slope,
            'traversability': 1.0 - roughness  # Simplified traversability
        }

class DynamicObjectTracker:
    def __init__(self):
        # Initialize tracking algorithms
        self.tracked_objects = []
        self.motion_threshold = 0.1  # m/s

    def has_objects_approaching(self):
        """Check if any tracked objects are approaching the robot"""
        # Check if any tracked object's velocity vector points toward robot
        # This is a simplified implementation
        return False
```

## Performance Optimization

### Navigation Performance Tuning

```python
import psutil
import time
from collections import deque
import threading

class Nav2PerformanceOptimizer:
    def __init__(self):
        self.cpu_usage_history = deque(maxlen=100)
        self.memory_usage_history = deque(maxlen=100)
        self.nav_frequency_history = deque(maxlen=100)
        self.path_computation_times = deque(maxlen=50)

        self.target_frequency = 20.0  # Hz
        self.max_cpu_percent = 80.0   # Percent
        self.adaptive_params = {
            'planner_frequency': 20.0,
            'controller_frequency': 20.0,
            'costmap_resolution': 0.05,
            'max_planning_time': 5.0
        }

        self.performance_lock = threading.Lock()
        self.monitoring_active = True

        # Start performance monitoring thread
        self.monitor_thread = threading.Thread(target=self.performance_monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def performance_monitor_loop(self):
        """Continuous performance monitoring"""
        while self.monitoring_active:
            # Monitor system resources
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent

            with self.performance_lock:
                self.cpu_usage_history.append(cpu_percent)
                self.memory_usage_history.append(memory_percent)

            # Adjust navigation parameters based on performance
            self.adjust_parameters_for_performance()

            time.sleep(0.5)  # Monitor every 0.5 seconds

    def adjust_parameters_for_performance(self):
        """Dynamically adjust navigation parameters based on performance"""
        with self.performance_lock:
            if not self.cpu_usage_history:
                return

            avg_cpu = sum(self.cpu_usage_history) / len(self.cpu_usage_history)
            avg_memory = sum(self.memory_usage_history) / len(self.memory_usage_history)

        # If CPU usage is high, reduce computation intensity
        if avg_cpu > self.max_cpu_percent:
            self.reduce_computational_load()
        elif avg_cpu < self.max_cpu_percent * 0.6:
            # If CPU usage is low, we can afford more computation
            self.increase_computational_load()

    def reduce_computational_load(self):
        """Reduce computational load to improve performance"""
        # Reduce planner frequency
        self.adaptive_params['planner_frequency'] = max(5.0,
            self.adaptive_params['planner_frequency'] * 0.8)

        # Increase costmap resolution (larger cells = less computation)
        self.adaptive_params['costmap_resolution'] = min(0.2,
            self.adaptive_params['costmap_resolution'] * 1.2)

        # Reduce maximum planning time
        self.adaptive_params['max_planning_time'] = max(1.0,
            self.adaptive_params['max_planning_time'] * 0.8)

        print(f"Reduced computational load: freq={self.adaptive_params['planner_frequency']:.1f}Hz, "
              f"res={self.adaptive_params['costmap_resolution']:.2f}m")

    def increase_computational_load(self):
        """Increase computational load for better performance"""
        # Increase planner frequency
        self.adaptive_params['planner_frequency'] = min(30.0,
            self.adaptive_params['planner_frequency'] * 1.1)

        # Decrease costmap resolution (smaller cells = more precision)
        self.adaptive_params['costmap_resolution'] = max(0.02,
            self.adaptive_params['costmap_resolution'] * 0.9)

        # Increase maximum planning time
        self.adaptive_params['max_planning_time'] = min(10.0,
            self.adaptive_params['max_planning_time'] * 1.1)

        print(f"Increased computational load: freq={self.adaptive_params['planner_frequency']:.1f}Hz, "
              f"res={self.adaptive_params['costmap_resolution']:.2f}m")

    def get_performance_metrics(self):
        """Get current performance metrics"""
        with self.performance_lock:
            if not self.cpu_usage_history:
                return {}

            metrics = {
                'cpu_percent': sum(self.cpu_usage_history) / len(self.cpu_usage_history),
                'memory_percent': sum(self.memory_usage_history) / len(self.memory_usage_history),
                'current_params': self.adaptive_params.copy()
            }

        if self.nav_frequency_history:
            metrics['actual_frequency'] = sum(self.nav_frequency_history) / len(self.nav_frequency_history)

        if self.path_computation_times:
            metrics['avg_path_time'] = sum(self.path_computation_times) / len(self.path_computation_times)
            metrics['max_path_time'] = max(self.path_computation_times)

        return metrics

    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
```

## Troubleshooting and Best Practices

### Common Navigation Issues and Solutions

```python
class Nav2Troubleshooter:
    def __init__(self):
        self.known_issues = {
            'oscillation': {
                'symptoms': ['robot moves back and forth', 'cannot reach goal'],
                'causes': ['narrow passages', 'incorrect costmap inflation'],
                'solutions': [
                    'increase inflation radius',
                    'adjust controller parameters',
                    'modify global planner tolerance'
                ]
            },
            'getting_stuck': {
                'symptoms': ['robot stops moving', 'high cmd_vel commands'],
                'causes': ['local minima', 'incorrect localization'],
                'solutions': [
                    'enable recovery behaviors',
                    'improve sensor coverage',
                    'adjust costmap obstacles'
                ]
            },
            'planning_failure': {
                'symptoms': ['no path found', 'global planner fails'],
                'causes': ['map issues', 'incorrect frame transforms'],
                'solutions': [
                    'verify map quality',
                    'check tf tree',
                    'adjust planner parameters'
                ]
            }
        }

    def diagnose_issue(self, symptoms):
        """Diagnose navigation issue based on symptoms"""
        possible_issues = []

        for issue_name, issue_info in self.known_issues.items():
            symptom_match = 0
            for symptom in symptoms:
                for known_symptom in issue_info['symptoms']:
                    if symptom.lower() in known_symptom.lower():
                        symptom_match += 1

            if symptom_match > 0:
                possible_issues.append({
                    'issue': issue_name,
                    'confidence': symptom_match / len(issue_info['symptoms']),
                    'solutions': issue_info['solutions']
                })

        return possible_issues

    def get_configuration_checklist(self):
        """Get checklist for Nav2 configuration verification"""
        checklist = [
            "Verify tf tree is complete and transforms are broadcasting",
            "Check that costmaps are updating and not static",
            "Ensure robot radius/footprint is correctly configured",
            "Verify sensor data is being received and processed",
            "Confirm global and local planners are loaded correctly",
            "Check that controllers are properly configured",
            "Validate behavior tree is executing as expected",
            "Ensure proper frame IDs are used throughout",
            "Test localization system independently",
            "Verify map is properly loaded and aligned"
        ]
        return checklist

    def performance_tuning_guide(self):
        """Provide performance tuning recommendations"""
        guide = {
            'frequency_tuning': {
                'global_planner': '5-10 Hz is typically sufficient',
                'local_planner': '20-50 Hz for responsive control',
                'costmap_updates': '5-10 Hz for local, 1 Hz for global'
            },
            'memory_optimization': {
                'costmap_sizes': 'Keep local costmap small (3-5m)',
                'resolution_tradeoffs': '0.025-0.05m for local, 0.1-0.2m for global',
                'layer_management': 'Remove unused costmap layers'
            },
            'computation_balancing': {
                'planning_vs_control': 'Separate heavy planning from control loops',
                'multi_threading': 'Use separate threads for perception and navigation',
                'load_distribution': 'Distribute computations across available cores'
            }
        }
        return guide
```

## Learning Objectives Review

- Understand Nav2 architecture and components ✓
- Configure Nav2 for humanoid robotics applications ✓
- Implement path planning algorithms for bipedal locomotion ✓
- Integrate perception systems with navigation ✓
- Optimize navigation for dynamic environments ✓
- Troubleshoot common navigation issues ✓

## Practical Exercise

1. Install and configure Nav2 for your robot platform
2. Set up the necessary configuration files and launch files
3. Implement a basic navigation system with path planning
4. Integrate perception data (laser, camera, IMU) for enhanced navigation
5. Test navigation in both simulation and real environments
6. Fine-tune parameters for optimal performance

## Assessment Questions

1. Explain the key differences between Nav2 and the legacy ROS navigation stack.
2. What are the main components of the Nav2 architecture?
3. How does the Behavior Tree system improve navigation flexibility?
4. What are the special considerations for bipedal robot navigation compared to wheeled robots?

## Further Reading

- Nav2 Documentation: https://navigation.ros.org/
- "The Navigation Stack for ROS" by Eitan Marder-Eppstein
- "Path Planning in Complex Environments" by Choset et al.
- Behavior Trees in Robotics: https://arxiv.org/abs/1709.00084

## Next Steps

Continue to [Sim-to-Real Transfer](./sim-to-real.md) to learn about transferring AI models from simulation to real robots.