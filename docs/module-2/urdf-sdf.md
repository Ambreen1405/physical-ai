---
id: urdf-sdf
title: URDF and SDF
sidebar_position: 4
description: Comprehensive guide to Unified Robot Description Format (URDF) and Simulation Description Format (SDF) for robotics simulation
keywords: [urdf, sdf, robot description, simulation description, robot modeling, robotics formats]
---

# URDF and SDF

This chapter covers the two primary formats used for describing robots and simulation environments: URDF (Unified Robot Description Format) for robot modeling and SDF (Simulation Description Format) for simulation environments. Understanding both formats is crucial for effective robotics simulation.

## Learning Objectives

- Understand the differences and use cases for URDF and SDF
- Create complex robot models using URDF
- Design simulation environments using SDF
- Convert between URDF and SDF formats
- Apply best practices for both formats

## URDF vs SDF: Key Differences

### URDF (Unified Robot Description Format)
- **Purpose**: Describes robot structure and kinematics
- **Scope**: Single robot or manipulator
- **Format**: XML-based
- **Focus**: Links, joints, kinematic chains
- **Usage**: Robot modeling, ROS integration

### SDF (Simulation Description Format)
- **Purpose**: Describes complete simulation environments
- **Scope**: Worlds, robots, objects, physics
- **Format**: XML-based
- **Focus**: Simulation entities, physics properties
- **Usage**: Gazebo simulation, complete environments

## URDF Fundamentals

### URDF Structure

```xml
<?xml version="1.0"?>
<robot name="my_robot">
  <!-- Links define rigid bodies -->
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </collision>
  </link>

  <!-- Joints connect links -->
  <joint name="base_to_wheel" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_link"/>
    <origin xyz="0 0.25 -0.1"/>
    <axis xyz="0 1 0"/>
  </joint>

  <link name="wheel_link">
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
  </link>
</robot>
```

### Link Components

#### Inertial Properties
The inertial properties are crucial for physics simulation:

```xml
<inertial>
  <!-- Mass in kilograms -->
  <mass value="1.0"/>

  <!-- Origin of the inertial reference frame relative to the link frame -->
  <origin xyz="0 0 0" rpy="0 0 0"/>

  <!-- Inertia matrix (symmetric, only 6 values needed) -->
  <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
</inertial>
```

For common shapes, here are the inertia formulas:
- **Box**: `ixx = m*(h² + d²)/12`, `iyy = m*(w² + d²)/12`, `izz = m*(w² + h²)/12`
- **Cylinder**: `ixx = iyy = m*(3*r² + h²)/12`, `izz = m*r²/2`
- **Sphere**: `ixx = iyy = izz = 2*m*r²/5`

#### Visual Properties
Visual properties define how the robot appears:

```xml
<visual>
  <!-- Origin relative to link frame -->
  <origin xyz="0 0 0" rpy="0 0 0"/>

  <!-- Geometry definition -->
  <geometry>
    <box size="0.5 0.5 0.2"/>  <!-- Box -->
    <!-- OR -->
    <cylinder radius="0.1" length="0.2"/>  <!-- Cylinder -->
    <!-- OR -->
    <sphere radius="0.1"/>  <!-- Sphere -->
    <!-- OR -->
    <mesh filename="package://my_robot/meshes/link.stl" scale="1 1 1"/>  <!-- Mesh -->
  </geometry>

  <!-- Material properties -->
  <material name="red">
    <color rgba="1 0 0 1"/>
    <!-- OR -->
    <texture filename="package://my_robot/materials/textures/red.png"/>
  </material>
</visual>
```

#### Collision Properties
Collision properties define physics interactions:

```xml
<collision>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry>
    <box size="0.5 0.5 0.2"/>
    <!-- OR other geometry types -->
  </geometry>
</collision>
```

### Joint Types in URDF

#### Revolute Joint
```xml
<joint name="revolute_joint" type="revolute">
  <parent link="link1"/>
  <child link="link2"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>
  <dynamics damping="0.1" friction="0.0"/>
</joint>
```

#### Continuous Joint
```xml
<joint name="continuous_joint" type="continuous">
  <parent link="link1"/>
  <child link="link2"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <dynamics damping="0.1"/>
</joint>
```

#### Prismatic Joint
```xml
<joint name="prismatic_joint" type="prismatic">
  <parent link="link1"/>
  <child link="link2"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>
  <limit lower="0" upper="0.5" effort="100" velocity="1"/>
</joint>
```

## SDF Fundamentals

### SDF Structure

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="my_world">
    <!-- Physics engine configuration -->
    <physics name="default_physics" type="ode">
      <gravity>0 0 -9.8</gravity>
      <max_step_size>0.001</max_step_size>
    </physics>

    <!-- Models in the world -->
    <model name="ground_plane" placement_frame="ground_plane::link">
      <include>
        <uri>model://ground_plane</uri>
      </include>
    </model>

    <model name="my_robot">
      <!-- Robot model definition (can include URDF) -->
      <pose>0 0 0.5 0 0 0</pose>
      <link name="chassis">
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.01</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.01</iyy>
            <iyz>0</iyz>
            <izz>0.01</izz>
          </inertia>
        </inertial>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.5 0.5 0.2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.5 0.5 0.2</size>
            </box>
          </geometry>
        </visual>
      </link>
    </model>

    <!-- Lights -->
    <light name="sun" type="directional">
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.4 0.2 -0.9</direction>
    </light>

    <!-- Plugins -->
    <plugin name="world_plugin" filename="libworld_plugin.so">
      <update_rate>1.0</update_rate>
    </plugin>
  </world>
</sdf>
```

### SDF Model Definition

```xml
<model name="my_robot" static="false">
  <!-- Model pose in the world -->
  <pose>0 0 0.5 0 0 0</pose>

  <!-- Self-collide to prevent self-intersection -->
  <self_collide>false</self_collide>

  <!-- Enable wind effects -->
  <enable_wind>false</enable_wind>

  <!-- Link definitions -->
  <link name="chassis">
    <pose>0 0 0 0 0 0</pose>
    <inertial>
      <mass>1.0</mass>
      <inertia>
        <ixx>0.01</ixx>
        <ixy>0</ixy>
        <ixz>0</ixz>
        <iyy>0.01</iyy>
        <iyz>0</iyz>
        <izz>0.01</izz>
      </inertia>
    </inertial>

    <!-- Multiple collisions per link -->
    <collision name="collision_1">
      <geometry>
        <box><size>0.5 0.5 0.2</size></box>
      </geometry>
    </collision>

    <!-- Multiple visuals per link -->
    <visual name="visual_1">
      <geometry>
        <box><size>0.5 0.5 0.2</size></box>
      </geometry>
    </visual>

    <!-- Sensors -->
    <sensor name="camera" type="camera">
      <pose>0.1 0 0.1 0 0 0</pose>
      <camera>
        <horizontal_fov>1.047</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
        </image>
        <clip>
          <near>0.1</near>
          <far>10</far>
        </clip>
      </camera>
    </sensor>
  </link>

  <!-- Joints -->
  <joint name="wheel_joint" type="continuous">
    <parent>chassis</parent>
    <child>wheel</child>
    <axis>
      <xyz>0 1 0</xyz>
    </axis>
  </joint>
</model>
```

## Converting Between URDF and SDF

### Using xacro to Generate SDF

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="robot_sdf">
  <xacro:property name="M_PI" value="3.1415926535897931" />

  <!-- Include URDF content -->
  <xacro:include filename="$(find my_robot_description)/urdf/robot.urdf.xacro" />

  <!-- Wrap in SDF tags -->
  <gazebo>
    <plugin name="ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/my_robot</robotNamespace>
      <robotSimType>gazebo_ros_control/DefaultRobotHWSim</robotSimType>
    </plugin>
  </gazebo>
</robot>
```

### Using Command Line Tools

```bash
# Convert URDF to SDF
gz sdf -p robot.urdf > robot.sdf

# Or using ROS tools
# rosrun xacro xacro --inorder robot.urdf.xacro > robot.urdf
# gz sdf -p robot.urdf > robot.sdf
```

## Advanced URDF Features

### Transmission Elements
For ROS control integration:

```xml
<transmission name="wheel_trans">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="wheel_joint">
    <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
  </joint>
  <actuator name="wheel_motor">
    <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

### Gazebo-Specific Tags in URDF
```xml
<gazebo reference="wheel_link">
  <mu1>0.5</mu1>
  <mu2>0.5</mu2>
  <kp>1000000.0</kp>
  <kd>100.0</kd>
  <material>Gazebo/Blue</material>
</gazebo>

<gazebo>
  <plugin name="differential_drive" filename="libgazebo_ros_diff_drive.so">
    <left_joint>left_wheel_joint</left_joint>
    <right_joint>right_wheel_joint</right_joint>
    <wheel_separation>0.3</wheel_separation>
    <wheel_diameter>0.15</wheel_diameter>
  </plugin>
</gazebo>
```

## Advanced SDF Features

### Plugins
SDF supports various plugins for extended functionality:

```xml
<!-- Physics plugin -->
<plugin name="physics_plugin" filename="libphysics_plugin.so">
  <param1>value1</param1>
</plugin>

<!-- Model plugin -->
<plugin name="model_plugin" filename="libmodel_plugin.so">
  <update_rate>100</update_rate>
</plugin>

<!-- Sensor plugin -->
<plugin name="sensor_plugin" filename="libsensor_plugin.so">
  <topic_name>/sensor_data</topic_name>
</plugin>
```

### Nested Models
SDF supports nested models for complex assemblies:

```xml
<model name="assembly">
  <model name="sub_model_1">
    <!-- Sub-model definition -->
  </model>
  <model name="sub_model_2">
    <!-- Another sub-model -->
  </model>
</model>
```

## Best Practices

### URDF Best Practices

#### 1. Proper Mass and Inertia
```xml
<!-- GOOD: Calculate based on actual geometry -->
<inertial>
  <mass value="0.5"/>
  <origin xyz="0 0 0"/>
  <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
</inertial>

<!-- AVOID: Placeholder values -->
<inertial>
  <mass value="0.001"/>  <!-- Too light -->
  <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
</inertial>
```

#### 2. Joint Limits
```xml
<!-- GOOD: Realistic limits based on hardware -->
<limit lower="-2.0" upper="2.0" effort="100" velocity="2"/>

<!-- AVOID: Unlimited or unrealistic limits -->
<limit lower="-1000" upper="1000" effort="1000" velocity="1000"/>
```

#### 3. Proper Naming Conventions
```xml
<!-- GOOD: Descriptive names -->
<link name="left_wheel_link"/>
<joint name="left_wheel_rotation_joint"/>

<!-- AVOID: Generic names -->
<link name="link1"/>
<joint name="joint1"/>
```

### SDF Best Practices

#### 1. Physics Configuration
```xml
<!-- Good configuration -->
<physics name="default_physics" type="ode">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>
  <ode>
    <solver>
      <type>quick</type>
      <iters>10</iters>
      <sor>1.3</sor>
    </solver>
  </ode>
</physics>
```

#### 2. World Organization
```xml
<!-- Organize world content logically -->
<world name="my_world">
  <!-- Physics first -->
  <physics name="default_physics" type="ode">
    <!-- ... -->
  </physics>

  <!-- Models next -->
  <model name="ground_plane">
    <!-- ... -->
  </model>

  <!-- Lights and plugins last -->
  <light name="sun">
    <!-- ... -->
  </light>
</world>
```

## Troubleshooting Common Issues

### URDF Issues

#### 1. Invalid URDF
```bash
# Check URDF validity
check_urdf my_robot.urdf

# Visualize URDF structure
urdf_to_graphiz my_robot.urdf
```

#### 2. Mass/Inertia Issues
- Objects falling through surfaces: Check mass values
- Unstable joints: Verify inertia tensors
- Incorrect dynamics: Validate center of mass

#### 3. Joint Issues
- Missing joint limits: Add realistic constraints
- Wrong joint axes: Verify coordinate frames
- Uncontrolled motion: Check transmission setup

### SDF Issues

#### 1. Physics Instability
- Reduce time step (`max_step_size`)
- Adjust solver parameters (iterations, SOR)
- Verify mass and inertia properties

#### 2. Model Loading Errors
- Check SDF version compatibility
- Verify file paths and URIs
- Validate XML syntax

## Integration with ROS

### URDF in ROS Launch Files
```xml
<!-- In launch file -->
<node name="robot_state_publisher" pkg="robot_state_publisher" type="robot_state_publisher">
  <param name="robot_description" command="$(find xacro)/xacro --inorder '$(find my_robot_description)/urdf/robot.urdf.xacro'" />
</node>
```

### Gazebo Integration
```xml
<!-- In launch file -->
<include file="$(find gazebo_ros)/launch/empty_world.launch">
  <arg name="world_name" value="$(find my_robot_gazebo)/worlds/my_world.sdf"/>
  <arg name="paused" value="false"/>
  <arg name="use_sim_time" value="true"/>
  <arg name="gui" value="true"/>
  <arg name="headless" value="false"/>
  <arg name="debug" value="false"/>
</include>

<!-- Spawn robot in Gazebo -->
<node name="spawn_urdf" pkg="gazebo_ros" type="spawn_model"
      args="-param robot_description -urdf -model my_robot -x 0 -y 0 -z 0.5" />
```

## Learning Objectives Review

- Understand the differences and use cases for URDF and SDF ✓
- Create complex robot models using URDF ✓
- Design simulation environments using SDF ✓
- Convert between URDF and SDF formats ✓
- Apply best practices for both formats ✓

## Practical Exercise

1. Create a simple robot URDF with at least 3 links and 2 joints
2. Add proper inertial, visual, and collision properties
3. Convert the URDF to SDF format
4. Create a simple world SDF file
5. Load both in Gazebo and verify proper simulation

## Assessment Questions

1. What are the main differences between URDF and SDF?
2. Explain the importance of proper mass and inertia values in URDF.
3. How do you integrate URDF robots with Gazebo simulation?
4. What are the best practices for joint limit configuration?

## Further Reading

- URDF Documentation: http://wiki.ros.org/urdf
- SDF Documentation: http://sdformat.org/
- ROS URDF Tutorials: http://wiki.ros.org/urdf/Tutorials
- Gazebo Model Tutorial: http://gazebosim.org/tutorials?tut=models

## Next Steps

Continue to [Unity Rendering](./unity-rendering.md) to learn about high-fidelity rendering using Unity for robotics applications.