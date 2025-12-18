---
id: urdf-humanoids
title: URDF for Humanoids
sidebar_position: 5
description: Comprehensive guide to Unified Robot Description Format for humanoid robots
keywords: [urdf, robot description, humanoid robots, robotics modeling, xacro]
---

# URDF for Humanoids

Unified Robot Description Format (URDF) is the standard XML-based format for representing robot models in ROS. This chapter focuses specifically on using URDF for humanoid robot modeling, which presents unique challenges and opportunities.

## Learning Objectives

- Understand the structure and components of URDF files
- Create detailed humanoid robot models using URDF
- Apply Xacro macros to simplify complex humanoid models
- Integrate visual, collision, and inertial properties for humanoids
- Validate and debug URDF models for humanoid robots

## URDF Fundamentals

### What is URDF?

URDF (Unified Robot Description Format) is an XML format for representing a robot model. It defines the physical and visual properties of a robot, including:
- Links (rigid bodies)
- Joints (connections between links)
- Visual and collision properties
- Inertial properties
- Materials and colors

### Basic URDF Structure

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Links define rigid bodies -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Joints connect links -->
  <joint name="base_to_head" type="fixed">
    <parent link="base_link"/>
    <child link="head_link"/>
    <origin xyz="0 0 0.3"/>
  </joint>

  <link name="head_link">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
</robot>
```

## Humanoid-Specific URDF Considerations

### Anthropomorphic Design Principles

Humanoid robots require special attention to anthropomorphic design:

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <!-- Main body structure -->
  <link name="base_link">
    <inertial>
      <mass value="10"/>
      <origin xyz="0 0 0.5"/>
      <inertia ixx="1.0" ixy="0" ixz="0" iyy="1.0" iyz="0" izz="1.0"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.5"/>
      <geometry>
        <box size="0.2 0.2 1.0"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.5"/>
      <geometry>
        <box size="0.2 0.2 1.0"/>
      </geometry>
    </collision>
  </link>

  <!-- Humanoid head -->
  <link name="head">
    <inertial>
      <mass value="2"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/>
    </inertial>
    <visual>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
      <material name="skin">
        <color rgba="0.9 0.7 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
    </collision>
  </link>

  <joint name="neck_joint" type="revolute">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 1.0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <!-- Humanoid arms -->
  <link name="left_upper_arm">
    <inertial>
      <mass value="1.5"/>
      <origin xyz="0 0 -0.15"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.005"/>
    </inertial>
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <origin rpy="1.57 0 0"/>
      <material name="arm_color">
        <color rgba="0.6 0.6 0.6 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <origin rpy="1.57 0 0"/>
    </collision>
  </link>

  <joint name="left_shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.15 0 0.8"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.0" upper="2.0" effort="50" velocity="2"/>
  </joint>
</robot>
```

## Joint Types for Humanoid Robots

### Revolute Joints (Rotational)

For human-like joints with rotational movement:

```xml
<joint name="elbow_joint" type="revolute">
  <parent link="upper_arm"/>
  <child link="lower_arm"/>
  <origin xyz="0 0 -0.3"/>
  <axis xyz="0 1 0"/>
  <limit lower="-2.5" upper="0.5" effort="30" velocity="3"/>
  <dynamics damping="0.1" friction="0.0"/>
</joint>
```

### Continuous Joints

For joints that can rotate continuously (like waist):

```xml
<joint name="waist_joint" type="continuous">
  <parent link="torso"/>
  <child link="pelvis"/>
  <origin xyz="0 0 -0.5"/>
  <axis xyz="0 0 1"/>
  <dynamics damping="0.2"/>
</joint>
```

### Fixed Joints

For non-moving connections:

```xml
<joint name="sensor_mount" type="fixed">
  <parent link="head"/>
  <child link="camera_link"/>
  <origin xyz="0.05 0 0.05" rpy="0 0 0"/>
</joint>
```

## Visual and Collision Properties

### Visual Properties for Realism

```xml
<link name="face_panel">
  <visual>
    <origin xyz="0.05 0 0"/>
    <geometry>
      <mesh filename="package://humanoid_description/meshes/face.dae" scale="1 1 1"/>
    </geometry>
    <material name="face_material">
      <color rgba="0.9 0.9 0.9 1"/>
      <texture filename="package://humanoid_description/textures/face.png"/>
    </material>
  </visual>
  <collision>
    <origin xyz="0.05 0 0"/>
    <geometry>
      <mesh filename="package://humanoid_description/meshes/face_collision.stl"/>
    </geometry>
  </collision>
</link>
```

### Collision Properties for Physics

```xml
<link name="foot_link">
  <collision>
    <!-- Simplified collision geometry for better physics performance -->
    <geometry>
      <box size="0.25 0.1 0.05"/>
    </geometry>
    <origin xyz="0 0 -0.025"/>
  </collision>
  <!-- Multiple collision elements for complex shapes -->
  <collision>
    <geometry>
      <sphere radius="0.05"/>
    </geometry>
    <origin xyz="0.1 0 0"/>
  </collision>
</link>
```

## Inertial Properties

### Calculating Inertial Properties

```xml
<link name="thigh">
  <inertial>
    <!-- Mass based on volume and material density -->
    <mass value="3.0"/>
    <!-- Origin at center of mass -->
    <origin xyz="0 0 -0.15"/>
    <!-- Inertia tensor for cylinder-like shape -->
    <inertia
      ixx="0.05"
      ixy="0"
      ixz="0"
      iyy="0.05"
      iyz="0"
      izz="0.01"/>
  </inertial>
</link>
```

## Xacro for Complex Humanoid Models

### Introduction to Xacro

Xacro (XML Macros) allows parameterization and reusability in URDF:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_xacro">

  <!-- Define properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="arm_length" value="0.3" />
  <xacro:property name="arm_radius" value="0.05" />
  <xacro:property name="arm_mass" value="1.5" />

  <!-- Macro for arm segments -->
  <xacro:macro name="arm_segment" params="name parent xyz axis *origin">
    <link name="${name}_link">
      <inertial>
        <mass value="${arm_mass}"/>
        <origin xyz="0 0 -${arm_length/2}"/>
        <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.005"/>
      </inertial>
      <visual>
        <xacro:insert_block name="origin"/>
        <geometry>
          <cylinder length="${arm_length}" radius="${arm_radius}"/>
        </geometry>
        <material name="arm_material">
          <color rgba="0.6 0.6 0.6 1"/>
        </material>
      </visual>
      <collision>
        <xacro:insert_block name="origin"/>
        <geometry>
          <cylinder length="${arm_length}" radius="${arm_radius}"/>
        </geometry>
      </collision>
    </link>

    <joint name="${name}_joint" type="revolute">
      <parent link="${parent}"/>
      <child link="${name}_link"/>
      <origin xyz="${xyz}"/>
      <axis xyz="${axis}"/>
      <limit lower="-2.0" upper="2.0" effort="50" velocity="2"/>
    </joint>
  </xacro:macro>

  <!-- Use the macro to create arms -->
  <xacro:arm_segment name="left_upper_arm" parent="torso" xyz="0.15 0 0" axis="0 0 1">
    <origin rpy="${M_PI/2} 0 0"/>
  </xacro:arm_segment>

  <xacro:arm_segment name="right_upper_arm" parent="torso" xyz="-0.15 0 0" axis="0 0 1">
    <origin rpy="${M_PI/2} 0 0"/>
  </xacro:arm_segment>
</robot>
```

### Complete Humanoid Xacro Example

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="advanced_humanoid">

  <!-- Include other xacro files -->
  <xacro:include filename="$(find humanoid_description)/urdf/materials.xacro" />
  <xacro:include filename="$(find humanoid_description)/urdf/transmission.xacro" />

  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="torso_height" value="0.5" />
  <xacro:property name="torso_width" value="0.2" />
  <xacro:property name="torso_depth" value="0.15" />

  <!-- Materials -->
  <xacro:include filename="$(find humanoid_description)/urdf/materials.xacro" />

  <!-- Macro for humanoid limb -->
  <xacro:macro name="humanoid_limb" params="side type length radius mass xyz axis limits_origin">
    <link name="${side}_${type}_segment">
      <inertial>
        <mass value="${mass}"/>
        <origin xyz="0 0 -${length/2}"/>
        <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.005"/>
      </inertial>
      <visual>
        <origin rpy="${M_PI/2} 0 0"/>
        <geometry>
          <cylinder length="${length}" radius="${radius}"/>
        </geometry>
        <material name="limb_material"/>
      </visual>
      <collision>
        <origin rpy="${M_PI/2} 0 0"/>
        <geometry>
          <cylinder length="${length}" radius="${radius}"/>
        </geometry>
      </collision>
    </link>

    <joint name="${side}_${type}_joint" type="revolute">
      <parent link="${limits_origin}"/>
      <child link="${side}_${type}_segment"/>
      <origin xyz="${xyz}"/>
      <axis xyz="${axis}"/>
      <limit lower="-2.0" upper="2.0" effort="100" velocity="3"/>
      <dynamics damping="0.1" friction="0.0"/>
    </joint>
  </xacro:macro>

  <!-- Base torso -->
  <link name="torso">
    <inertial>
      <mass value="10"/>
      <origin xyz="0 0 ${torso_height/2}"/>
      <inertia ixx="1.0" ixy="0" ixz="0" iyy="1.0" iyz="0" izz="1.0"/>
    </inertial>
    <visual>
      <origin xyz="0 0 ${torso_height/2}"/>
      <geometry>
        <box size="${torso_width} ${torso_depth} ${torso_height}"/>
      </geometry>
      <material name="torso_material"/>
    </visual>
    <collision>
      <origin xyz="0 0 ${torso_height/2}"/>
      <geometry>
        <box size="${torso_width} ${torso_depth} ${torso_height}"/>
      </geometry>
    </collision>
  </link>

  <!-- Create limbs using macro -->
  <xacro:humanoid_limb
    side="left"
    type="upper_arm"
    length="0.3"
    radius="0.05"
    mass="1.5"
    xyz="0.15 0 0.3"
    axis="0 0 1"
    limits_origin="torso"/>

  <xacro:humanoid_limb
    side="right"
    type="upper_arm"
    length="0.3"
    radius="0.05"
    mass="1.5"
    xyz="-0.15 0 0.3"
    axis="0 0 1"
    limits_origin="torso"/>

  <!-- Head -->
  <link name="head">
    <inertial>
      <mass value="2"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/>
    </inertial>
    <visual>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
      <material name="head_material"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
    </collision>
  </link>

  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 ${torso_height}"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.0" upper="1.0" effort="10" velocity="1"/>
  </joint>

  <!-- Sensors -->
  <link name="imu_link">
    <inertial>
      <mass value="0.01"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="imu_joint" type="fixed">
    <parent link="torso"/>
    <child link="imu_link"/>
    <origin xyz="0 0 0.1"/>
  </joint>

  <!-- Transmissions for simulation -->
  <xacro:include filename="$(find humanoid_description)/urdf/transmission.xacro" />

</robot>
```

## Sensors Integration

### Camera and IMU Integration

```xml
<!-- Camera sensor -->
<link name="camera_optical_frame">
  <inertial>
    <mass value="0.01"/>
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
  </inertial>
</link>

<joint name="camera_optical_joint" type="fixed">
  <parent link="head"/>
  <child link="camera_optical_frame"/>
  <origin xyz="0.05 0 0.05" rpy="${M_PI/2} 0 ${M_PI/2}"/>
</joint>

<!-- IMU sensor -->
<link name="imu_link">
  <inertial>
    <mass value="0.01"/>
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
  </inertial>
</link>

<joint name="imu_joint" type="fixed">
  <parent link="torso"/>
  <child link="imu_link"/>
  <origin xyz="0 0 0.1"/>
</joint>
```

## Validation and Debugging

### URDF Validation

```bash
# Check URDF for errors
check_urdf /path/to/robot.urdf

# Parse and display URDF structure
urdf_to_graphiz /path/to/robot.urdf
```

### Common Issues and Solutions

1. **Joint Limits**: Ensure realistic joint limits for humanoid anatomy
2. **Inertial Properties**: Use realistic mass and inertia values
3. **Collision Geometry**: Simplify complex shapes for better physics performance
4. **Origin Alignments**: Verify joint origins and link placements

## Advanced Features

### Gazebo-Specific Tags

```xml
<gazebo reference="head">
  <material>Gazebo/Blue</material>
  <turnGravityOff>false</turnGravityOff>
</gazebo>

<gazebo>
  <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
    <alwaysOn>true</alwaysOn>
    <updateRate>30.0</updateRate>
    <cameraName>head_camera</cameraName>
    <imageTopicName>image_raw</imageTopicName>
    <cameraInfoTopicName>camera_info</cameraInfoTopicName>
    <frameName>head_camera_optical_frame</frameName>
  </plugin>
</gazebo>
```

### Transmission Elements

```xml
<transmission name="left_shoulder_trans">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="left_shoulder_joint">
    <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
  </joint>
  <actuator name="left_shoulder_motor">
    <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

## Best Practices for Humanoid URDF

### Design Principles

1. **Anatomical Accuracy**: Model joints and ranges of motion based on human anatomy
2. **Mass Distribution**: Ensure realistic mass distribution for stable locomotion
3. **Modularity**: Use Xacro macros for reusable components
4. **Performance**: Simplify collision geometry for efficient physics simulation
5. **Standards**: Follow ROS conventions for link and joint naming

### Common Humanoid Joint Ranges

```xml
<!-- Shoulder joints (2 DOF) -->
<limit lower="-2.0" upper="2.0" effort="100" velocity="2"/>

<!-- Elbow joints (1 DOF) -->
<limit lower="0" upper="2.5" effort="80" velocity="3"/>

<!-- Hip joints (3 DOF) -->
<limit lower="-1.57" upper="1.57" effort="150" velocity="1.5"/>

<!-- Knee joints (1 DOF) -->
<limit lower="0" upper="2.5" effort="120" velocity="1.5"/>

<!-- Ankle joints (2 DOF) -->
<limit lower="-0.87" upper="0.87" effort="60" velocity="1"/>
```

## Next Steps

Continue to [Launch Files](./launch-files.md) to learn how to configure and launch your humanoid robot system.

## Learning Objectives Review

- Understand the structure and components of URDF files ✓
- Create detailed humanoid robot models using URDF ✓
- Apply Xacro macros to simplify complex humanoid models ✓
- Integrate visual, collision, and inertial properties for humanoids ✓
- Validate and debug URDF models for humanoid robots ✓

## Practical Exercise

Create a simplified humanoid model with:
1. Torso, head, and 4 limbs (arms and legs)
2. Proper joint definitions with realistic limits
3. Visual and collision properties
4. Use Xacro macros to reduce code duplication
5. Validate the URDF file for errors

## Assessment Questions

1. What are the three main components of a URDF link?
2. Explain the difference between visual and collision properties.
3. Why is Xacro useful for humanoid robot models?
4. What are the key considerations for defining inertial properties in humanoid robots?

## Further Reading

- URDF Documentation: http://wiki.ros.org/urdf
- Xacro Documentation: http://wiki.ros.org/xacro
- ROS URDF Tutorials: http://wiki.ros.org/urdf/Tutorials
- Biomechanics of Human Motion: For realistic joint limits and ranges