---
id: physics-simulation
title: Physics Simulation
sidebar_position: 3
description: Comprehensive guide to physics simulation in robotics, including gravity, collisions, and constraints
keywords: [physics simulation, gravity, collisions, constraints, rigid body dynamics, robotics simulation]
---

# Physics Simulation

Physics simulation is fundamental to creating realistic digital twins of robotic systems. This chapter covers the core physics principles and implementation techniques used in robotics simulation environments.

## Learning Objectives

- Understand the principles of rigid body dynamics in simulation
- Implement gravity and environmental forces in simulation
- Configure collision detection and response systems
- Apply constraints and joints in physics simulation
- Optimize physics simulation for performance and stability

## Rigid Body Dynamics Fundamentals

### Newtonian Mechanics

Physics simulation in robotics is based on Newtonian mechanics, which describes the motion of rigid bodies:

- **Newton's First Law**: An object at rest stays at rest, and an object in motion stays in motion at constant velocity, unless acted upon by an unbalanced force
- **Newton's Second Law**: F = ma (Force equals mass times acceleration)
- **Newton's Third Law**: For every action, there is an equal and opposite reaction

### Key Physics Concepts

#### Mass and Inertia
```xml
<inertial>
  <mass>1.0</mass>  <!-- Mass in kg -->
  <inertia>
    <ixx>0.01</ixx>  <!-- Moments of inertia -->
    <ixy>0</ixy>
    <ixz>0</ixz>
    <iyy>0.01</iyy>
    <iyz>0</iyz>
    <izz>0.01</izz>
  </inertia>
</inertial>
```

The inertia tensor describes how mass is distributed in a rigid body, affecting its rotational motion.

#### Center of Mass
The center of mass is the point where the total mass of the body can be considered concentrated. It's crucial for realistic physics simulation.

### Forces in Simulation

#### Gravity
Gravity is the most fundamental force in physics simulation:

```xml
<physics type="ode">
  <gravity>0 0 -9.8</gravity>  <!-- Standard Earth gravity -->
</physics>
```

#### Contact Forces
When objects touch, contact forces prevent them from penetrating each other. These forces are calculated based on:
- Material properties (friction, restitution)
- Contact geometry
- Relative velocities

## Collision Detection and Response

### Collision Detection Methods

#### Broad Phase
- Uses spatial partitioning to quickly identify potentially colliding pairs
- Common methods: Axis-Aligned Bounding Box (AABB) trees, spatial hashing

#### Narrow Phase
- Performs precise collision detection between potentially colliding objects
- Calculates contact points and penetration depths

### Collision Properties

#### Friction
Friction models the resistance to sliding motion between surfaces:

```xml
<surface>
  <friction>
    <ode>
      <mu>1.0</mu>      <!-- Static friction coefficient -->
      <mu2>1.0</mu2>    <!-- Dynamic friction coefficient -->
      <slip1>0.0</slip1> <!-- Primary slip coefficient -->
      <slip2>0.0</slip2> <!-- Secondary slip coefficient -->
    </ode>
  </friction>
</surface>
```

#### Restitution (Bounciness)
Restitution determines how "bouncy" collisions are:

```xml
<surface>
  <bounce>
    <restitution_coefficient>0.2</restitution_coefficient>
    <threshold>100000.0</threshold>
  </bounce>
</surface>
```

### Collision Geometries

#### Primitive Shapes
- **Box**: `box` with size parameter
- **Sphere**: `sphere` with radius parameter
- **Cylinder**: `cylinder` with radius and length parameters
- **Capsule**: `capsule` for rounded cylinders

#### Mesh Collisions
```xml
<collision name="mesh_collision">
  <geometry>
    <mesh>
      <uri>model://my_robot/meshes/link.stl</uri>
      <scale>1.0 1.0 1.0</scale>
    </mesh>
  </geometry>
</collision>
```

## Constraints and Joints

### Joint Types

#### Revolute Joint
A rotational joint with one degree of freedom:

```xml
<joint name="elbow_joint" type="revolute">
  <parent>upper_arm</parent>
  <child>lower_arm</child>
  <axis>
    <xyz>0 1 0</xyz>  <!-- Rotation axis -->
    <limit>
      <lower>-1.57</lower>  <!-- Joint limits -->
      <upper>1.57</upper>
      <effort>100</effort>
      <velocity>1</velocity>
    </limit>
    <dynamics>
      <damping>0.1</damping>    <!-- Damping coefficient -->
      <friction>0.0</friction>  <!-- Static friction -->
    </dynamics>
  </axis>
</joint>
```

#### Prismatic Joint
A sliding joint with linear motion:

```xml
<joint name="slider_joint" type="prismatic">
  <parent>base</parent>
  <child>slider</child>
  <axis>
    <xyz>1 0 0</xyz>
    <limit>
      <lower>0</lower>
      <upper>0.5</upper>
      <effort>50</effort>
      <velocity>0.5</velocity>
    </limit>
  </axis>
</joint>
```

#### Fixed Joint
A joint that prevents all relative motion:

```xml
<joint name="fixed_joint" type="fixed">
  <parent>base</parent>
  <child>sensor_mount</child>
</joint>
```

### Joint Dynamics

#### Damping
Damping simulates energy loss due to friction and other dissipative forces:

```xml
<dynamics>
  <damping>0.1</damping>      <!-- Viscous damping -->
  <friction>0.01</friction>   <!-- Static friction -->
  <spring_reference>0</spring_reference>
  <spring_stiffness>0</spring_stiffness>
</dynamics>
```

#### Limits
Joint limits prevent motion beyond specified ranges:

```xml
<limit>
  <lower>-2.0</lower>    <!-- Lower limit (radians or meters) -->
  <upper>2.0</upper>     <!-- Upper limit -->
  <effort>100</effort>   <!-- Maximum effort -->
  <velocity>2</velocity> <!-- Maximum velocity -->
</limit>
```

## Physics Engine Comparison

### ODE (Open Dynamics Engine)
- **Strengths**: Stable for most applications, good collision detection
- **Weaknesses**: Can be slower for complex simulations
- **Best for**: General robotics simulation

```xml
<physics type="ode">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>
  <ode>
    <solver>
      <type>quick</type>        <!-- Solver type -->
      <iters>10</iters>         <!-- Iterations per step -->
      <sor>1.3</sor>            <!-- Successive over-relaxation -->
    </solver>
    <constraints>
      <cfm>0.0</cfm>            <!-- Constraint force mixing -->
      <erp>0.2</erp>            <!-- Error reduction parameter -->
      <contact_max_correcting_vel>100</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

### Bullet Physics
- **Strengths**: Fast performance, good for complex scenes
- **Weaknesses**: Less stable than ODE for some scenarios
- **Best for**: High-performance simulations

### DART (Dynamic Animation and Robotics Toolkit)
- **Strengths**: Advanced dynamics, soft body support
- **Weaknesses**: More complex to configure
- **Best for**: Advanced dynamics simulation

## Simulation Parameters

### Time Step Configuration
The time step affects both stability and performance:

```xml
<physics type="ode">
  <max_step_size>0.001</max_step_size>  <!-- Smaller = more accurate but slower -->
  <real_time_factor>1</real_time_factor> <!-- 1 = real-time, >1 = faster than real-time -->
  <real_time_update_rate>1000</real_time_update_rate> <!-- Updates per second -->
</physics>
```

### Stability Considerations
- Smaller time steps improve stability but reduce performance
- Higher solver iterations improve accuracy but reduce performance
- Proper mass and inertia values are crucial for stability

## Physics Optimization Techniques

### Simplification Strategies

#### Collision Geometry Simplification
Use simpler shapes for collision detection while keeping detailed meshes for visualization:

```xml
<!-- Detailed visual mesh -->
<visual name="visual">
  <geometry>
    <mesh>
      <uri>model://robot/meshes/detailed_model.dae</uri>
    </mesh>
  </geometry>
</visual>

<!-- Simplified collision geometry -->
<collision name="collision">
  <geometry>
    <box>
      <size>0.2 0.2 0.1</size>
    </box>
  </geometry>
</collision>
```

#### Level of Detail (LOD)
Switch between different detail levels based on distance or importance:

```xml
<collision name="collision_lod">
  <geometry>
    <mesh>
      <uri>model://robot/meshes/simple_collision.stl</uri>
    </mesh>
  </geometry>
  <surface>
    <contact>
      <ode>
        <max_vel>100</max_vel>
        <min_depth>0.001</min_depth>
      </ode>
    </contact>
  </surface>
</collision>
```

### Performance Optimization

#### Spatial Partitioning
Organize objects spatially to reduce collision checks:

```xml
<!-- In world file -->
<physics type="ode">
  <ode>
    <broadphase>hash_space</broadphase>  <!-- or "octree", "simple" -->
  </ode>
</physics>
```

#### Fixed Joints Optimization
Combine multiple links connected by fixed joints into a single body when possible.

## Real-World Physics Considerations

### Scaling Effects
Physics properties must scale appropriately:

- **Mass**: Should be proportional to volume (scale³)
- **Inertia**: Should scale with mass and dimensions squared
- **Forces**: May need adjustment based on simulation scale

### Material Properties
Realistic material properties enhance simulation fidelity:

```xml
<surface>
  <friction>
    <ode>
      <mu>0.8</mu>    <!-- Rubber on concrete: ~0.8 -->
      <mu2>0.7</mu2>
    </ode>
  </friction>
  <bounce>
    <restitution_coefficient>0.1</restitution_coefficient>  <!-- Low bounce for rubber -->
  </bounce>
</surface>
```

## Debugging Physics Issues

### Common Problems and Solutions

#### Objects Falling Through Surfaces
- **Cause**: Penetration depth too small or time step too large
- **Solution**: Increase `min_depth` or decrease `max_step_size`

#### Unstable Joint Motion
- **Cause**: Improper mass/inertia or constraint violations
- **Solution**: Verify mass properties and adjust ERP/CFM parameters

#### Excessive Penetration
- **Cause**: Weak contact constraints
- **Solution**: Increase `contact_max_correcting_vel` or adjust ERP

### Debug Visualization
Enable physics debug visualization to identify issues:

```bash
# In Gazebo GUI, enable View -> Contacts
# Or add to world file:
<world>
  <gui>
    <camera name="user_camera">
      <view_controller>orbit</view_controller>
    </camera>
  </gui>
</world>
```

## Advanced Physics Concepts

### Soft Body Simulation
For deformable objects, though not commonly used in basic robotics:

```xml
<!-- Requires special physics engine support -->
<model name="soft_object">
  <link name="soft_link">
    <collision name="collision">
      <geometry>
        <mesh>
          <uri>model://soft_object/soft_mesh.stl</uri>
        </mesh>
      </geometry>
      <surface>
        <soft_erp>0.8</soft_erp>  <!-- Soft error reduction -->
      </surface>
    </collision>
  </link>
</model>
```

### Fluid Simulation
For simulating interaction with liquids or gases (advanced):

```xml
<!-- In world file -->
<physics type="ode">
  <fluid>
    <density>1000</density>  <!-- Water density -->
    <viscosity>0.001</viscosity>
  </fluid>
</physics>
```

## Learning Objectives Review

- Understand the principles of rigid body dynamics in simulation ✓
- Implement gravity and environmental forces in simulation ✓
- Configure collision detection and response systems ✓
- Apply constraints and joints in physics simulation ✓
- Optimize physics simulation for performance and stability ✓

## Practical Exercise

1. Create a simulation world with multiple objects of different materials
2. Configure different friction and restitution properties
3. Set up a simple robot with revolute joints
4. Run the simulation and observe the physics behavior
5. Adjust parameters to achieve stable, realistic motion

## Assessment Questions

1. Explain the difference between static and dynamic friction in physics simulation.
2. What is the role of the error reduction parameter (ERP) in joint constraints?
3. Why is it important to have proper mass and inertia values in simulation?
4. How do you balance simulation accuracy with performance?

## Further Reading

- "Real-Time Collision Detection" by Christer Ericson
- "Game Physics Engine Development" by Ian Millington
- Gazebo Physics Tutorial: http://gazebosim.org/tutorials?tut=physics_ros
- ODE User Guide: http://ode.org/wiki/index.php?title=Manual

## Next Steps

Continue to [URDF and SDF](./urdf-sdf.md) to learn about the formats used to describe robots and environments in simulation.