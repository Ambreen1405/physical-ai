---
id: isaac-sim
title: Isaac Sim
sidebar_position: 3
description: Comprehensive guide to NVIDIA Isaac Sim for photorealistic robotics simulation
keywords: [isaac sim, simulation, photorealistic, omniverse, robotics simulation, domain randomization]
---

# Isaac Sim: Photorealistic Robotics Simulation

Isaac Sim is NVIDIA's high-fidelity simulation environment built on the Omniverse platform. It provides photorealistic rendering, accurate physics simulation, and domain randomization capabilities for training AI models for robotics applications.

## Learning Objectives

- Understand Isaac Sim architecture and core components
- Set up and configure Isaac Sim for robotics applications
- Create photorealistic environments for robotics training
- Implement domain randomization techniques
- Generate synthetic datasets for AI model training
- Integrate Isaac Sim with ROS/ROS 2 workflows

## Isaac Sim Architecture

### Core Components

Isaac Sim consists of several integrated components:

```
┌─────────────────────────────────────────────────────────────┐
│                    Isaac Sim Architecture                   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │   USD Scene     │  │   PhysX Physics │  │   RTX       │  │
│  │   Description   │  │   Engine        │  │   Renderer  │  │
│  │                 │  │                 │  │             │  │
│  └─────────────────┘  └─────────────────┘  └─────────────┘  │
│           │                     │                   │       │
│           ▼                     ▼                   ▼       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │   Scene Graph   │  │   Collision     │  │   Lighting  │  │
│  │   Management    │  │   Detection     │  │   System    │  │
│  └─────────────────┘  └─────────────────┘  └─────────────┘  │
│           │                     │                   │       │
│           └─────────────────────┼───────────────────┘       │
│                                 ▼                           │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Omniverse Kit Framework                    │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │ │
│  │  │ Extensions  │  │ Extensions  │  │ Extensions      │  │ │
│  │  │ (Robotics)  │  │ (AI/ML)     │  │ (ROS Bridge)    │  │ │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘  │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### USD (Universal Scene Description)

USD is the core scene representation format used by Isaac Sim:

- **Hierarchical Structure**: Tree-based scene organization
- **Layering System**: Composable scene layers
- **Variant Sets**: Multiple scene configurations
- **Animation Support**: Timeline and keyframe animation
- **Multi-User Collaboration**: Concurrent editing capabilities

#### USD Prim Hierarchy Example

```python
# Isaac Sim USD prim structure
/World
├── /GroundPlane
├── /Lights
│   ├── /DomeLight
│   ├── /KeyLight
│   └── /FillLight
├── /Cameras
│   └── /RGB_Camera
├── /Robots
│   └── /MyRobot
│       ├── /base_link
│       ├── /left_wheel
│       └── /right_wheel
└── /Objects
    ├── /Table
    └── /Cubes
        ├── /Cube_1
        └── /Cube_2
```

## Installation and Setup

### Prerequisites

Before installing Isaac Sim, ensure your system meets the requirements:

```bash
# Check GPU compatibility
nvidia-smi

# Verify CUDA installation
nvcc --version

# Check available disk space (recommended: 100GB+)
df -h $HOME

# Install dependencies
sudo apt update
sudo apt install -y build-essential cmake git python3-dev python3-pip
```

### Isaac Sim Installation

#### Method 1: Direct Installation

```bash
# Download Isaac Sim
wget https://developer.nvidia.com/isaac/downloads/isaac-sim-2023-2-0-release.tar.gz

# Extract
tar -xzf isaac-sim-2023-2-0-release.tar.gz

# Navigate to directory
cd isaac-sim-2023.2.0

# Install
./install.sh

# Launch Isaac Sim
./isaac-sim.sh
```

#### Method 2: Docker Installation

```bash
# Pull Isaac Sim Docker image
docker pull nvcr.io/nvidia/isaac-sim:2023.2.1

# Run Isaac Sim container
docker run --gpus all -it \
  --rm \
  --network=host \
  --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw \
  --volume=$HOME/.Xauthority:/root/.Xauthority:rw \
  --volume=$PWD:/workspace \
  --env="DISPLAY" \
  --env="ACCEPT_EULA=Y" \
  --env="NVIDIA_VISIBLE_DEVICES=all" \
  --env="NVIDIA_DRIVER_CAPABILITIES=all" \
  --name="isaac-sim" \
  nvcr.io/nvidia/isaac-sim:2023.2.1
```

### Initial Configuration

After installation, configure Isaac Sim for optimal performance:

```python
# Configure Isaac Sim settings programmatically
import omni
from omni.isaac.core.utils.settings import set_carb_setting

# Set rendering quality
set_carb_setting("/app/viewport/renderQuality", 3)  # High quality

# Set physics substeps for stability
set_carb_setting("/physics/solverVelocityIterationCount", 8)
set_carb_setting("/physics/solverPositionIterationCount", 4)

# Configure texture streaming
set_carb_setting("/renderer/textureStreaming/resolutionScale", 1.0)
set_carb_setting("/renderer/textureStreaming/enable", True)

# Set up GPU acceleration
set_carb_setting("/renderer/antiAliasingMode", 1)  # FXAA
set_carb_setting("/renderer/physicallyBased", True)
```

## Creating Photorealistic Environments

### Environment Setup Script

```python
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.carb import carb_settings_path
import carb

def setup_photorealistic_environment():
    """Create a photorealistic environment for robotics simulation"""

    # Initialize the world
    world = World(stage_units_in_meters=1.0)

    # Add ground plane
    world.scene.add_default_ground_plane(
        prim_path="/World/defaultGroundPlane",
        name="default_ground_plane",
        static_friction=0.5,
        dynamic_friction=0.5,
        restitution=0.8
    )

    # Add dome light for realistic environment lighting
    dome_light = world.scene.add(
        omni.isaac.core.objects.DomeLight(
            prim_path="/World/DomeLight",
            name="dome_light",
            intensity=3000,
            color=(0.8, 0.8, 0.8)
        )
    )

    # Add key light for directional illumination
    key_light = world.scene.add(
        omni.isaac.core.objects.DistantLight(
            prim_path="/World/KeyLight",
            name="key_light",
            intensity=400,
            color=(1.0, 1.0, 1.0),
            position=(5, 5, 10),
            look_at=(0, 0, 0)
        )
    )

    # Add fill light to reduce harsh shadows
    fill_light = world.scene.add(
        omni.isaac.core.objects.DistantLight(
            prim_path="/World/FillLight",
            name="fill_light",
            intensity=150,
            color=(0.7, 0.7, 0.9),
            position=(-3, 2, 8),
            look_at=(0, 0, 0)
        )
    )

    return world

def add_textured_objects(world):
    """Add textured objects to the environment"""

    # Add a textured table
    table = world.scene.add(
        omni.isaac.core.objects.FixedCuboid(
            prim_path="/World/Table",
            name="table",
            position=[0, 0, 0.5],
            size=[1.0, 0.6, 0.05],
            color=[0.8, 0.6, 0.4]
        )
    )

    # Add textured cubes with random materials
    materials = [
        (0.8, 0.2, 0.2),  # Red
        (0.2, 0.8, 0.2),  # Green
        (0.2, 0.2, 0.8),  # Blue
        (0.8, 0.8, 0.2),  # Yellow
    ]

    import random
    for i in range(4):
        cube = world.scene.add(
            omni.isaac.core.objects.DynamicCuboid(
                prim_path=f"/World/Cube_{i}",
                name=f"cube_{i}",
                position=[random.uniform(-0.4, 0.4), random.uniform(-0.3, 0.3), 0.55],
                size=0.1,
                color=materials[i]
            )
        )

def main():
    world = setup_photorealistic_environment()
    add_textured_objects(world)

    # Play the simulation
    world.reset()

    # Run simulation loop
    for i in range(1000):
        world.step(render=True)

    world.stop()

if __name__ == "__main__":
    main()
```

### Material and Texture Configuration

```python
import omni
from pxr import UsdShade, Sdf, Gf, UsdGeom

def create_realistic_materials():
    """Create realistic materials for photorealistic rendering"""

    stage = omni.usd.get_context().get_stage()

    # Create a realistic metal material
    metal_material_path = Sdf.Path("/World/Materials/MetalMaterial")
    metal_material = UsdShade.Material.Define(stage, metal_material_path)

    # Create PBR shader
    metal_shader = UsdShade.Shader.Define(stage, metal_material_path.AppendChild("Surface"))
    metal_shader.CreateIdAttr("OmniPBR")

    # Set material properties
    metal_shader.CreateInput("diffuse_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.7, 0.7, 0.8))
    metal_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.9)
    metal_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.1)
    metal_shader.CreateInput("specular_reflection", Sdf.ValueTypeNames.Float).Set(0.5)

    # Connect shader to material
    metal_material.CreateSurfaceOutput().ConnectToSource(metal_shader.ConnectableAPI(), "outputs:surface")

    # Create a realistic fabric material
    fabric_material_path = Sdf.Path("/World/Materials/FabricMaterial")
    fabric_material = UsdShade.Material.Define(stage, fabric_material_path)

    fabric_shader = UsdShade.Shader.Define(stage, fabric_material_path.AppendChild("Surface"))
    fabric_shader.CreateIdAttr("OmniPBR")

    fabric_shader.CreateInput("diffuse_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.4, 0.6, 0.8))
    fabric_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    fabric_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    fabric_shader.CreateInput("specular_reflection", Sdf.ValueTypeNames.Float).Set(0.2)

    fabric_material.CreateSurfaceOutput().ConnectToSource(fabric_shader.ConnectableAPI(), "outputs:surface")

def apply_material_to_object(object_path, material_path):
    """Apply material to an object"""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(object_path)

    # Apply material to the object
    UsdShade.MaterialBindingAPI(prim).Bind(
        UsdShade.Material(stage.GetPrimAtPath(material_path))
    )
```

## Domain Randomization

### Understanding Domain Randomization

Domain randomization is a technique that increases the diversity of training data by varying environmental properties randomly during simulation:

- **Lighting Conditions**: Randomize light positions, intensities, and colors
- **Material Properties**: Randomize colors, textures, and surface properties
- **Object Placement**: Randomize positions, orientations, and scales
- **Camera Parameters**: Randomize focal length, sensor size, and noise
- **Background Environments**: Randomize backgrounds and environments

### Domain Randomization Implementation

```python
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.prims import get_prim_at_path
import numpy as np
import random

class DomainRandomizer:
    def __init__(self):
        self.world = None
        self.randomization_config = {}
        self.step_count = 0

    def setup_randomization(self, config):
        """Configure domain randomization parameters"""
        self.randomization_config = config

    def randomize_lighting(self):
        """Randomize lighting conditions"""
        if 'lights' not in self.randomization_config:
            return

        lights_config = self.randomization_config['lights']

        # Randomize dome light
        dome_light = get_prim_at_path("/World/DomeLight")
        if dome_light:
            # Randomize intensity
            intensity = random.uniform(
                lights_config['intensity']['min'],
                lights_config['intensity']['max']
            )
            dome_light.GetAttribute("inputs:intensity").Set(intensity)

            # Randomize color
            color = (
                random.uniform(lights_config['color']['min'][0], lights_config['color']['max'][0]),
                random.uniform(lights_config['color']['min'][1], lights_config['color']['max'][1]),
                random.uniform(lights_config['color']['min'][2], lights_config['color']['max'][2])
            )
            dome_light.GetAttribute("inputs:color").Set(color)

    def randomize_materials(self):
        """Randomize material properties"""
        if 'materials' not in self.randomization_config:
            return

        materials_config = self.randomization_config['materials']

        # Iterate through objects and randomize materials
        stage = omni.usd.get_context().get_stage()
        for prim in stage.Traverse():
            if prim.GetTypeName() == "Geometry":
                # Randomize diffuse color
                if random.random() < materials_config['color_variation_probability']:
                    new_color = (
                        random.uniform(0.1, 1.0),
                        random.uniform(0.1, 1.0),
                        random.uniform(0.1, 1.0)
                    )

                    # Find material shader and update color
                    material_binding_api = omni.usd.get_prim_material(prim)
                    if material_binding_api:
                        shader = material_binding_api.GetBoundMaterial()
                        if shader:
                            shader.GetShader().GetInput("diffuse_color").Set(new_color)

    def randomize_objects(self):
        """Randomize object positions and properties"""
        if 'objects' not in self.randomization_config:
            return

        objects_config = self.randomization_config['objects']

        # Randomize object positions
        for i in range(objects_config['count']):
            object_path = f"/World/Object_{i}"
            prim = get_prim_at_path(object_path)

            if prim:
                # Randomize position
                new_pos = [
                    random.uniform(objects_config['position']['x']['min'], objects_config['position']['x']['max']),
                    random.uniform(objects_config['position']['y']['min'], objects_config['position']['y']['max']),
                    random.uniform(objects_config['position']['z']['min'], objects_config['position']['z']['max'])
                ]

                # Apply new position
                xform = UsdGeom.Xformable(prim)
                xform.AddTranslateOp().Set(new_pos)

    def randomize_cameras(self):
        """Randomize camera parameters"""
        if 'cameras' not in self.randomization_config:
            return

        cameras_config = self.randomization_config['cameras']

        # Randomize camera intrinsics
        camera_prim = get_prim_at_path("/World/Camera")
        if camera_prim:
            # Randomize focal length
            focal_length = random.uniform(
                cameras_config['focal_length']['min'],
                cameras_config['focal_length']['max']
            )
            camera_prim.GetAttribute("inputs:horizontalAperture").Set(focal_length)

    def apply_randomization(self):
        """Apply all randomization effects"""
        self.randomize_lighting()
        self.randomize_materials()
        self.randomize_objects()
        self.randomize_cameras()

        self.step_count += 1

        # Reset after certain steps if needed
        if self.step_count % self.randomization_config.get('reset_interval', 100) == 0:
            self.reset_randomization()

    def reset_randomization(self):
        """Reset randomization for new episode"""
        self.step_count = 0
        # Optionally reset specific parameters

def setup_domain_randomization():
    """Setup domain randomization configuration"""

    config = {
        'lights': {
            'intensity': {'min': 1000, 'max': 5000},
            'color': {
                'min': [0.5, 0.5, 0.5],
                'max': [1.0, 1.0, 1.0]
            }
        },
        'materials': {
            'color_variation_probability': 0.3
        },
        'objects': {
            'count': 10,
            'position': {
                'x': {'min': -2.0, 'max': 2.0},
                'y': {'min': -2.0, 'max': 2.0},
                'z': {'min': 0.1, 'max': 2.0}
            }
        },
        'cameras': {
            'focal_length': {'min': 18.0, 'max': 55.0}
        },
        'reset_interval': 50
    }

    randomizer = DomainRandomizer()
    randomizer.setup_randomization(config)

    return randomizer

# Example usage
def main():
    # Setup world
    world = World(stage_units_in_meters=1.0)

    # Setup domain randomization
    randomizer = setup_domain_randomization()

    # Play simulation
    world.play()

    # Run simulation with randomization
    for i in range(10000):
        if i % 10 == 0:  # Apply randomization every 10 steps
            randomizer.apply_randomization()

        world.step(render=True)

    world.stop()

if __name__ == "__main__":
    main()
```

## Synthetic Data Generation

### Data Generation Pipeline

```python
import omni
from omni.isaac.core import World
from omni.isaac.sensor import Camera
from PIL import Image
import numpy as np
import json
import os
from pxr import UsdGeom

class SyntheticDataGenerator:
    def __init__(self, output_dir="synthetic_data"):
        self.output_dir = output_dir
        self.world = None
        self.cameras = []
        self.annotation_data = []

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)

    def add_camera(self, name, position, look_at, resolution=(640, 480)):
        """Add a camera for data capture"""
        camera = Camera(
            prim_path=f"/World/Cameras/{name}",
            name=name,
            position=position,
            look_at=look_at,
            resolution=resolution
        )

        self.cameras.append({
            'camera': camera,
            'name': name,
            'resolution': resolution
        })

        return camera

    def capture_rgb_data(self, camera_idx=0):
        """Capture RGB image from camera"""
        camera = self.cameras[camera_idx]['camera']

        # Get RGB data
        rgb_data = camera.get_rgb()

        # Convert to PIL Image
        img = Image.fromarray(rgb_data, mode="RGB")

        return img

    def capture_depth_data(self, camera_idx=0):
        """Capture depth data from camera"""
        camera = self.cameras[camera_idx]['camera']

        # Get depth data
        depth_data = camera.get_depth()

        return depth_data

    def capture_segmentation_data(self, camera_idx=0):
        """Capture semantic segmentation data"""
        camera = self.cameras[camera_idx]['camera']

        # Get segmentation data
        seg_data = camera.get_semantic_segmentation()

        return seg_data

    def generate_annotations(self):
        """Generate annotation data for the current scene"""
        stage = omni.usd.get_context().get_stage()

        annotations = {
            'objects': [],
            'scene': {
                'timestamp': self.world.current_time_step_index,
                'environment': 'photorealistic_room'
            }
        }

        # Extract object information
        for prim in stage.Traverse():
            if prim.GetTypeName() in ["Mesh", "Cone", "Cube", "Cylinder", "Sphere"]:
                prim_name = prim.GetName()

                # Get object pose
                xformable = UsdGeom.Xformable(prim)
                transform_matrix = xformable.ComputeLocalToWorldTransform(0)

                # Get bounding box
                bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
                bbox = bbox_cache.ComputeWorldBound(prim)

                if not bbox.IsEmpty():
                    min_point = bbox.GetMin()
                    max_point = bbox.GetMax()

                    annotations['objects'].append({
                        'name': prim_name,
                        'type': prim.GetTypeName(),
                        'bbox': {
                            'min': [min_point[0], min_point[1], min_point[2]],
                            'max': [max_point[0], max_point[1], max_point[2]]
                        },
                        'pose': {
                            'translation': [transform_matrix[i][3] for i in range(3)],
                            'rotation_matrix': [[transform_matrix[i][j] for j in range(3)] for i in range(3)]
                        }
                    })

        return annotations

    def save_data_sample(self, sample_id):
        """Save a complete data sample with all modalities"""
        # Create sample directory
        sample_dir = os.path.join(self.output_dir, f"sample_{sample_id}")
        os.makedirs(sample_dir, exist_ok=True)

        # Capture data from all cameras
        for cam_idx, cam_info in enumerate(self.cameras):
            # RGB image
            rgb_img = self.capture_rgb_data(cam_idx)
            rgb_path = os.path.join(sample_dir, f"{cam_info['name']}_rgb.png")
            rgb_img.save(rgb_path)

            # Depth data
            depth_data = self.capture_depth_data(cam_idx)
            depth_path = os.path.join(sample_dir, f"{cam_info['name']}_depth.npy")
            np.save(depth_path, depth_data)

            # Segmentation data
            seg_data = self.capture_segmentation_data(cam_idx)
            seg_path = os.path.join(sample_dir, f"{cam_info['name']}_seg.png")
            seg_img = Image.fromarray(seg_data.astype(np.uint8), mode="L")
            seg_img.save(seg_path)

        # Generate and save annotations
        annotations = self.generate_annotations()
        annotation_path = os.path.join(sample_dir, "annotations.json")
        with open(annotation_path, 'w') as f:
            json.dump(annotations, f, indent=2)

        # Add to annotation data
        self.annotation_data.append({
            'sample_id': sample_id,
            'path': sample_dir,
            'annotations': annotations
        })

    def generate_dataset(self, num_samples=1000):
        """Generate a complete synthetic dataset"""
        print(f"Generating {num_samples} synthetic data samples...")

        for i in range(num_samples):
            if i % 100 == 0:
                print(f"Generated {i}/{num_samples} samples")

            # Apply domain randomization
            # (Assuming domain randomizer is set up)

            # Save the current sample
            self.save_data_sample(i)

            # Step the simulation
            self.world.step(render=True)

        # Save dataset metadata
        metadata = {
            'total_samples': num_samples,
            'generated_at': str(self.world.current_time_step_index),
            'cameras_used': [cam['name'] for cam in self.cameras],
            'annotations_format': 'json',
            'modalities': ['rgb', 'depth', 'segmentation']
        }

        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Dataset generation complete! Saved to {self.output_dir}")

def setup_synthetic_data_pipeline():
    """Setup the complete synthetic data generation pipeline"""

    # Initialize world
    world = World(stage_units_in_meters=1.0)

    # Setup environment
    world.scene.add_default_ground_plane()

    # Add lighting
    world.scene.add(
        omni.isaac.core.objects.DomeLight(
            prim_path="/World/DomeLight",
            name="dome_light",
            intensity=3000
        )
    )

    # Add cameras for data capture
    generator = SyntheticDataGenerator()
    generator.world = world

    # Add multiple cameras for different viewpoints
    generator.add_camera(
        name="front_camera",
        position=[2, 0, 1],
        look_at=[0, 0, 0.5],
        resolution=(640, 480)
    )

    generator.add_camera(
        name="top_camera",
        position=[0, 0, 3],
        look_at=[0, 0, 0],
        resolution=(640, 480)
    )

    return world, generator

# Example usage
def main():
    world, generator = setup_synthetic_data_pipeline()

    # Play the simulation
    world.play()

    # Generate dataset
    generator.generate_dataset(num_samples=100)  # Reduced for example

    # Stop simulation
    world.stop()

if __name__ == "__main__":
    main()
```

## ROS/ROS 2 Integration

### Isaac Sim ROS Bridge

```python
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.core.utils.stage import add_reference_to_stage
import carb

def setup_ros_bridge():
    """Setup ROS bridge for Isaac Sim"""

    # Enable ROS bridge extension
    enable_extension("omni.isaac.ros2_bridge")

    # Initialize world with ROS support
    world = World(stage_units_in_meters=1.0)

    # Add robot to simulation
    add_reference_to_stage(
        usd_path="path/to/robot/model.usd",
        prim_path="/World/Robot"
    )

    # Configure ROS settings
    settings = carb.settings.get_settings()
    settings.set("/app/ros2_context", "isaac_sim")
    settings.set("/app/ros2_namespace", "isaac_sim_robot")

    return world

def ros_integration_example():
    """Example of ROS integration with Isaac Sim"""

    world = setup_ros_bridge()

    # Import ROS bridge components
    try:
        import rclpy
        from geometry_msgs.msg import Twist
        from sensor_msgs.msg import Image, CameraInfo
        from std_msgs.msg import String

        # Initialize ROS
        rclpy.init()

        # Create ROS node
        ros_node = rclpy.create_node('isaac_sim_controller')

        # Publishers for robot control
        cmd_vel_pub = ros_node.create_publisher(Twist, '/cmd_vel', 10)
        status_pub = ros_node.create_publisher(String, '/status', 10)

        # Subscribers for sensor data
        def image_callback(msg):
            # Process camera image from Isaac Sim
            pass

        image_sub = ros_node.create_subscription(
            Image, '/camera/image_raw', image_callback, 10
        )

        # Timer for sending commands
        def send_commands():
            twist_msg = Twist()
            twist_msg.linear.x = 0.5  # Move forward
            twist_msg.angular.z = 0.1  # Turn slightly
            cmd_vel_pub.publish(twist_msg)

        # Create timer to send commands every 100ms
        timer = ros_node.create_timer(0.1, send_commands)

        # Main simulation loop
        world.reset()

        try:
            while simulation_app.is_running():
                # Process ROS callbacks
                rclpy.spin_once(ros_node, timeout_sec=0.01)

                # Step simulation
                world.step(render=True)

                # Publish status
                status_msg = String()
                status_msg.data = f"Simulation step: {world.current_time_step_index}"
                status_pub.publish(status_msg)

        except KeyboardInterrupt:
            print("Stopping simulation...")
        finally:
            # Cleanup
            ros_node.destroy_timer(timer)
            ros_node.destroy_node()
            rclpy.shutdown()

    except ImportError:
        print("ROS2 not available, skipping ROS integration")

def main():
    # Setup and run ROS integration example
    ros_integration_example()

if __name__ == "__main__":
    main()
```

## Performance Optimization

### GPU Optimization Techniques

```python
import omni
from omni.isaac.core.utils.settings import set_carb_setting

def optimize_isaac_sim_performance():
    """Optimize Isaac Sim for maximum performance"""

    # Set rendering quality appropriately
    set_carb_setting("/app/viewport/renderQuality", 1)  # Medium quality for better performance

    # Optimize physics solver
    set_carb_setting("/physics/solverVelocityIterationCount", 4)  # Reduce iterations
    set_carb_setting("/physics/solverPositionIterationCount", 2)  # Reduce iterations
    set_carb_setting("/physics/maxSubSteps", 1)  # Single substep for performance

    # Optimize rendering
    set_carb_setting("/renderer/antiAliasingMode", 0)  # Disable anti-aliasing
    set_carb_setting("/renderer/lightCulling", True)  # Enable light culling
    set_carb_setting("/renderer/clusteredLighting", True)  # Enable clustered lighting

    # Optimize texture streaming
    set_carb_setting("/renderer/textureStreaming/resolutionScale", 0.5)  # Lower resolution
    set_carb_setting("/renderer/textureStreaming/maxTexturePoolSize", 1024)  # Limit pool size

    # Optimize shadow maps
    set_carb_setting("/renderer/shadowMap/atmosphericLight", [2048, 2048])  # Smaller shadow maps
    set_carb_setting("/renderer/shadowMap/distantLight", [1024, 1024])

    # Disable unnecessary features
    set_carb_setting("/app/enableProgressiveRenderer", False)  # Disable progressive rendering
    set_carb_setting("/app/renderer/enableViewportRender", False)  # Render off-screen if not needed

def configure_simulation_for_training():
    """Configure simulation for maximum training efficiency"""

    # Optimize for synthetic data generation
    set_carb_setting("/app/viewport/displayOptions", 0)  # Hide viewport for headless training
    set_carb_setting("/app/window/visible", False)  # Hide window

    # Optimize for domain randomization
    set_carb_setting("/app/autoSave/enabled", False)  # Disable auto-save during training
    set_carb_setting("/app/backup/enabled", False)  # Disable backup during training

    # Optimize for high-frequency simulation
    set_carb_setting("/app/runInBackground", True)  # Allow background execution
    set_carb_setting("/app/enableIdleRun", False)  # Disable idle processing

def setup_optimized_world():
    """Setup world with performance optimizations"""

    # Apply optimizations
    optimize_isaac_sim_performance()
    configure_simulation_for_training()

    # Initialize world with optimized settings
    from omni.isaac.core import World
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0/60.0,  # 60 FPS physics
        rendering_dt=1.0/30.0  # 30 FPS rendering (can be adjusted)
    )

    return world
```

## Troubleshooting and Best Practices

### Common Issues and Solutions

1. **GPU Memory Issues**
   - Reduce scene complexity
   - Lower texture resolution
   - Use texture streaming

2. **Performance Bottlenecks**
   - Profile with NVIDIA Nsight
   - Optimize USD stage complexity
   - Use instancing for repeated objects

3. **Physics Instability**
   - Reduce physics timestep
   - Adjust solver iterations
   - Verify mass and inertia properties

### Best Practices

1. **Stage Organization**
   - Use meaningful prim paths
   - Organize objects hierarchically
   - Use variants for different configurations

2. **Asset Management**
   - Use USD composition for complex scenes
   - Reference external assets instead of copying
   - Use point instancers for repeated objects

3. **Simulation Efficiency**
   - Batch randomization operations
   - Use async operations when possible
   - Cache computed values

## Learning Objectives Review

- Understand Isaac Sim architecture and core components ✓
- Set up and configure Isaac Sim for robotics applications ✓
- Create photorealistic environments for robotics training ✓
- Implement domain randomization techniques ✓
- Generate synthetic datasets for AI model training ✓
- Integrate Isaac Sim with ROS/ROS 2 workflows ✓

## Practical Exercise

1. Install Isaac Sim and verify the installation
2. Create a simple scene with a robot and objects
3. Configure domain randomization for lighting and materials
4. Set up synthetic data generation pipeline
5. Integrate with ROS for robot control
6. Optimize the simulation for performance

## Assessment Questions

1. Explain the USD scene description format and its importance in Isaac Sim.
2. How does domain randomization improve the robustness of AI models?
3. What are the key components of the Isaac Sim architecture?
4. How do you optimize Isaac Sim for synthetic data generation?

## Further Reading

- Isaac Sim User Guide: https://docs.omniverse.nvidia.com/isaacsim/latest/
- USD Documentation: https://graphics.pixar.com/usd/docs/index.html
- Omniverse Kit Documentation: https://docs.omniverse.nvidia.com/dev-guide/latest/
- Domain Randomization: "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World"

## Next Steps

Continue to [Synthetic Data Generation](./synthetic-data.md) to learn about creating training datasets for AI models.