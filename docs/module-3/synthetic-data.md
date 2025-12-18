---
id: synthetic-data
title: Synthetic Data Generation
sidebar_position: 4
description: Comprehensive guide to generating synthetic training data for robotics AI models
keywords: [synthetic data, training datasets, computer vision, deep learning, domain randomization, data augmentation]
---

# Synthetic Data Generation

Synthetic data generation is a critical component of modern AI development in robotics. This chapter covers techniques for creating high-quality synthetic datasets that can be used to train AI models for robotics applications, leveraging simulation environments like Isaac Sim.

## Learning Objectives

- Understand the importance and benefits of synthetic data in robotics
- Learn techniques for generating diverse synthetic datasets
- Implement domain randomization strategies for robust training
- Create multi-modal synthetic data (RGB, depth, segmentation)
- Apply data augmentation techniques for synthetic datasets
- Validate synthetic data quality and effectiveness

## Introduction to Synthetic Data

### Why Synthetic Data?

Synthetic data generation addresses several challenges in robotics AI development:

#### Data Scarcity
- Real-world data collection is expensive and time-consuming
- Rare scenarios difficult to capture in real data
- Safety concerns limit data collection in dangerous environments

#### Data Diversity
- Real-world data may not cover all edge cases
- Weather and lighting conditions vary
- Object appearances and arrangements limited

#### Annotation Quality
- Manual annotation is labor-intensive and error-prone
- 3D annotations particularly challenging
- Consistent labeling across datasets

### Benefits of Synthetic Data

#### Cost Efficiency
- Eliminate expensive data collection campaigns
- Reduce annotation costs significantly
- Enable rapid dataset generation

#### Controlled Environments
- Precise control over scene parameters
- Perfect ground truth annotations
- Repeatable experiments

#### Safety
- Train in dangerous scenarios without risk
- Test edge cases safely
- Validate before real-world deployment

#### Scalability
- Generate unlimited data variations
- Parallel data generation
- Automated annotation processes

## Synthetic Data Generation Pipeline

### Core Components

The synthetic data generation pipeline consists of several key components:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Scene Setup   │───▶│   Randomization  │───▶│   Data Capture  │
│   (Objects,     │    │   (Domain,      │    │   (Cameras,     │
│   Lighting,     │    │   Materials)     │    │   Sensors)      │
│   Environment)  │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Annotation    │───▶│   Post-Processing│───▶│   Dataset       │
│   Generation    │    │   (Augmentation,│    │   Storage       │
│   (Labels,      │    │   Filtering)    │    │                 │
│   Metadata)     │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Scene Setup

Creating realistic and diverse scenes is the foundation of good synthetic data:

```python
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.prims import get_prim_at_path, create_primitive
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.carb import carb_settings_path
import numpy as np
import random
import json

class SceneSetup:
    def __init__(self, world):
        self.world = world
        self.scene_objects = []
        self.lighting_config = {}
        self.materials_library = []

    def setup_basic_environment(self):
        """Setup basic environment components"""
        # Add ground plane
        self.world.scene.add_default_ground_plane(
            prim_path="/World/groundPlane",
            name="ground_plane",
            size=10.0
        )

        # Add sky dome
        sky_dome = self.world.scene.add(
            omni.isaac.core.objects.DomeLight(
                prim_path="/World/SkyDome",
                name="sky_dome",
                intensity=3000,
                color=(0.5, 0.6, 1.0)
            )
        )

        return sky_dome

    def add_objects_to_scene(self, object_configs):
        """Add objects to the scene based on configuration"""
        for config in object_configs:
            obj_type = config.get('type', 'cube')
            position = config.get('position', [0, 0, 0])
            scale = config.get('scale', [1, 1, 1])
            color = config.get('color', [0.8, 0.2, 0.2])

            if obj_type == 'cube':
                obj = self.world.scene.add(
                    omni.isaac.core.objects.DynamicCuboid(
                        prim_path=f"/World/Objects/Object_{len(self.scene_objects)}",
                        name=f"object_{len(self.scene_objects)}",
                        position=position,
                        size=0.2,
                        color=color
                    )
                )
            elif obj_type == 'sphere':
                obj = self.world.scene.add(
                    omni.isaac.core.objects.DynamicSphere(
                        prim_path=f"/World/Objects/Object_{len(self.scene_objects)}",
                        name=f"object_{len(self.scene_objects)}",
                        position=position,
                        radius=0.1,
                        color=color
                    )
                )
            elif obj_type == 'cylinder':
                obj = self.world.scene.add(
                    omni.isaac.core.objects.DynamicCylinder(
                        prim_path=f"/World/Objects/Object_{len(self.scene_objects)}",
                        name=f"object_{len(self.scene_objects)}",
                        position=position,
                        radius=0.1,
                        height=0.2,
                        color=color
                    )
                )

            self.scene_objects.append(obj)

    def setup_camera_rigs(self, camera_configs):
        """Setup camera configurations for data capture"""
        cameras = []

        for i, config in enumerate(camera_configs):
            camera = omni.isaac.sensor.Camera(
                prim_path=f"/World/Cameras/Camera_{i}",
                name=f"camera_{i}",
                position=config['position'],
                look_at=config['look_at'],
                resolution=config['resolution']
            )

            # Set camera parameters
            camera.set_focal_length(config.get('focal_length', 24.0))
            camera.set_horizontal_aperture(config.get('horizontal_aperture', 20.955))
            camera.set_vertical_aperture(config.get('vertical_aperture', 15.29))

            cameras.append({
                'camera': camera,
                'config': config,
                'modalities': config.get('modalities', ['rgb'])
            })

        return cameras

    def create_diverse_scenes(self, num_scenes=100):
        """Create multiple scene variations"""
        scenes = []

        for scene_idx in range(num_scenes):
            # Randomize scene parameters
            num_objects = random.randint(3, 10)
            scene_config = {
                'objects': [],
                'lighting': self.randomize_lighting(),
                'materials': self.randomize_materials(),
                'layout': self.randomize_layout()
            }

            # Generate objects for this scene
            for obj_idx in range(num_objects):
                obj_config = {
                    'type': random.choice(['cube', 'sphere', 'cylinder']),
                    'position': [
                        random.uniform(-3, 3),
                        random.uniform(-3, 3),
                        random.uniform(0.1, 2)
                    ],
                    'scale': [
                        random.uniform(0.05, 0.3),
                        random.uniform(0.05, 0.3),
                        random.uniform(0.05, 0.3)
                    ],
                    'color': [
                        random.uniform(0.1, 1.0),
                        random.uniform(0.1, 1.0),
                        random.uniform(0.1, 1.0)
                    ]
                }
                scene_config['objects'].append(obj_config)

            scenes.append(scene_config)

        return scenes

    def randomize_lighting(self):
        """Randomize lighting conditions"""
        lighting_config = {
            'intensity': random.uniform(1000, 5000),
            'color': (
                random.uniform(0.5, 1.0),
                random.uniform(0.5, 1.0),
                random.uniform(0.5, 1.0)
            ),
            'position': [
                random.uniform(-5, 5),
                random.uniform(-5, 5),
                random.uniform(3, 8)
            ],
            'direction': [
                random.uniform(-1, 1),
                random.uniform(-1, 1),
                random.uniform(-1, 0)
            ]
        }
        return lighting_config

    def randomize_materials(self):
        """Randomize material properties"""
        material_config = {
            'textures': [
                'wood', 'metal', 'fabric', 'plastic', 'glass'
            ],
            'colors': [
                (0.8, 0.2, 0.2),  # Red
                (0.2, 0.8, 0.2),  # Green
                (0.2, 0.2, 0.8),  # Blue
                (0.8, 0.8, 0.2),  # Yellow
                (0.8, 0.2, 0.8),  # Magenta
            ],
            'properties': {
                'roughness': random.uniform(0.0, 1.0),
                'metallic': random.uniform(0.0, 1.0),
                'specular': random.uniform(0.0, 1.0)
            }
        }
        return material_config

    def randomize_layout(self):
        """Randomize object layout in scene"""
        layout_config = {
            'distribution': random.choice(['uniform', 'clustered', 'grid']),
            'spacing': random.uniform(0.1, 0.5),
            'orientation': random.uniform(0, 2 * np.pi)
        }
        return layout_config

def main():
    # Initialize Isaac Sim world
    world = World(stage_units_in_meters=1.0)

    # Setup scene
    scene_setup = SceneSetup(world)
    scene_setup.setup_basic_environment()

    # Create diverse scenes
    scenes = scene_setup.create_diverse_scenes(num_scenes=50)

    # Process each scene
    for i, scene_config in enumerate(scenes):
        print(f"Setting up scene {i+1}/{len(scenes)}")

        # Clear previous objects
        for obj in scene_setup.scene_objects:
            # Remove object (implementation depends on Isaac Sim API)

        # Add objects for current scene
        scene_setup.add_objects_to_scene(scene_config['objects'])

        # Apply lighting configuration
        # Apply material configuration

        # Wait for scene to stabilize
        for _ in range(10):
            world.step(render=True)

    world.stop()

if __name__ == "__main__":
    main()
```

## Domain Randomization Techniques

### Understanding Domain Randomization

Domain randomization is a technique that increases the diversity of training data by varying environmental properties randomly during simulation:

```python
import random
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class DomainRandomizationConfig:
    """Configuration for domain randomization"""
    lighting: Dict[str, Tuple[float, float]] = None
    materials: Dict[str, Tuple[float, float]] = None
    objects: Dict[str, Tuple[float, float]] = None
    camera: Dict[str, Tuple[float, float]] = None
    environment: Dict[str, Tuple[float, float]] = None

class DomainRandomizer:
    def __init__(self, config: DomainRandomizationConfig):
        self.config = config
        self.step_count = 0

    def randomize_lighting(self, stage):
        """Randomize lighting conditions in the scene"""
        if not self.config.lighting:
            return

        # Randomize dome light
        dome_light_path = "/World/SkyDome"
        dome_light = get_prim_at_path(dome_light_path)
        if dome_light:
            intensity_range = self.config.lighting.get('intensity', (1000, 5000))
            intensity = random.uniform(*intensity_range)
            dome_light.GetAttribute("inputs:intensity").Set(intensity)

            color_range = self.config.lighting.get('color', ((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)))
            color = (
                random.uniform(color_range[0][0], color_range[1][0]),
                random.uniform(color_range[0][1], color_range[1][1]),
                random.uniform(color_range[0][2], color_range[1][2])
            )
            dome_light.GetAttribute("inputs:color").Set(color)

        # Randomize directional lights
        for i in range(3):  # Up to 3 directional lights
            light_path = f"/World/DirectionalLight_{i}"
            light = get_prim_at_path(light_path)
            if light:
                intensity = random.uniform(500, 2000)
                light.GetAttribute("inputs:intensity").Set(intensity)

    def randomize_materials(self, stage):
        """Randomize material properties"""
        if not self.config.materials:
            return

        materials_config = self.config.materials

        # Iterate through all prims in the scene
        for prim in stage.Traverse():
            if prim.GetTypeName() in ["Mesh", "Cube", "Sphere", "Cylinder"]:
                # Randomize diffuse color
                if random.random() < materials_config.get('color_variation_prob', 0.5):
                    color_range = materials_config.get('diffuse_color', ((0.1, 0.1, 0.1), (1.0, 1.0, 1.0)))
                    new_color = (
                        random.uniform(color_range[0][0], color_range[1][0]),
                        random.uniform(color_range[0][1], color_range[1][1]),
                        random.uniform(color_range[0][2], color_range[1][2])
                    )

                    # Apply to material (simplified)
                    # In practice, you'd need to find and modify the material shader

                # Randomize roughness
                if random.random() < materials_config.get('roughness_variation_prob', 0.3):
                    roughness_range = materials_config.get('roughness', (0.0, 1.0))
                    new_roughness = random.uniform(*roughness_range)

                # Randomize metallic
                if random.random() < materials_config.get('metallic_variation_prob', 0.2):
                    metallic_range = materials_config.get('metallic', (0.0, 1.0))
                    new_metallic = random.uniform(*metallic_range)

    def randomize_objects(self, stage):
        """Randomize object properties"""
        if not self.config.objects:
            return

        objects_config = self.config.objects

        # Randomize positions of objects
        for prim in stage.Traverse():
            if prim.GetTypeName() in ["Mesh", "Cube", "Sphere", "Cylinder"]:
                if random.random() < objects_config.get('position_variation_prob', 0.7):
                    pos_range = objects_config.get('position', {
                        'x': (-2.0, 2.0),
                        'y': (-2.0, 2.0),
                        'z': (0.1, 2.0)
                    })

                    new_pos = [
                        random.uniform(pos_range['x'][0], pos_range['x'][1]),
                        random.uniform(pos_range['y'][0], pos_range['y'][1]),
                        random.uniform(pos_range['z'][0], pos_range['z'][1])
                    ]

                    # Apply new position
                    xform = UsdGeom.Xformable(prim)
                    xform.ClearXformOpOrder()
                    xform.AddTranslateOp().Set(new_pos)

                # Randomize scale
                if random.random() < objects_config.get('scale_variation_prob', 0.5):
                    scale_range = objects_config.get('scale', (0.5, 2.0))
                    new_scale = random.uniform(*scale_range)

                    # Apply scale transformation
                    scale_op = xform.AddScaleOp()
                    scale_op.Set((new_scale, new_scale, new_scale))

    def randomize_camera(self, cameras):
        """Randomize camera parameters"""
        if not self.config.camera:
            return

        for camera_info in cameras:
            camera = camera_info['camera']

            # Randomize focal length
            focal_range = self.config.camera.get('focal_length', (18.0, 55.0))
            new_focal = random.uniform(*focal_range)
            camera.set_focal_length(new_focal)

            # Randomize aperture
            aperture_range = self.config.camera.get('aperture', (1.4, 16.0))
            new_aperture = random.uniform(*aperture_range)
            camera.set_f_stop(new_aperture)

            # Randomize camera position (with constraints)
            pos_range = self.config.camera.get('position', {
                'x': (-5.0, 5.0),
                'y': (-5.0, 5.0),
                'z': (1.0, 10.0)
            })

            new_pos = [
                random.uniform(pos_range['x'][0], pos_range['x'][1]),
                random.uniform(pos_range['y'][0], pos_range['y'][1]),
                random.uniform(pos_range['z'][0], pos_range['z'][1])
            ]

            # Update camera position
            camera.set_position(new_pos)

    def apply_randomization(self, stage, cameras):
        """Apply all domain randomization effects"""
        self.randomize_lighting(stage)
        self.randomize_materials(stage)
        self.randomize_objects(stage)
        self.randomize_camera(cameras)

        self.step_count += 1

        # Reset randomization after certain steps
        if self.step_count % 50 == 0:  # Reset every 50 steps
            self.reset_randomization()

    def reset_randomization(self):
        """Reset randomization for new episode"""
        self.step_count = 0

def setup_domain_randomization():
    """Setup domain randomization configuration"""

    config = DomainRandomizationConfig(
        lighting={
            'intensity': (1000, 5000),
            'color': ((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
        },
        materials={
            'color_variation_prob': 0.8,
            'roughness_variation_prob': 0.5,
            'metallic_variation_prob': 0.3,
            'diffuse_color': ((0.1, 0.1, 0.1), (1.0, 1.0, 1.0)),
            'roughness': (0.0, 1.0),
            'metallic': (0.0, 1.0)
        },
        objects={
            'position_variation_prob': 0.7,
            'scale_variation_prob': 0.5,
            'position': {
                'x': (-3.0, 3.0),
                'y': (-3.0, 3.0),
                'z': (0.1, 2.0)
            },
            'scale': (0.5, 2.0)
        },
        camera={
            'focal_length': (18.0, 85.0),
            'aperture': (1.4, 16.0),
            'position': {
                'x': (-5.0, 5.0),
                'y': (-5.0, 5.0),
                'z': (1.0, 10.0)
            }
        }
    )

    randomizer = DomainRandomizer(config)
    return randomizer
```

## Multi-Modal Data Generation

### RGB Data Capture

```python
from PIL import Image
import numpy as np
import cv2

class RGBCapture:
    def __init__(self, camera):
        self.camera = camera

    def capture_rgb_image(self):
        """Capture RGB image from camera"""
        try:
            # Get raw RGB data from camera
            rgb_data = self.camera.get_rgb()

            # Convert to PIL Image
            if len(rgb_data.shape) == 3 and rgb_data.shape[2] == 3:
                # Convert from RGB to BGR if needed for OpenCV
                img_pil = Image.fromarray(rgb_data, mode="RGB")
                return img_pil
            else:
                raise ValueError(f"Invalid RGB data shape: {rgb_data.shape}")

        except Exception as e:
            print(f"Error capturing RGB image: {e}")
            return None

    def save_rgb_image(self, filepath, quality=95):
        """Save RGB image to file"""
        img = self.capture_rgb_image()
        if img:
            img.save(filepath, "PNG", quality=quality)
            return True
        return False

    def apply_augmentations(self, image, aug_params=None):
        """Apply data augmentations to RGB image"""
        if aug_params is None:
            aug_params = {
                'brightness': 0.0,
                'contrast': 1.0,
                'saturation': 1.0,
                'hue': 0.0,
                'blur': 0.0,
                'noise': 0.0
            }

        # Convert PIL to numpy for OpenCV operations
        img_array = np.array(image)

        # Apply brightness adjustment
        if aug_params['brightness'] != 0:
            img_array = np.clip(img_array.astype(np.float32) + aug_params['brightness'] * 255, 0, 255).astype(np.uint8)

        # Apply contrast adjustment
        if aug_params['contrast'] != 1.0:
            img_array = np.clip((img_array.astype(np.float32) - 128) * aug_params['contrast'] + 128, 0, 255).astype(np.uint8)

        # Apply saturation (convert to HSV, modify S channel)
        if aug_params['saturation'] != 1.0:
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * aug_params['saturation'], 0, 255).astype(np.uint8)
            img_array = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # Apply blur
        if aug_params['blur'] > 0:
            kernel_size = int(aug_params['blur'] * 2) + 1
            if kernel_size > 1:
                img_array = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), 0)

        # Apply noise
        if aug_params['noise'] > 0:
            noise = np.random.normal(0, aug_params['noise'] * 255, img_array.shape).astype(np.uint8)
            img_array = np.clip(img_array.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Convert back to PIL
        augmented_img = Image.fromarray(img_array)
        return augmented_img
```

### Depth Data Capture

```python
import numpy as np

class DepthCapture:
    def __init__(self, camera):
        self.camera = camera

    def capture_depth_data(self):
        """Capture depth data from camera"""
        try:
            # Get raw depth data from camera
            depth_data = self.camera.get_depth()

            # Validate depth data
            if depth_data is None:
                raise ValueError("No depth data returned from camera")

            # Convert to appropriate format (usually meters)
            depth_array = np.array(depth_data)

            # Handle invalid depth values (often represented as inf or -inf)
            depth_array[np.isinf(depth_array)] = 0  # Set infinite values to 0
            depth_array[depth_array < 0] = 0        # Set negative values to 0

            return depth_array

        except Exception as e:
            print(f"Error capturing depth data: {e}")
            return None

    def save_depth_data(self, filepath):
        """Save depth data to file"""
        depth_data = self.capture_depth_data()
        if depth_data is not None:
            # Save as numpy array
            np.save(filepath, depth_data)
            return True
        return False

    def visualize_depth(self, depth_data, colormap=cv2.COLORMAP_JET):
        """Visualize depth data with color mapping"""
        if depth_data is None:
            return None

        # Normalize depth data for visualization
        normalized_depth = (depth_data - np.min(depth_data)) / (np.max(depth_data) - np.min(depth_data))
        normalized_depth = (normalized_depth * 255).astype(np.uint8)

        # Apply color map
        colored_depth = cv2.applyColorMap(normalized_depth, colormap)

        # Convert to PIL Image
        vis_img = Image.fromarray(colored_depth)
        return vis_img

    def generate_point_cloud(self, depth_data, camera_intrinsics):
        """Generate point cloud from depth data"""
        if depth_data is None:
            return None

        height, width = depth_data.shape
        fx, fy = camera_intrinsics['fx'], camera_intrinsics['fy']
        cx, cy = camera_intrinsics['cx'], camera_intrinsics['cy']

        # Generate coordinate grids
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

        # Convert pixel coordinates to camera coordinates
        x_cam = (x_coords - cx) * depth_data / fx
        y_cam = (y_coords - cy) * depth_data / fy

        # Stack to form point cloud
        points = np.stack([x_cam, y_cam, depth_data], axis=-1)

        # Reshape to (N, 3) format
        points = points.reshape(-1, 3)

        # Remove points with invalid depth
        valid_points = points[~np.isnan(points).any(axis=1)]
        valid_points = valid_points[~np.isinf(valid_points).any(axis=1)]

        return valid_points
```

### Semantic Segmentation Capture

```python
class SemanticSegmentationCapture:
    def __init__(self, camera):
        self.camera = camera
        self.class_mapping = {}  # Maps semantic IDs to class names

    def capture_segmentation_data(self):
        """Capture semantic segmentation data"""
        try:
            # Get semantic segmentation from camera
            seg_data = self.camera.get_semantic_segmentation()

            if seg_data is None:
                raise ValueError("No segmentation data returned from camera")

            # Convert to numpy array
            seg_array = np.array(seg_data)

            return seg_array

        except Exception as e:
            print(f"Error capturing segmentation data: {e}")
            return None

    def save_segmentation_mask(self, filepath, format='png'):
        """Save segmentation mask to file"""
        seg_data = self.capture_segmentation_data()
        if seg_data is not None:
            # Convert to PIL Image (assuming grayscale for class IDs)
            seg_img = Image.fromarray(seg_data.astype(np.uint8), mode="L")
            seg_img.save(filepath, format)
            return True
        return False

    def create_colored_segmentation(self, seg_data, colormap='random'):
        """Create colored visualization of segmentation"""
        if seg_data is None:
            return None

        # Create color mapping for each unique class
        unique_classes = np.unique(seg_data)
        color_map = {}

        if colormap == 'random':
            for class_id in unique_classes:
                if class_id != 0:  # Skip background
                    color_map[class_id] = [
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255)
                    ]
                else:
                    color_map[class_id] = [0, 0, 0]  # Black for background
        else:
            # Use predefined colormap
            for i, class_id in enumerate(unique_classes):
                color_map[class_id] = self.get_predefined_color(i)

        # Create colored image
        height, width = seg_data.shape
        colored_seg = np.zeros((height, width, 3), dtype=np.uint8)

        for class_id in unique_classes:
            mask = (seg_data == class_id)
            colored_seg[mask] = color_map[class_id]

        # Convert to PIL Image
        colored_img = Image.fromarray(colored_seg)
        return colored_img

    def get_predefined_color(self, index):
        """Get predefined color from palette"""
        colors = [
            [0, 0, 0],        # Background
            [128, 0, 0],      # Red
            [0, 128, 0],      # Green
            [128, 128, 0],    # Yellow
            [0, 0, 128],      # Blue
            [128, 0, 128],    # Purple
            [0, 128, 128],    # Cyan
            [128, 128, 128],  # Gray
            [64, 0, 0],       # Dark red
            [192, 0, 0],      # Bright red
            # Add more colors as needed
        ]
        return colors[index % len(colors)]

    def generate_instance_masks(self, seg_data):
        """Generate separate masks for each instance"""
        if seg_data is None:
            return {}

        instance_masks = {}
        unique_instances = np.unique(seg_data)

        for instance_id in unique_instances:
            if instance_id != 0:  # Skip background
                mask = (seg_data == instance_id).astype(np.uint8)
                instance_masks[instance_id] = mask

        return instance_masks
```

## Data Augmentation Techniques

### Geometric Transformations

```python
import cv2
import numpy as np
from PIL import Image

class GeometricAugmentation:
    def __init__(self):
        pass

    def random_rotation(self, image, max_angle=15):
        """Apply random rotation to image"""
        angle = np.random.uniform(-max_angle, max_angle)
        img_array = np.array(image)

        # Get image dimensions
        height, width = img_array.shape[:2]

        # Calculate rotation matrix
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Apply rotation
        rotated_img = cv2.warpAffine(img_array, rotation_matrix, (width, height),
                                     borderMode=cv2.BORDER_REFLECT)

        return Image.fromarray(rotated_img)

    def random_translation(self, image, max_shift_ratio=0.1):
        """Apply random translation to image"""
        img_array = np.array(image)
        height, width = img_array.shape[:2]

        # Calculate max shift in pixels
        max_shift_x = int(width * max_shift_ratio)
        max_shift_y = int(height * max_shift_ratio)

        # Random shifts
        shift_x = np.random.randint(-max_shift_x, max_shift_x + 1)
        shift_y = np.random.randint(-max_shift_y, max_shift_y + 1)

        # Translation matrix
        translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

        # Apply translation
        translated_img = cv2.warpAffine(img_array, translation_matrix, (width, height),
                                        borderMode=cv2.BORDER_REFLECT)

        return Image.fromarray(translated_img)

    def random_scaling(self, image, scale_range=(0.8, 1.2)):
        """Apply random scaling to image"""
        img_array = np.array(image)
        height, width = img_array.shape[:2]

        # Random scale factor
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])

        # Calculate new dimensions
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Resize image
        scaled_img = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Crop or pad to original size
        if scale_factor > 1.0:  # Scale up, crop center
            start_x = (new_width - width) // 2
            start_y = (new_height - height) // 2
            final_img = scaled_img[start_y:start_y + height, start_x:start_x + width]
        else:  # Scale down, pad with reflection
            pad_x = (width - new_width) // 2
            pad_y = (height - new_height) // 2
            final_img = np.pad(scaled_img,
                              ((pad_y, height - new_height - pad_y),
                               (pad_x, width - new_width - pad_x), (0, 0)),
                              mode='reflect')

        return Image.fromarray(final_img)

    def random_shear(self, image, max_shear=0.2):
        """Apply random shearing to image"""
        img_array = np.array(image)
        height, width = img_array.shape[:2]

        # Random shear factors
        shear_x = np.random.uniform(-max_shear, max_shear)
        shear_y = np.random.uniform(-max_shear, max_shear)

        # Define three points before and after transformation
        src_points = np.float32([[0, 0], [width, 0], [0, height]])
        dst_points = np.float32([
            [0, 0],
            [width, shear_y * width],
            [shear_x * height, height]
        ])

        # Calculate shear matrix
        shear_matrix = cv2.getAffineTransform(src_points, dst_points)

        # Apply shear
        sheared_img = cv2.warpAffine(img_array, shear_matrix, (width, height),
                                     borderMode=cv2.BORDER_REFLECT)

        return Image.fromarray(sheared_img)

    def random_flip(self, image, flip_prob=0.5):
        """Apply random flipping to image"""
        if np.random.rand() < flip_prob:
            # Randomly choose between horizontal, vertical, or both
            flip_type = np.random.choice(['horizontal', 'vertical', 'both'])

            img_array = np.array(image)

            if flip_type == 'horizontal':
                flipped_img = cv2.flip(img_array, 1)  # Horizontal flip
            elif flip_type == 'vertical':
                flipped_img = cv2.flip(img_array, 0)  # Vertical flip
            else:  # Both
                flipped_img = cv2.flip(img_array, -1)  # Both flips

            return Image.fromarray(flipped_img)

        return image  # Return original if no flip
```

### Photometric Transformations

```python
class PhotometricAugmentation:
    def __init__(self):
        pass

    def adjust_brightness(self, image, brightness_factor_range=(0.7, 1.3)):
        """Adjust image brightness"""
        factor = np.random.uniform(brightness_factor_range[0], brightness_factor_range[1])

        img_array = np.array(image).astype(np.float32)
        adjusted = img_array * factor

        # Clip values to valid range
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

        return Image.fromarray(adjusted)

    def adjust_contrast(self, image, contrast_factor_range=(0.8, 1.2)):
        """Adjust image contrast"""
        factor = np.random.uniform(contrast_factor_range[0], contrast_factor_range[1])

        img_array = np.array(image).astype(np.float32)

        # Subtract mean, multiply by factor, add mean back
        mean = np.mean(img_array, axis=(0, 1), keepdims=True)
        adjusted = (img_array - mean) * factor + mean

        # Clip values to valid range
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

        return Image.fromarray(adjusted)

    def adjust_saturation(self, image, saturation_factor_range=(0.8, 1.2)):
        """Adjust image saturation"""
        factor = np.random.uniform(saturation_factor_range[0], saturation_factor_range[1])

        img_array = np.array(image).astype(np.float32)

        # Convert to grayscale (take luminance)
        gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
        gray = gray[..., np.newaxis]  # Make it broadcastable

        # Blend original and grayscale
        saturated = img_array * factor + gray * (1 - factor)

        # Clip values to valid range
        saturated = np.clip(saturated, 0, 255).astype(np.uint8)

        return Image.fromarray(saturated)

    def adjust_hue(self, image, hue_delta=0.1):
        """Adjust image hue"""
        delta = np.random.uniform(-hue_delta, hue_delta)

        img_array = np.array(image).astype(np.float32) / 255.0

        # Convert RGB to HSV
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

        # Adjust hue
        hsv[:, :, 0] = (hsv[:, :, 0] + delta) % 1.0

        # Convert back to RGB
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # Convert back to uint8
        adjusted = (rgb * 255).astype(np.uint8)

        return Image.fromarray(adjusted)

    def add_noise(self, image, noise_std_range=(0.0, 0.05)):
        """Add random noise to image"""
        std = np.random.uniform(noise_std_range[0], noise_std_range[1])

        img_array = np.array(image).astype(np.float32)

        # Generate random noise
        noise = np.random.normal(0, std * 255, img_array.shape).astype(np.float32)

        # Add noise
        noisy_img = img_array + noise

        # Clip values to valid range
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

        return Image.fromarray(noisy_img)

    def adjust_gamma(self, image, gamma_range=(0.8, 1.2)):
        """Adjust image gamma"""
        gamma = np.random.uniform(gamma_range[0], gamma_range[1])

        img_array = np.array(image).astype(np.float32)

        # Apply gamma correction
        corrected = 255.0 * np.power(img_array / 255.0, 1.0 / gamma)

        # Clip values to valid range
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)

        return Image.fromarray(corrected)
```

## Data Quality Validation

### Synthetic Data Quality Metrics

```python
import numpy as np
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim
from skimage.feature import canny
import cv2

class SyntheticDataValidator:
    def __init__(self):
        pass

    def assess_realism_score(self, synthetic_img, real_img):
        """Assess how realistic synthetic data appears compared to real data"""
        syn_array = np.array(synthetic_img)
        real_array = np.array(real_img)

        # Structural Similarity Index (SSIM)
        ssim_score = ssim(syn_array, real_array, multichannel=True, data_range=255)

        # Edge density comparison
        syn_edges = canny(cv2.cvtColor(syn_array, cv2.COLOR_RGB2GRAY))
        real_edges = canny(cv2.cvtColor(real_array, cv2.COLOR_RGB2GRAY))

        syn_edge_density = np.sum(syn_edges) / syn_edges.size
        real_edge_density = np.sum(real_edges) / real_edges.size

        edge_similarity = 1 - abs(syn_edge_density - real_edge_density) / max(syn_edge_density, real_edge_density, 1e-6)

        # Color distribution similarity
        syn_hist = [cv2.calcHist([syn_array], [i], None, [256], [0, 256]) for i in range(3)]
        real_hist = [cv2.calcHist([real_array], [i], None, [256], [0, 256]) for i in range(3)]

        hist_similarity = 0
        for i in range(3):
            correlation = cv2.compareHist(syn_hist[i], real_hist[i], cv2.HISTCMP_CORREL)
            hist_similarity += correlation

        hist_similarity /= 3

        # Combine scores (weights can be adjusted)
        realism_score = 0.4 * ssim_score + 0.3 * edge_similarity + 0.3 * hist_similarity

        return {
            'ssim': ssim_score,
            'edge_similarity': edge_similarity,
            'histogram_similarity': hist_similarity,
            'realism_score': realism_score
        }

    def validate_annotation_quality(self, annotations, image_shape):
        """Validate quality of synthetic annotations"""
        validation_results = {
            'valid_objects': 0,
            'invalid_objects': 0,
            'occlusion_issues': 0,
            'annotation_completeness': 0.0
        }

        total_pixels = image_shape[0] * image_shape[1]
        annotated_pixels = 0

        for obj in annotations.get('objects', []):
            bbox = obj.get('bbox', {})
            if bbox:
                min_pt = bbox.get('min', [])
                max_pt = bbox.get('max', [])

                if len(min_pt) >= 2 and len(max_pt) >= 2:
                    width = max_pt[0] - min_pt[0]
                    height = max_pt[1] - min_pt[1]

                    if width > 0 and height > 0:  # Valid bounding box
                        validation_results['valid_objects'] += 1
                        obj_area = width * height
                        annotated_pixels += obj_area

                        # Check for potential occlusion issues
                        if width * height > total_pixels * 0.8:  # Object too large
                            validation_results['occlusion_issues'] += 1
                    else:
                        validation_results['invalid_objects'] += 1
                else:
                    validation_results['invalid_objects'] += 1

        validation_results['annotation_completeness'] = annotated_pixels / total_pixels if total_pixels > 0 else 0

        return validation_results

    def check_domain_gap(self, synthetic_data, real_data_stats):
        """Check the domain gap between synthetic and real data"""
        syn_array = np.array(synthetic_data)

        # Calculate synthetic data statistics
        syn_mean = np.mean(syn_array, axis=(0, 1))
        syn_std = np.std(syn_array, axis=(0, 1))
        syn_median = np.median(syn_array, axis=(0, 1))

        # Compare with real data statistics
        mean_diff = np.abs(syn_mean - real_data_stats['mean'])
        std_diff = np.abs(syn_std - real_data_stats['std'])

        # Calculate domain gap score (lower is better)
        gap_score = np.mean(mean_diff) + np.mean(std_diff)

        return {
            'mean_difference': mean_diff.tolist(),
            'std_difference': std_diff.tolist(),
            'median_difference': np.abs(syn_median - real_data_stats['median']).tolist(),
            'domain_gap_score': float(gap_score)
        }

    def validate_consistency_across_modalities(self, rgb_data, depth_data, seg_data):
        """Validate consistency between different modalities"""
        consistency_report = {
            'rgb_depth_alignment': True,
            'depth_seg_alignment': True,
            'occlusion_consistency': True,
            'overall_consistency_score': 0.0
        }

        if rgb_data is None or depth_data is None or seg_data is None:
            consistency_report['overall_consistency_score'] = 0.0
            return consistency_report

        # Check that all modalities have the same dimensions
        rgb_shape = rgb_data.shape[:2]
        depth_shape = depth_data.shape
        seg_shape = seg_data.shape

        if not (rgb_shape == depth_shape == seg_shape):
            consistency_report['rgb_depth_alignment'] = False
            consistency_report['overall_consistency_score'] = 0.0
            return consistency_report

        # Check depth-segmentation consistency
        # Objects in segmentation should have consistent depth values
        unique_classes = np.unique(seg_data)
        depth_consistency = 0
        total_classes = 0

        for class_id in unique_classes:
            if class_id != 0:  # Skip background
                mask = (seg_data == class_id)
                class_depth_values = depth_data[mask]

                if len(class_depth_values) > 10:  # Need sufficient pixels for statistics
                    class_depth_std = np.std(class_depth_values)
                    class_depth_mean = np.mean(class_depth_values)

                    # Depth should be relatively consistent for the same object
                    if class_depth_std < class_depth_mean * 0.1:  # Threshold can be adjusted
                        depth_consistency += 1
                    total_classes += 1

        if total_classes > 0:
            consistency_report['depth_seg_alignment'] = (depth_consistency / total_classes) > 0.7
        else:
            consistency_report['depth_seg_alignment'] = True

        # Calculate overall consistency score
        alignment_score = 1.0 if consistency_report['rgb_depth_alignment'] else 0.0
        seg_depth_score = 1.0 if consistency_report['depth_seg_alignment'] else 0.0

        consistency_report['overall_consistency_score'] = (alignment_score + seg_depth_score) / 2.0

        return consistency_report
```

## Dataset Organization and Management

### Dataset Structure

```python
import os
import json
import shutil
from pathlib import Path

class SyntheticDatasetManager:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.annotations_path = self.dataset_path / "annotations"
        self.images_path = self.dataset_path / "images"
        self.depth_path = self.dataset_path / "depth"
        self.seg_path = self.dataset_path / "segmentation"

        # Create directory structure
        self._create_directories()

    def _create_directories(self):
        """Create dataset directory structure"""
        dirs_to_create = [
            self.dataset_path,
            self.annotations_path,
            self.images_path,
            self.depth_path,
            self.seg_path
        ]

        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)

    def save_sample(self, sample_id, rgb_image, depth_data, seg_data, annotations):
        """Save a complete data sample"""
        # Save RGB image
        rgb_path = self.images_path / f"{sample_id}.png"
        rgb_image.save(str(rgb_path))

        # Save depth data
        depth_path = self.depth_path / f"{sample_id}.npy"
        np.save(str(depth_path), depth_data)

        # Save segmentation mask
        seg_path = self.seg_path / f"{sample_id}.png"
        seg_img = Image.fromarray(seg_data.astype(np.uint8), mode="L")
        seg_img.save(str(seg_path))

        # Save annotations
        annotation_path = self.annotations_path / f"{sample_id}.json"
        with open(annotation_path, 'w') as f:
            json.dump(annotations, f, indent=2)

    def create_train_val_split(self, split_ratio=0.8):
        """Create train/validation split"""
        all_samples = [f.stem for f in self.annotations_path.glob("*.json")]

        # Shuffle samples
        np.random.shuffle(all_samples)

        split_idx = int(len(all_samples) * split_ratio)
        train_samples = all_samples[:split_idx]
        val_samples = all_samples[split_idx:]

        # Create splits directory
        splits_path = self.dataset_path / "splits"
        splits_path.mkdir(exist_ok=True)

        # Save split information
        split_info = {
            'train': train_samples,
            'val': val_samples,
            'split_ratio': split_ratio
        }

        with open(splits_path / "split_info.json", 'w') as f:
            json.dump(split_info, f, indent=2)

        return split_info

    def generate_dataset_metadata(self):
        """Generate comprehensive dataset metadata"""
        samples = list(self.annotations_path.glob("*.json"))

        metadata = {
            'dataset_name': self.dataset_path.name,
            'total_samples': len(samples),
            'modalities': ['rgb', 'depth', 'segmentation'],
            'created_at': str(np.datetime64('now')),
            'generator': 'Isaac Sim Synthetic Data Generator',
            'version': '1.0',
            'license': 'MIT',
            'statistics': {
                'rgb_formats': [],
                'depth_ranges': {},
                'class_distribution': {}
            }
        }

        # Calculate statistics
        if samples:
            first_sample = samples[0]
            with open(first_sample, 'r') as f:
                first_annotation = json.load(f)

            # Class distribution
            if 'objects' in first_annotation:
                classes = [obj.get('class', 'unknown') for obj in first_annotation['objects']]
                from collections import Counter
                class_counts = Counter(classes)
                metadata['statistics']['class_distribution'] = dict(class_counts)

        # Save metadata
        metadata_path = self.dataset_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return metadata

    def validate_dataset_integrity(self):
        """Validate that all samples have complete data"""
        samples = [f.stem for f in self.annotations_path.glob("*.json")]
        integrity_issues = []

        for sample_id in samples:
            missing_modalities = []

            # Check RGB image
            if not (self.images_path / f"{sample_id}.png").exists():
                missing_modalities.append('rgb')

            # Check depth data
            if not (self.depth_path / f"{sample_id}.npy").exists():
                missing_modalities.append('depth')

            # Check segmentation
            if not (self.seg_path / f"{sample_id}.png").exists():
                missing_modalities.append('segmentation')

            # Check annotation
            if not (self.annotations_path / f"{sample_id}.json").exists():
                missing_modalities.append('annotation')

            if missing_modalities:
                integrity_issues.append({
                    'sample_id': sample_id,
                    'missing_modalities': missing_modalities
                })

        return {
            'total_samples': len(samples),
            'complete_samples': len(samples) - len(integrity_issues),
            'integrity_issues': integrity_issues,
            'integrity_score': (len(samples) - len(integrity_issues)) / len(samples) if samples else 0
        }

def main():
    # Example usage of the synthetic data generation pipeline
    print("Setting up synthetic data generation pipeline...")

    # Initialize dataset manager
    dataset_manager = SyntheticDatasetManager("synthetic_robotics_dataset")

    # Create sample data (in practice, this would come from Isaac Sim)
    for i in range(10):  # Generate 10 sample images
        # Create dummy data for example
        rgb_img = Image.new('RGB', (640, 480), color=(255, 255, 255))
        depth_data = np.random.rand(480, 640).astype(np.float32) * 10.0  # 0-10m depth
        seg_data = np.random.randint(0, 5, size=(480, 640)).astype(np.int32)  # 5 semantic classes

        annotations = {
            'sample_id': f'sample_{i}',
            'timestamp': f'timestamp_{i}',
            'objects': [
                {
                    'class': 'object_1',
                    'bbox': {
                        'min': [50, 50],
                        'max': [100, 100]
                    },
                    'pose': {
                        'position': [1, 2, 3],
                        'orientation': [0, 0, 0, 1]
                    }
                }
            ]
        }

        dataset_manager.save_sample(f'sample_{i}', rgb_img, depth_data, seg_data, annotations)

    # Create train/val split
    split_info = dataset_manager.create_train_val_split(split_ratio=0.8)
    print(f"Created train/val split: {len(split_info['train'])} train, {len(split_info['val'])} val")

    # Generate metadata
    metadata = dataset_manager.generate_dataset_metadata()
    print(f"Generated dataset metadata for {metadata['total_samples']} samples")

    # Validate dataset integrity
    integrity_report = dataset_manager.validate_dataset_integrity()
    print(f"Dataset integrity: {integrity_report['integrity_score']:.2f}")

    print("Synthetic data generation pipeline setup complete!")

if __name__ == "__main__":
    main()
```

## Learning Objectives Review

- Understand the importance and benefits of synthetic data in robotics ✓
- Learn techniques for generating diverse synthetic datasets ✓
- Implement domain randomization strategies for robust training ✓
- Create multi-modal synthetic data (RGB, depth, segmentation) ✓
- Apply data augmentation techniques for synthetic datasets ✓
- Validate synthetic data quality and effectiveness ✓

## Practical Exercise

1. Set up Isaac Sim with domain randomization
2. Create a synthetic dataset with RGB, depth, and segmentation modalities
3. Apply domain randomization techniques to increase dataset diversity
4. Implement data augmentation pipelines
5. Validate the quality of generated synthetic data
6. Organize the dataset according to standard formats

## Assessment Questions

1. Explain the concept of domain randomization and its importance in synthetic data generation.
2. What are the key components of a synthetic data generation pipeline?
3. How do you validate the quality of synthetic datasets?
4. What are the advantages of multi-modal synthetic data over single modality?

## Further Reading

- "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network" - Ledig et al.
- "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World" - Tobin et al.
- "Synthetic Data Generation for End-to-End Thermal Infrared Pedestrian Detection" - Hübner et al.
- NVIDIA Isaac Sim Documentation: https://docs.omniverse.nvidia.com/isaacsim/latest/

## Next Steps

Continue to [Isaac ROS VSLAM](./isaac-vslam.md) to learn about visual SLAM implementation using Isaac ROS.