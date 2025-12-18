---
id: sim-to-real
title: Sim-to-Real Transfer
sidebar_position: 7
description: Comprehensive guide to transferring AI models and behaviors from simulation to real humanoid robots
keywords: [sim-to-real, transfer learning, robotics simulation, domain adaptation, Isaac Sim, reality gap]
---

# Sim-to-Real Transfer: Bridging Digital and Physical AI

Sim-to-real transfer is a critical capability that enables the deployment of AI models and behaviors developed in simulation onto real humanoid robots. This chapter covers techniques to minimize the "reality gap" and ensure successful transfer of learned behaviors from virtual to physical environments.

## Learning Objectives

- Understand the challenges and solutions in sim-to-real transfer
- Implement domain randomization techniques for robust simulation
- Apply domain adaptation methods for transfer learning
- Create realistic simulation environments that match real conditions
- Validate and verify transferred behaviors on physical robots
- Optimize simulation parameters for better transfer performance

## Introduction to Sim-to-Real Transfer

### The Reality Gap Problem

The "reality gap" refers to the discrepancy between simulation and real-world performance. This gap arises from several factors:

- **Model Imperfections**: Inaccuracies in robot dynamics, kinematics, and sensor models
- **Environmental Differences**: Lighting, textures, friction coefficients, and surface properties
- **Sensor Noise**: Differences in sensor characteristics, noise patterns, and latencies
- **Actuator Dynamics**: Motor responses, gear backlash, and mechanical compliance
- **Physical Phenomena**: Air resistance, electromagnetic interference, and thermal effects

### Sim-to-Real Transfer Approaches

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Sim-to-Real Pipeline                          │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────────┐    │
│  │   Physical  │    │   Domain         │    │   Transfer          │    │
│  │   System    │◄───┤   Randomization  │◄───┤   Learning &        │    │
│  │   Modeling  │    │   & Adaptation   │    │   Validation        │    │
│  └─────────────┘    └──────────────────┘    └─────────────────────┘    │
│         │                       │                       │              │
│         ▼                       ▼                       ▼              │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────────┐    │
│  │   Reality   │    │   Synthetic      │    │   Real Robot        │    │
│  │   Gap       │    │   Data &         │    │   Deployment        │    │
│  │   Analysis  │    │   Domain Shift   │    │   & Testing         │    │
│  └─────────────┘    └──────────────────┘    └─────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Challenges in Sim-to-Real Transfer

1. **Visual Domain Shift**: Differences in lighting, textures, colors, and camera characteristics
2. **Physical Property Differences**: Friction, mass, inertia, and compliance variations
3. **Sensor Noise Characteristics**: Different noise patterns between simulated and real sensors
4. **Temporal Dynamics**: Timing differences in actuator responses and sensor updates
5. **Environmental Factors**: Temperature, humidity, and atmospheric conditions

## Domain Randomization Techniques

### Understanding Domain Randomization

Domain randomization is a technique that increases the diversity of training data by varying environmental properties randomly during simulation:

```python
import numpy as np
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List

@dataclass
class DomainRandomizationConfig:
    """Configuration for domain randomization parameters"""
    lighting: Dict[str, Tuple[float, float]] = None
    materials: Dict[str, Tuple[float, float]] = None
    objects: Dict[str, Tuple[float, float]] = None
    camera: Dict[str, Tuple[float, float]] = None
    environment: Dict[str, Tuple[float, float]] = None
    physics: Dict[str, Tuple[float, float]] = None

class DomainRandomizer:
    def __init__(self, config: DomainRandomizationConfig):
        self.config = config
        self.episode_count = 0

    def randomize_lighting(self):
        """Randomize lighting conditions in the simulation"""
        lighting_config = self.config.lighting
        if not lighting_config:
            return {
                'intensity': 3000,
                'color': [1.0, 1.0, 1.0],
                'position': [5.0, 5.0, 10.0],
                'direction': [-0.5, -0.5, -1.0]
            }

        # Randomize lighting parameters
        intensity = random.uniform(*lighting_config.get('intensity', (1000, 5000)))
        color = [
            random.uniform(*lighting_config.get('color_range', (0.5, 1.0))) for _ in range(3)
        ]
        position = [
            random.uniform(*lighting_config.get('position_range', (-10, 10))) for _ in range(3)
        ]
        direction = [
            random.uniform(*lighting_config.get('direction_range', (-1, 1))) for _ in range(3)
        ]

        return {
            'intensity': intensity,
            'color': color,
            'position': position,
            'direction': direction
        }

    def randomize_materials(self):
        """Randomize material properties"""
        materials_config = self.config.materials
        if not materials_config:
            return {
                'roughness': 0.5,
                'metallic': 0.0,
                'specular': 0.5,
                'albedo': [0.8, 0.8, 0.8]
            }

        roughness = random.uniform(*materials_config.get('roughness_range', (0.0, 1.0)))
        metallic = random.uniform(*materials_config.get('metallic_range', (0.0, 1.0)))
        specular = random.uniform(*materials_config.get('specular_range', (0.0, 1.0)))
        albedo = [
            random.uniform(*materials_config.get('albedo_range', (0.1, 1.0))) for _ in range(3)
        ]

        return {
            'roughness': roughness,
            'metallic': metallic,
            'specular': specular,
            'albedo': albedo
        }

    def randomize_physics_properties(self):
        """Randomize physics properties to account for real-world uncertainties"""
        physics_config = self.config.physics
        if not physics_config:
            return {
                'gravity': [0, 0, -9.81],
                'friction': 0.5,
                'restitution': 0.1,
                'mass_multiplier': 1.0
            }

        gravity = [
            random.uniform(*physics_config.get('gravity_range', (-0.1, 0.1))),
            random.uniform(*physics_config.get('gravity_range', (-0.1, 0.1))),
            random.uniform(*physics_config.get('gravity_range', (-9.9, -9.7)))
        ]

        friction = random.uniform(*physics_config.get('friction_range', (0.1, 1.0)))
        restitution = random.uniform(*physics_config.get('restitution_range', (0.0, 0.5)))
        mass_multiplier = random.uniform(*physics_config.get('mass_range', (0.8, 1.2)))

        return {
            'gravity': gravity,
            'friction': friction,
            'restitution': restitution,
            'mass_multiplier': mass_multiplier
        }

    def randomize_camera_parameters(self):
        """Randomize camera parameters to simulate sensor variations"""
        camera_config = self.config.camera
        if not camera_config:
            return {
                'focal_length': 24.0,
                'sensor_width': 36.0,
                'sensor_height': 24.0,
                'distortion_coefficients': [0, 0, 0, 0, 0]
            }

        focal_length = random.uniform(*camera_config.get('focal_range', (18.0, 55.0)))
        sensor_width = random.uniform(*camera_config.get('sensor_width_range', (35.0, 37.0)))
        sensor_height = random.uniform(*camera_config.get('sensor_height_range', (23.0, 25.0)))

        # Simulate lens distortion
        distortion_coeffs = [
            random.uniform(*camera_config.get('distortion_range', (-0.1, 0.1))) for _ in range(5)
        ]

        return {
            'focal_length': focal_length,
            'sensor_width': sensor_width,
            'sensor_height': sensor_height,
            'distortion_coefficients': distortion_coeffs
        }

    def apply_randomization(self):
        """Apply all domain randomization effects"""
        lighting_params = self.randomize_lighting()
        material_params = self.randomize_materials()
        physics_params = self.randomize_physics_properties()
        camera_params = self.randomize_camera_parameters()

        self.episode_count += 1

        return {
            'lighting': lighting_params,
            'materials': material_params,
            'physics': physics_params,
            'camera': camera_params
        }

def setup_domain_randomization():
    """Setup domain randomization configuration for Isaac Sim"""

    config = DomainRandomizationConfig(
        lighting={
            'intensity_range': (1000, 8000),
            'color_range': (0.3, 1.0),
            'position_range': (-15.0, 15.0),
            'direction_range': (-1.0, 1.0)
        },
        materials={
            'roughness_range': (0.0, 1.0),
            'metallic_range': (0.0, 1.0),
            'specular_range': (0.0, 1.0),
            'albedo_range': (0.1, 1.0)
        },
        physics={
            'gravity_range': (-0.2, 0.2),
            'friction_range': (0.1, 1.5),
            'restitution_range': (0.0, 0.8),
            'mass_range': (0.7, 1.3)
        },
        camera={
            'focal_range': (18.0, 85.0),
            'sensor_width_range': (34.0, 38.0),
            'sensor_height_range': (22.0, 26.0),
            'distortion_range': (-0.3, 0.3)
        }
    )

    randomizer = DomainRandomizer(config)
    return randomizer
```

### Advanced Domain Randomization

```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class DomainRandomizationAugmentation(nn.Module):
    """PyTorch module for domain randomization during training"""

    def __init__(self,
                 brightness_range=(0.8, 1.2),
                 contrast_range=(0.8, 1.2),
                 saturation_range=(0.8, 1.2),
                 hue_range=(-0.1, 0.1),
                 noise_std_range=(0.0, 0.05),
                 blur_kernel_range=(1, 3)):
        super().__init__()
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.noise_std_range = noise_std_range
        self.blur_kernel_range = blur_kernel_range

    def forward(self, images):
        """Apply domain randomization augmentation to batch of images"""
        augmented_images = []

        for image in images:
            # Randomize brightness
            brightness_factor = torch.empty(1).uniform_(*self.brightness_range).item()
            augmented_image = transforms.functional.adjust_brightness(image, brightness_factor)

            # Randomize contrast
            contrast_factor = torch.empty(1).uniform_(*self.contrast_range).item()
            augmented_image = transforms.functional.adjust_contrast(augmented_image, contrast_factor)

            # Randomize saturation
            saturation_factor = torch.empty(1).uniform_(*self.saturation_range).item()
            augmented_image = transforms.functional.adjust_saturation(augmented_image, saturation_factor)

            # Randomize hue
            hue_factor = torch.empty(1).uniform_(*self.hue_range).item()
            augmented_image = transforms.functional.adjust_hue(augmented_image, hue_factor)

            # Add random noise
            noise_std = torch.empty(1).uniform_(*self.noise_std_range).item()
            if noise_std > 0:
                noise = torch.randn_like(augmented_image) * noise_std
                augmented_image = torch.clamp(augmented_image + noise, 0, 1)

            # Apply random blur
            blur_kernel_size = random.randint(*self.blur_kernel_range)
            if blur_kernel_size > 1:
                # Simple average blur (in practice, use Gaussian blur)
                kernel = torch.ones(1, 1, blur_kernel_size, blur_kernel_size) / (blur_kernel_size ** 2)
                # Apply blur to each channel separately
                blurred_channels = []
                for c in range(augmented_image.shape[0]):
                    channel = augmented_image[c:c+1, :, :].unsqueeze(0)  # Add batch dimension
                    blurred_channel = torch.nn.functional.conv2d(channel, kernel, padding=blur_kernel_size//2)
                    blurred_channels.append(blurred_channel.squeeze(0))

                augmented_image = torch.cat(blurred_channels, dim=0)

            augmented_images.append(augmented_image)

        return torch.stack(augmented_images)

class SyntheticToRealDataset(Dataset):
    """Dataset that combines synthetic and real data with domain randomization"""

    def __init__(self, synthetic_data, real_data, domain_randomization_transform=None):
        self.synthetic_data = synthetic_data
        self.real_data = real_data
        self.domain_randomization = domain_randomization_transform
        self.synth_len = len(synthetic_data)
        self.real_len = len(real_data)

    def __len__(self):
        return self.synth_len + self.real_len

    def __getitem__(self, idx):
        if idx < self.synth_len:
            # Synthetic data with domain randomization
            image, label = self.synthetic_data[idx]
            if self.domain_randomization:
                image = self.domain_randomization(image.unsqueeze(0)).squeeze(0)
            domain_label = 0  # Synthetic domain
        else:
            # Real data without randomization
            image, label = self.real_data[idx - self.synth_len]
            domain_label = 1  # Real domain

        return image, label, domain_label  # Return domain label for domain adaptation

def create_domain_adaptation_model(input_dim, hidden_dim, output_dim):
    """Create a model suitable for domain adaptation"""

    class DomainAdaptationModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()

            # Feature extractor (shared between domains)
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.5)
            )

            # Label classifier (task-specific)
            self.label_classifier = nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, output_dim)
            )

            # Domain classifier (for domain adaptation)
            self.domain_classifier = nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 1),
                nn.Sigmoid()
            )

        def forward(self, x, alpha=1.0):
            features = self.feature_extractor(x)

            # Reverse gradient for domain adaptation
            reversed_features = GradientReversalFunction.apply(features, alpha)

            label_output = self.label_classifier(features)
            domain_output = self.domain_classifier(reversed_features)

            return label_output, domain_output

    class GradientReversalFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            ctx.alpha = alpha
            return input

        @staticmethod
        def backward(ctx, grad_output):
            output = grad_output.neg() * ctx.alpha
            return output, None

    return DomainAdaptationModel(input_dim, hidden_dim, output_dim)
```

## Physics Simulation Accuracy

### Accurate Physics Modeling

```python
import numpy as np
from scipy.spatial.transform import Rotation as R
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path

class PhysicsCalibrator:
    """Calibrate physics properties to match real-world behavior"""

    def __init__(self, world: World):
        self.world = world
        self.calibration_data = {}

    def calibrate_robot_dynamics(self, robot_prim_path: str):
        """Calibrate robot dynamics properties to match real robot"""

        # Get robot articulation
        robot_articulation = self.world.scene.get_object(robot_prim_path.split('/')[-1])

        # Calibrate joint properties
        joint_names = robot_articulation.dof_names

        for joint_name in joint_names:
            # Get joint prim
            joint_prim = get_prim_at_path(f"{robot_prim_path}/{joint_name}")

            # Calibrate friction and damping based on real robot data
            self.calibrate_joint_properties(joint_prim)

        # Calibrate link masses and inertias
        link_paths = self.get_robot_link_paths(robot_prim_path)
        for link_path in link_paths:
            link_prim = get_prim_at_path(link_path)
            self.calibrate_link_properties(link_prim)

    def calibrate_joint_properties(self, joint_prim):
        """Calibrate joint friction and damping"""
        # Example calibration values - these would come from real robot measurements
        joint_calibration = {
            'friction': 0.1,  # N*m*s
            'damping': 0.05,   # N*m*s/rad
            'stiction': 0.02   # N*m
        }

        # Apply calibration
        # Note: This is pseudocode - actual Isaac Sim API calls would be different
        # joint_prim.GetAttribute("physics:jointFriction").Set(joint_calibration['friction'])
        # joint_prim.GetAttribute("physics:jointDamping").Set(joint_calibration['damping'])

    def calibrate_link_properties(self, link_prim):
        """Calibrate link mass and inertia properties"""
        # Get real robot mass properties (from CAD model or measurements)
        real_mass = self.get_real_link_mass(link_prim)
        real_inertia = self.get_real_link_inertia(link_prim)

        # Update simulation properties
        # link_prim.GetAttribute("physics:mass").Set(real_mass)
        # link_prim.GetAttribute("physics:diagonalInertia").Set(real_inertia)

    def get_real_link_mass(self, link_prim):
        """Get real link mass from calibration data"""
        link_name = link_prim.GetName()
        if link_name in self.calibration_data:
            return self.calibration_data[link_name].get('mass', 1.0)
        return 1.0  # Default mass

    def get_real_link_inertia(self, link_prim):
        """Get real link inertia from calibration data"""
        link_name = link_prim.GetName()
        if link_name in self.calibration_data:
            return self.calibration_data[link_name].get('inertia', [1.0, 1.0, 1.0])
        return [1.0, 1.0, 1.0]  # Default inertia

    def calibrate_surface_properties(self):
        """Calibrate surface properties (friction, restitution)"""
        # Get common surface materials
        surfaces = [
            'floor', 'table', 'walls', 'objects'
        ]

        for surface in surfaces:
            # Calibrate based on real-world measurements
            surface_properties = self.measure_real_surface_properties(surface)
            self.apply_surface_calibration(surface, surface_properties)

    def measure_real_surface_properties(self, surface_name):
        """Measure real-world surface properties"""
        # This would involve physical experiments
        # For now, return example values
        surface_properties = {
            'floor': {
                'static_friction': 0.7,
                'dynamic_friction': 0.5,
                'restitution': 0.1
            },
            'table': {
                'static_friction': 0.6,
                'dynamic_friction': 0.4,
                'restitution': 0.05
            },
            'walls': {
                'static_friction': 0.8,
                'dynamic_friction': 0.6,
                'restitution': 0.05
            },
            'objects': {
                'static_friction': 0.5,
                'dynamic_friction': 0.3,
                'restitution': 0.2
            }
        }

        return surface_properties.get(surface_name, {
            'static_friction': 0.5,
            'dynamic_friction': 0.3,
            'restitution': 0.1
        })

    def apply_surface_calibration(self, surface_name, properties):
        """Apply surface property calibration to simulation"""
        # Find all prims with this surface material
        # Apply the calibrated properties
        pass

class SensorCalibrator:
    """Calibrate sensor properties to match real sensors"""

    def __init__(self, world: World):
        self.world = world
        self.sensor_specs = {}

    def calibrate_camera_parameters(self, camera_path: str):
        """Calibrate camera intrinsic and extrinsic parameters"""
        # Get real camera calibration
        real_calib = self.get_real_camera_calibration(camera_path)

        # Apply to simulation camera
        camera_prim = get_prim_at_path(camera_path)

        # Update focal length and principal point
        # camera_prim.GetAttribute("focalLength").Set(real_calib['focal_length'])
        # camera_prim.GetAttribute("horizontalAperture").Set(real_calib['aperture_width'])
        # camera_prim.GetAttribute("verticalAperture").Set(real_calib['aperture_height'])

        # Apply distortion coefficients
        # camera_prim.GetAttribute("physics:distortionCoefficientK1").Set(real_calib['k1'])
        # camera_prim.GetAttribute("physics:distortionCoefficientK2").Set(real_calib['k2'])
        # camera_prim.GetAttribute("physics:distortionCoefficientP1").Set(real_calib['p1'])
        # camera_prim.GetAttribute("physics:distortionCoefficientP2").Set(real_calib['p2'])

    def get_real_camera_calibration(self, camera_path):
        """Get real camera calibration parameters"""
        # This would load calibration from file or database
        # For now, return example values
        return {
            'focal_length': 24.0,  # mm
            'aperture_width': 36.0,  # mm
            'aperture_height': 24.0,  # mm
            'k1': -0.17187,  # Radial distortion
            'k2': 0.03843,
            'k3': -0.00076,
            'p1': 0.00031,  # Tangential distortion
            'p2': -0.00014
        }

    def calibrate_imu_properties(self, imu_path: str):
        """Calibrate IMU noise properties"""
        # Get real IMU specifications
        real_spec = self.get_real_imu_specifications(imu_path)

        # Apply noise parameters to simulation
        # This would involve setting noise parameters on the IMU sensor
        pass

    def get_real_imu_specifications(self, imu_path):
        """Get real IMU specifications"""
        # Example specifications for common IMUs
        imu_specs = {
            'imu_link': {
                'accelerometer_noise_density': 0.002,  # m/s^2/√Hz
                'gyroscope_noise_density': 0.0001,     # rad/s/√Hz
                'accelerometer_bias_instability': 0.005,  # m/s^2
                'gyroscope_bias_instability': 0.00001    # rad/s
            }
        }
        return imu_specs.get(imu_path, {
            'accelerometer_noise_density': 0.003,
            'gyroscope_noise_density': 0.0002,
            'accelerometer_bias_instability': 0.008,
            'gyroscope_bias_instability': 0.00002
        })

    def calibrate_lidar_properties(self, lidar_path: str):
        """Calibrate LiDAR properties"""
        # Get real LiDAR specifications
        real_spec = self.get_real_lidar_specifications(lidar_path)

        # Apply to simulation LiDAR
        # lidar_prim = get_prim_at_path(lidar_path)
        # lidar_prim.GetAttribute("physics:rangeMin").Set(real_spec['range_min'])
        # lidar_prim.GetAttribute("physics:rangeMax").Set(real_spec['range_max'])
        # lidar_prim.GetAttribute("physics:noiseStdDev").Set(real_spec['noise_std'])

    def get_real_lidar_specifications(self, lidar_path):
        """Get real LiDAR specifications"""
        lidar_specs = {
            'lidar_sensor': {
                'range_min': 0.1,    # meters
                'range_max': 30.0,   # meters
                'noise_std': 0.01,   # meters
                'angular_resolution': 0.01745,  # radians (1 degree)
                'update_rate': 10.0  # Hz
            }
        }
        return lidar_specs.get(lidar_path, {
            'range_min': 0.2,
            'range_max': 25.0,
            'noise_std': 0.02,
            'angular_resolution': 0.0349,  # 2 degrees
            'update_rate': 5.0
        })
```

## Synthetic Data Generation

### High-Fidelity Synthetic Data

```python
import omni
from omni.isaac.core import World
from omni.isaac.sensor import Camera, LidarRtx
from PIL import Image
import numpy as np
import json
from typing import Dict, List, Tuple
import os

class SyntheticDataManager:
    """Manage synthetic data generation for sim-to-real transfer"""

    def __init__(self, output_dir: str = "synthetic_data"):
        self.output_dir = output_dir
        self.data_counter = 0
        self.annotation_data = []

        # Create output directories
        os.makedirs(f"{output_dir}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/depth", exist_ok=True)
        os.makedirs(f"{output_dir}/seg", exist_ok=True)
        os.makedirs(f"{output_dir}/annotations", exist_ok=True)

    def setup_sensors_for_data_collection(self, world: World):
        """Setup sensors for comprehensive data collection"""

        # Create RGB camera
        self.rgb_camera = Camera(
            prim_path="/World/RGB_Camera",
            name="rgb_camera",
            position=np.array([0.3, 0, 0.8]),
            frequency=30
        )
        self.rgb_camera.add_render_product(resolution=(640, 480))

        # Create depth camera
        self.depth_camera = Camera(
            prim_path="/World/Depth_Camera",
            name="depth_camera",
            position=np.array([0.3, 0, 0.8]),
            frequency=30
        )
        self.depth_camera.add_render_product(resolution=(640, 480))

        # Create semantic segmentation camera
        self.seg_camera = Camera(
            prim_path="/World/Seg_Camera",
            name="seg_camera",
            position=np.array([0.3, 0, 0.8]),
            frequency=30
        )
        self.seg_camera.add_render_product(resolution=(640, 480))

        # Add LiDAR sensor
        self.lidar = LidarRtx(
            prim_path="/World/Lidar",
            name="Lidar",
            translation=np.array([0.2, 0, 0.9]),
            orientation=np.array([0, 0, 0, 1]),
            config="Example_Rotary_Mechanical_Lidar",
            depth_clamp_near=0.1,
            depth_clamp_far=100,
            horizontal_resolution=0.5,
            vertical_resolution=0.25,
            horizontal_lasers=1024,
            vertical_lasers=64,
            max_range=100,
            min_range=0.1,
        )

        # Initialize sensors
        world.scene.add(self.rgb_camera)
        world.scene.add(self.depth_camera)
        world.scene.add(self.seg_camera)
        world.scene.add(self.lidar)

    def capture_multimodal_data(self) -> Dict:
        """Capture synchronized multimodal sensor data"""

        # Capture RGB image
        rgb_data = self.rgb_camera.get_render_product().get_texture()
        rgb_image = Image.fromarray(rgb_data, mode="RGB")

        # Capture depth data
        depth_data = self.depth_camera.get_current_depth()

        # Capture segmentation data
        seg_data = self.seg_camera.get_current_segmentation()

        # Capture LiDAR data
        lidar_data = self.lidar.get_linear_depth_data()

        # Get robot state for annotations
        robot_state = self.get_robot_state()

        # Create annotation
        annotation = {
            'timestamp': self.data_counter,
            'image_path': f"images/frame_{self.data_counter:06d}.png",
            'depth_path': f"depth/depth_{self.data_counter:06d}.npy",
            'seg_path': f"seg/seg_{self.data_counter:06d}.png",
            'lidar_path': f"lidar/lidar_{self.data_counter:06d}.npy",
            'robot_state': robot_state,
            'objects': self.get_scene_objects(),
            'camera_pose': self.get_camera_pose()
        }

        self.annotation_data.append(annotation)

        return {
            'rgb': rgb_image,
            'depth': depth_data,
            'segmentation': seg_data,
            'lidar': lidar_data,
            'annotation': annotation
        }

    def save_multimodal_sample(self, data: Dict):
        """Save a complete multimodal data sample"""

        # Save RGB image
        rgb_path = f"{self.output_dir}/{data['annotation']['image_path']}"
        data['rgb'].save(rgb_path)

        # Save depth data
        depth_path = f"{self.output_dir}/{data['annotation']['depth_path']}"
        np.save(depth_path, data['depth'])

        # Save segmentation
        seg_path = f"{self.output_dir}/{data['annotation']['seg_path']}"
        seg_image = Image.fromarray(data['segmentation'].astype(np.uint8))
        seg_image.save(seg_path)

        # Save LiDAR data
        lidar_path = f"{self.output_dir}/{data['annotation']['lidar_path']}"
        np.save(lidar_path, data['lidar'])

        # Update counter
        self.data_counter += 1

    def get_robot_state(self):
        """Get current robot state for annotation"""
        # This would interface with the robot in the simulation
        # For now, return example state
        return {
            'position': [0.0, 0.0, 0.0],
            'orientation': [0.0, 0.0, 0.0, 1.0],  # quaternion
            'joint_positions': [0.0] * 12,  # Example for 12 DOF
            'joint_velocities': [0.0] * 12
        }

    def get_scene_objects(self):
        """Get information about objects in the scene"""
        # This would scan the USD stage for objects
        # For now, return example objects
        return [
            {
                'name': 'table',
                'class': 'furniture',
                'bbox': {'min': [-1, -0.5, 0], 'max': [1, 0.5, 0.8]},
                'pose': {'position': [0, 0, 0.4], 'orientation': [0, 0, 0, 1]}
            },
            {
                'name': 'cube',
                'class': 'object',
                'bbox': {'min': [0.5, 0.2, 0.1], 'max': [0.7, 0.4, 0.3]},
                'pose': {'position': [0.6, 0.3, 0.2], 'orientation': [0, 0, 0, 1]}
            }
        ]

    def get_camera_pose(self):
        """Get camera pose for annotation"""
        # Get camera transform from USD
        return {
            'position': [0.3, 0, 0.8],
            'orientation': [0.707, 0, 0, 0.707]  # Looking forward
        }

    def generate_dataset(self, num_samples: int = 1000,
                        domain_randomization: bool = True):
        """Generate synthetic dataset with domain randomization"""

        print(f"Generating {num_samples} synthetic samples...")

        for i in range(num_samples):
            if domain_randomization and i % 50 == 0:  # Randomize every 50 samples
                self.apply_domain_randomization()

            # Capture data
            data = self.capture_multimodal_data()

            # Save sample
            self.save_multimodal_sample(data)

            # Progress indicator
            if i % 100 == 0:
                print(f"Generated {i}/{num_samples} samples")

        # Save annotations
        self.save_annotations()

        print(f"Dataset generation complete! Generated {self.data_counter} samples")

    def apply_domain_randomization(self):
        """Apply domain randomization to the scene"""
        # This would randomize lighting, materials, textures, etc.
        # In practice, this would use Isaac Sim's domain randomization tools
        print("Applying domain randomization...")

    def save_annotations(self):
        """Save annotation data to JSON file"""
        annotations_path = f"{self.output_dir}/annotations/dataset_annotations.json"
        with open(annotations_path, 'w') as f:
            json.dump(self.annotation_data, f, indent=2)

class DomainRandomizationManager:
    """Manage domain randomization for synthetic data generation"""

    def __init__(self, world: World):
        self.world = world
        self.randomization_config = self.get_default_config()

    def get_default_config(self) -> Dict:
        """Get default domain randomization configuration"""
        return {
            'lighting': {
                'intensity_range': (1000, 8000),
                'color_temperature_range': (3000, 8000),  # Kelvin
                'position_range': ([-10, -10, 5], [10, 10, 15])
            },
            'materials': {
                'roughness_range': (0.0, 1.0),
                'metallic_range': (0.0, 1.0),
                'albedo_range': ([0.1, 0.1, 0.1], [1.0, 1.0, 1.0])
            },
            'textures': {
                'randomize_albedo': True,
                'randomize_normal': True,
                'randomize_roughness': True,
                'texture_library': [
                    'wood', 'metal', 'plastic', 'fabric', 'stone', 'grass'
                ]
            },
            'environment': {
                'randomize_backgrounds': True,
                'randomize_objects': True,
                'object_count_range': (5, 20),
                'object_size_range': (0.05, 0.5)
            }
        }

    def randomize_scene(self):
        """Apply domain randomization to the current scene"""

        # Randomize lighting
        self.randomize_lighting()

        # Randomize materials
        self.randomize_materials()

        # Randomize textures
        self.randomize_textures()

        # Randomize environment
        self.randomize_environment()

    def randomize_lighting(self):
        """Randomize lighting conditions"""
        # Find all lights in the scene
        stage = omni.usd.get_context().get_stage()

        for prim in stage.Traverse():
            if prim.GetTypeName() in ["DistantLight", "SphereLight", "DiskLight", "DomeLight"]:
                # Randomize intensity
                intensity_range = self.randomization_config['lighting']['intensity_range']
                new_intensity = np.random.uniform(*intensity_range)
                prim.GetAttribute("inputs:intensity").Set(new_intensity)

                # Randomize color (based on temperature)
                temp_range = self.randomization_config['lighting']['color_temperature_range']
                temp = np.random.uniform(*temp_range)
                color = self.temperature_to_rgb(temp)
                prim.GetAttribute("inputs:color").Set(color)

    def temperature_to_rgb(self, kelvin: float) -> Tuple[float, float, float]:
        """Convert temperature in Kelvin to RGB color"""
        # Simplified approximation
        temp = kelvin / 100

        if temp <= 66:
            red = 255
            green = temp
            green = 99.4708025861 * np.log(green) - 161.1195681661
        else:
            red = temp - 60
            red = 329.698727446 * (red ** -0.1332047592)
            green = temp - 60
            green = 288.1221695283 * (green ** -0.0755148492)

        blue = temp - 10
        blue = 138.5177312231 * np.log(blue) - 305.0447927307

        # Clamp values to [0, 255] then convert to [0, 1]
        red = np.clip(red, 0, 255) / 255.0
        green = np.clip(green, 0, 255) / 255.0
        blue = np.clip(blue, 0, 255) / 255.0

        return (red, green, blue)

    def randomize_materials(self):
        """Randomize material properties"""
        # This would iterate through materials and randomize their properties
        # For now, this is a placeholder
        pass

    def randomize_textures(self):
        """Randomize textures on objects"""
        # This would apply random textures to objects
        # For now, this is a placeholder
        pass

    def randomize_environment(self):
        """Randomize environment objects"""
        # This would add/remove/randomize objects in the environment
        # For now, this is a placeholder
        pass

def setup_synthetic_data_pipeline():
    """Setup the complete synthetic data generation pipeline"""

    # Initialize world
    world = World(stage_units_in_meters=1.0)

    # Add ground plane
    world.scene.add_default_ground_plane()

    # Setup domain randomization
    dr_manager = DomainRandomizationManager(world)

    # Setup synthetic data manager
    data_manager = SyntheticDataManager("synthetic_robotics_dataset")
    data_manager.setup_sensors_for_data_collection(world)

    # Add a simple robot for data collection
    add_reference_to_stage(
        usd_path=f"{get_assets_root_path()}/Isaac/Robots/Franka/franka_alt_fingers.usd",
        prim_path="/World/Robot"
    )

    return world, data_manager, dr_manager

def main():
    # Setup the synthetic data pipeline
    world, data_manager, dr_manager = setup_synthetic_data_pipeline()

    # Play the simulation
    world.play()

    # Generate synthetic dataset
    data_manager.generate_dataset(num_samples=500, domain_randomization=True)

    # Stop simulation
    world.stop()

    print("Synthetic data generation complete!")

if __name__ == "__main__":
    main()
```

## Sensor Simulation and Calibration

### Realistic Sensor Simulation

```python
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R
import cv2
from typing import Dict, Tuple, Optional

class RealisticCameraSimulator:
    """Simulate realistic camera behavior with noise and distortions"""

    def __init__(self,
                 width: int = 640,
                 height: int = 480,
                 fov: float = 60.0,  # degrees
                 noise_std: float = 0.01,
                 pixel_size: float = 3.45e-6):  # 3.45 μm for typical camera
        self.width = width
        self.height = height
        self.fov = fov
        self.noise_std = noise_std
        self.pixel_size = pixel_size  # meters per pixel

        # Calculate camera intrinsic matrix
        f = (self.width / 2) / np.tan(np.radians(self.fov / 2))
        self.fx = f
        self.fy = f
        self.cx = self.width / 2
        self.cy = self.height / 2

        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])

        # Distortion coefficients (k1, k2, p1, p2, k3)
        self.distortion_coeffs = np.array([0.1, -0.2, 0.005, -0.005, 0.1])

    def add_shot_noise(self, image: np.ndarray) -> np.ndarray:
        """Add shot noise (photon noise) to image"""
        # Shot noise is Poisson distributed, approximated as Gaussian for high photon counts
        # Variance is proportional to signal intensity
        signal = np.clip(image.astype(np.float32) / 255.0, 0, 1)
        variance = signal * self.noise_std**2
        noise = np.random.normal(0, np.sqrt(variance))
        noisy_image = np.clip(signal + noise, 0, 1) * 255
        return noisy_image.astype(np.uint8)

    def add_read_noise(self, image: np.ndarray) -> np.ndarray:
        """Add read noise to image"""
        read_noise = np.random.normal(0, self.noise_std * 10, image.shape)
        noisy_image = np.clip(image.astype(np.float32) + read_noise, 0, 255)
        return noisy_image.astype(np.uint8)

    def apply_lens_distortion(self, image: np.ndarray) -> np.ndarray:
        """Apply lens distortion to image"""
        h, w = image.shape[:2]

        # Generate coordinate grids
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        points = np.stack([x, y], axis=-1).reshape(-1, 2).astype(np.float32)

        # Normalize points to camera coordinates
        points_norm = (points - np.array([self.cx, self.cy])) / np.array([self.fx, self.fy])

        # Apply distortion
        k1, k2, p1, p2, k3 = self.distortion_coeffs

        r2 = points_norm[:, 0]**2 + points_norm[:, 1]**2
        r4 = r2**2
        r6 = r2**3

        radial_factor = 1 + k1*r2 + k2*r4 + k3*r6
        tangential_x = 2*p1*points_norm[:, 0]*points_norm[:, 1] + p2*(r2 + 2*points_norm[:, 0]**2)
        tangential_y = p1*(r2 + 2*points_norm[:, 1]**2) + 2*p2*points_norm[:, 0]*points_norm[:, 1]

        distorted_points_norm = points_norm.copy()
        distorted_points_norm[:, 0] = points_norm[:, 0] * radial_factor + tangential_x
        distorted_points_norm[:, 1] = points_norm[:, 1] * radial_factor + tangential_y

        # Convert back to pixel coordinates
        distorted_points = distorted_points_norm * np.array([self.fx, self.fy]) + np.array([self.cx, self.cy])

        # Reshape for OpenCV
        distorted_points = distorted_points.reshape(h, w, 2)

        # Remap image
        map_x = distorted_points[:, :, 0].astype(np.float32)
        map_y = distorted_points[:, :, 1].astype(np.float32)

        undistorted_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        return undistorted_image

    def simulate_motion_blur(self, image: np.ndarray, motion_vector: Tuple[float, float]) -> np.ndarray:
        """Simulate motion blur based on motion between frames"""
        dx, dy = motion_vector

        # Calculate kernel size based on motion magnitude
        kernel_size = max(1, int(np.sqrt(dx**2 + dy**2) * 10))
        if kernel_size <= 1:
            return image

        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        if dx != 0 or dy != 0:
            # Normalize motion vector
            length = np.sqrt(dx**2 + dy**2)
            nx, ny = dx/length, dy/length

            # Create line kernel in the direction of motion
            center = kernel_size // 2
            for i in range(kernel_size):
                x = int(center + nx * (i - center))
                y = int(center + ny * (i - center))
                if 0 <= x < kernel_size and 0 <= y < kernel_size:
                    kernel[y, x] = 1

            kernel = kernel / np.sum(kernel)

        # Apply convolution
        if len(image.shape) == 3:
            # Apply to each channel separately
            blurred = np.zeros_like(image)
            for c in range(image.shape[2]):
                blurred[:, :, c] = cv2.filter2D(image[:, :, c], -1, kernel)
        else:
            blurred = cv2.filter2D(image, -1, kernel)

        return blurred

    def add_defocus_blur(self, image: np.ndarray, blur_radius: float) -> np.ndarray:
        """Add defocus blur for out-of-focus regions"""
        if blur_radius <= 0:
            return image

        kernel_size = max(3, int(blur_radius * 10))
        if kernel_size % 2 == 0:
            kernel_size += 1  # Make odd

        # Create circular kernel for defocus blur
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                if dist <= kernel_size // 2:
                    kernel[i, j] = 1

        kernel = kernel / np.sum(kernel)

        if len(image.shape) == 3:
            # Apply to each channel separately
            blurred = np.zeros_like(image)
            for c in range(image.shape[2]):
                blurred[:, :, c] = cv2.filter2D(image[:, :, c], -1, kernel)
        else:
            blurred = cv2.filter2D(image, -1, kernel)

        return blurred

    def simulate_camera_image(self,
                             scene_image: np.ndarray,
                             motion_vector: Tuple[float, float] = (0, 0),
                             focus_distance: float = 1.0) -> np.ndarray:
        """Simulate complete camera pipeline with all effects"""
        # Start with original image
        simulated_image = scene_image.copy()

        # Apply lens distortion
        simulated_image = self.apply_lens_distortion(simulated_image)

        # Add shot noise
        simulated_image = self.add_shot_noise(simulated_image)

        # Add read noise
        simulated_image = self.add_read_noise(simulated_image)

        # Apply motion blur
        simulated_image = self.simulate_motion_blur(simulated_image, motion_vector)

        # Add defocus blur based on focus distance
        # For simplicity, assume blur increases with distance from focus
        blur_radius = max(0, abs(1.0 - focus_distance) * 0.5)
        simulated_image = self.add_defocus_blur(simulated_image, blur_radius)

        return simulated_image

class RealisticLidarSimulator:
    """Simulate realistic LiDAR sensor with noise and dropouts"""

    def __init__(self,
                 max_range: float = 25.0,
                 min_range: float = 0.1,
                 angular_resolution: float = 0.25,  # degrees
                 noise_std: float = 0.01,
                 dropout_probability: float = 0.01):
        self.max_range = max_range
        self.min_range = min_range
        self.angular_resolution = np.radians(angular_resolution)
        self.noise_std = noise_std
        self.dropout_probability = dropout_probability

    def add_measurement_noise(self, ranges: np.ndarray) -> np.ndarray:
        """Add realistic measurement noise to LiDAR ranges"""
        # Add Gaussian noise proportional to distance (more noise at longer ranges)
        noise_factor = 1.0 + ranges / self.max_range  # More noise at longer distances
        noise = np.random.normal(0, self.noise_std * noise_factor, ranges.shape)
        noisy_ranges = ranges + noise

        # Ensure valid range values
        noisy_ranges = np.clip(noisy_ranges, self.min_range, self.max_range)

        return noisy_ranges

    def simulate_beam_divergence(self, ranges: np.ndarray, beam_width: float = 0.002) -> np.ndarray:
        """Simulate beam divergence effect (returns from multiple surfaces in beam)"""
        # In reality, this would require more complex modeling of beam shape
        # For now, we'll add a small amount of variation based on beam width
        beam_effect = np.random.normal(0, beam_width, ranges.shape)
        return ranges + beam_effect

    def simulate_dropouts(self, ranges: np.ndarray) -> np.ndarray:
        """Simulate LiDAR dropouts (missing returns)"""
        # Randomly set some ranges to max range (dropout)
        dropouts = np.random.random(ranges.shape) < self.dropout_probability
        dropout_ranges = ranges.copy()
        dropout_ranges[dropouts] = self.max_range

        return dropout_ranges

    def simulate_multiple_returns(self, ranges: np.ndarray) -> np.ndarray:
        """Simulate multiple returns for partially reflective surfaces"""
        # This is a simplified simulation - in reality, LiDAR can return multiple hits per beam
        # For now, we'll just add some variation
        multiple_returns = np.zeros_like(ranges)

        for i in range(len(ranges)):
            if ranges[i] < self.max_range and np.random.random() < 0.1:  # 10% chance of multiple returns
                # Add a secondary return closer than the primary
                secondary_return = ranges[i] * np.random.uniform(0.7, 0.95)
                multiple_returns[i] = secondary_return
            else:
                multiple_returns[i] = ranges[i]

        return multiple_returns

    def simulate_lidar_scan(self,
                           ground_truth_ranges: np.ndarray,
                           surface_properties: Optional[np.ndarray] = None) -> np.ndarray:
        """Simulate complete LiDAR scan with all realistic effects"""
        simulated_scan = ground_truth_ranges.copy()

        # Add measurement noise
        simulated_scan = self.add_measurement_noise(simulated_scan)

        # Simulate beam divergence
        simulated_scan = self.simulate_beam_divergence(simulated_scan)

        # Simulate dropouts
        simulated_scan = self.simulate_dropouts(simulated_scan)

        # Simulate multiple returns
        simulated_scan = self.simulate_multiple_returns(simulated_scan)

        return simulated_scan

class IMUSimulator:
    """Simulate realistic IMU sensor data with bias and noise"""

    def __init__(self,
                 accel_noise_density: float = 0.002,  # m/s^2 / sqrt(Hz)
                 gyro_noise_density: float = 0.0001,   # rad/s / sqrt(Hz)
                 accel_bias_instability: float = 0.005,  # m/s^2
                 gyro_bias_instability: float = 0.00001,  # rad/s
                 sampling_rate: float = 100.0):  # Hz
        self.accel_noise_density = accel_noise_density
        self.gyro_noise_density = gyro_noise_density
        self.accel_bias_instability = accel_bias_instability
        self.gyro_bias_instability = gyro_bias_instability
        self.sampling_rate = sampling_rate

        # Initialize biases (random walk process)
        self.accel_bias = np.random.normal(0, accel_bias_instability, 3)
        self.gyro_bias = np.random.normal(0, gyro_bias_instability, 3)

        # Allan variance constants for bias random walk
        self.accel_bias_rw = accel_bias_instability / np.sqrt(sampling_rate)
        self.gyro_bias_rw = gyro_bias_instability / np.sqrt(sampling_rate)

    def update_biases(self, dt: float):
        """Update biases using random walk model"""
        # Accelerometer bias random walk
        self.accel_bias += np.random.normal(0, self.accel_bias_rw * np.sqrt(dt), 3)

        # Gyroscope bias random walk
        self.gyro_bias += np.random.normal(0, self.gyro_bias_rw * np.sqrt(dt), 3)

    def simulate_accelerometer(self, true_accel: np.ndarray, dt: float) -> np.ndarray:
        """Simulate accelerometer measurement"""
        # Update biases
        self.update_biases(dt)

        # Add noise (white noise + bias)
        accel_noise = np.random.normal(0, self.accel_noise_density / np.sqrt(dt), 3)

        # Simulate accelerometer measurement
        measured_accel = true_accel + self.accel_bias + accel_noise

        return measured_accel

    def simulate_gyroscope(self, true_angular_velocity: np.ndarray, dt: float) -> np.ndarray:
        """Simulate gyroscope measurement"""
        # Update biases
        self.update_biases(dt)

        # Add noise (white noise + bias)
        gyro_noise = np.random.normal(0, self.gyro_noise_density / np.sqrt(dt), 3)

        # Simulate gyroscope measurement
        measured_gyro = true_angular_velocity + self.gyro_bias + gyro_noise

        return measured_gyro

    def simulate_imu_reading(self,
                           true_state: Dict[str, np.ndarray],
                           dt: float) -> Dict[str, np.ndarray]:
        """Simulate complete IMU reading"""
        # Calculate true values
        true_accel = true_state['linear_acceleration']
        true_gyro = true_state['angular_velocity']

        # Simulate measurements
        accel_measurement = self.simulate_accelerometer(true_accel, dt)
        gyro_measurement = self.simulate_gyroscope(true_gyro, dt)

        return {
            'accelerometer': accel_measurement,
            'gyroscope': gyro_measurement,
            'timestamp': true_state.get('timestamp', 0.0)
        }
```

## Transfer Learning Techniques

### Domain Adaptation Methods

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class DomainAdaptationNetwork(nn.Module):
    """Neural network with domain adaptation capabilities"""

    def __init__(self, input_dim, hidden_dims, output_dim, domain_adaptation=True):
        super().__init__()

        # Feature extractor (shared between domains)
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5)
            ])
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Label classifier (task-specific)
        self.label_classifier = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(prev_dim // 2, output_dim)
        )

        # Domain classifier for adversarial training
        self.domain_adaptation = domain_adaptation
        if domain_adaptation:
            self.domain_classifier = nn.Sequential(
                GradientReversalLayer(),
                nn.Linear(prev_dim, prev_dim // 4),
                nn.ReLU(),
                nn.Linear(prev_dim // 4, 1),
                nn.Sigmoid()
            )

    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)

        label_output = self.label_classifier(features)

        domain_output = None
        if self.domain_adaptation:
            # Apply gradient reversal with alpha parameter
            reversed_features = self.gradient_reversal(features, alpha)
            domain_output = self.domain_classifier(reversed_features)

        return label_output, domain_output

    def gradient_reversal(self, x, alpha):
        """Gradient reversal layer"""
        return GradientReversalFunction.apply(x, alpha)

class GradientReversalFunction(torch.autograd.Function):
    """Gradient reversal function"""

    @staticmethod
    def forward(ctx, input, alpha):
        ctx.alpha = alpha
        return input

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(nn.Module):
    """Gradient reversal layer module"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return GradientReversalFunction.apply(x, 1.0)

class CycleConsistencyNetwork(nn.Module):
    """Network implementing cycle consistency for domain adaptation"""

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()

        # Encoder networks
        self.synth_encoder = self._make_encoder(input_dim, hidden_dim, latent_dim)
        self.real_encoder = self._make_encoder(input_dim, hidden_dim, latent_dim)

        # Decoder networks
        self.synth_decoder = self._make_decoder(latent_dim, hidden_dim, input_dim)
        self.real_decoder = self._make_decoder(latent_dim, hidden_dim, input_dim)

        # Discriminator networks
        self.synth_discriminator = self._make_discriminator(input_dim, hidden_dim)
        self.real_discriminator = self._make_discriminator(input_dim, hidden_dim)

    def _make_encoder(self, input_dim, hidden_dim, latent_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim)
        )

    def _make_decoder(self, latent_dim, hidden_dim, output_dim):
        return nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def _make_discriminator(self, input_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, synth_data, real_data):
        # Encode to latent space
        synth_latent = self.synth_encoder(synth_data)
        real_latent = self.real_encoder(real_data)

        # Decode back to original space
        synth_reconstructed = self.synth_decoder(synth_latent)
        real_reconstructed = self.real_decoder(real_latent)

        # Cross-domain reconstruction
        synth_to_real = self.real_decoder(synth_latent)
        real_to_synth = self.synth_decoder(real_latent)

        # Reconstruct back to original domains
        cycle_synth = self.synth_encoder(real_to_synth)
        cycle_real = self.real_encoder(synth_to_real)

        # Discriminator outputs
        synth_discrim = self.synth_discriminator(synth_data)
        real_discrim = self.real_discriminator(real_data)

        return {
            'synth_reconstructed': synth_reconstructed,
            'real_reconstructed': real_reconstructed,
            'synth_to_real': synth_to_real,
            'real_to_synth': real_to_synth,
            'cycle_synth': cycle_synth,
            'cycle_real': cycle_real,
            'synth_discrim': synth_discrim,
            'real_discrim': real_discrim
        }

class FeatureAlignmentLoss(nn.Module):
    """Loss function for aligning feature distributions between domains"""

    def __init__(self, method='mmd', kernel_type='rbf'):
        super().__init__()
        self.method = method
        self.kernel_type = kernel_type

    def forward(self, source_features, target_features):
        if self.method == 'mmd':
            return self.compute_mmd_loss(source_features, target_features)
        elif self.method == 'correlation':
            return self.compute_correlation_loss(source_features, target_features)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def compute_mmd_loss(self, source, target):
        """Compute Maximum Mean Discrepancy loss"""
        if self.kernel_type == 'rbf':
            return self._rbf_mmd_loss(source, target)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

    def _rbf_mmd_loss(self, source, target):
        """Compute RBF MMD loss"""
        XX = torch.matmul(source, source.t())
        YY = torch.matmul(target, target.t())
        XY = torch.matmul(source, target.t())

        source_norm = torch.diag(XX)
        target_norm = torch.diag(YY)

        source_norm_expand = source_norm.unsqueeze(1).expand(-1, source.size(0))
        target_norm_expand = target_norm.unsqueeze(1).expand(-1, target.size(0))

        XX_expand = source_norm_expand + source_norm_expand.t() - 2 * XX
        YY_expand = target_norm_expand + target_norm_expand.t() - 2 * YY
        XY_expand = source_norm_expand + target_norm_expand.t() - 2 * XY

        gamma = 1.0 / (2 * source.size(1) * torch.mean(XX_expand))

        K_XX = torch.exp(-gamma * XX_expand)
        K_YY = torch.exp(-gamma * YY_expand)
        K_XY = torch.exp(-gamma * XY_expand)

        mmd_loss = torch.mean(K_XX) + torch.mean(K_YY) - 2 * torch.mean(K_XY)

        return mmd_loss

    def compute_correlation_loss(self, source, target):
        """Compute correlation alignment loss"""
        source_mean = torch.mean(source, dim=0, keepdim=True)
        target_mean = torch.mean(target, dim=0, keepdim=True)

        source_centered = source - source_mean
        target_centered = target - target_mean

        source_cov = torch.matmul(source_centered.t(), source_centered) / (source.size(0) - 1)
        target_cov = torch.matmul(target_centered.t(), target_centered) / (target.size(0) - 1)

        # Compute Frobenius norm of difference
        correlation_loss = torch.norm(source_cov - target_cov, p='fro')

        return correlation_loss

class Sim2RealTrainer:
    """Training framework for sim-to-real transfer"""

    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)

        # Loss functions
        self.task_loss_fn = nn.CrossEntropyLoss()
        self.domain_loss_fn = nn.BCELoss()
        self.feature_alignment_loss = FeatureAlignmentLoss(method='mmd')

        # Training metrics
        self.training_history = {
            'task_loss': [],
            'domain_loss': [],
            'alignment_loss': [],
            'total_loss': []
        }

    def train_epoch(self, synth_loader, real_loader, epoch):
        """Train for one epoch with domain adaptation"""
        self.model.train()

        task_losses = []
        domain_losses = []
        alignment_losses = []
        total_losses = []

        # Get iterators for both domains
        synth_iter = iter(synth_loader)
        real_iter = iter(real_loader)

        # Determine the length of the shorter loader
        num_batches = min(len(synth_loader), len(real_loader))

        for _ in range(num_batches):
            try:
                # Get batch from synthetic data
                synth_batch = next(synth_iter)
                synth_data, synth_labels = synth_batch[0].to(self.device), synth_batch[1].to(self.device)

                # Get batch from real data
                real_batch = next(real_iter)
                real_data, real_labels = real_batch[0].to(self.device), real_batch[1].to(self.device)

                # Combine batches for processing
                combined_data = torch.cat([synth_data, real_data], dim=0)
                domain_labels = torch.cat([
                    torch.zeros(synth_data.size(0)),  # Synthetic domain = 0
                    torch.ones(real_data.size(0))     # Real domain = 1
                ]).to(self.device)

                # Compute alpha for gradient reversal (increases during training)
                alpha = 2.0 / (1.0 + np.exp(-10 * epoch / 100)) - 1.0

                # Forward pass
                task_preds, domain_preds = self.model(combined_data, alpha)

                # Task prediction loss (supervised on synthetic, unsupervised on real)
                synth_task_preds = task_preds[:synth_data.size(0)]
                task_loss = self.task_loss_fn(synth_task_preds, synth_labels)

                # Domain classification loss
                domain_loss = self.domain_loss_fn(domain_preds.squeeze(), domain_labels)

                # Feature alignment loss
                synth_features = self.model.feature_extractor(synth_data)
                real_features = self.model.feature_extractor(real_data)
                alignment_loss = self.feature_alignment_loss(synth_features, real_features)

                # Total loss
                total_loss = task_loss + 0.5 * domain_loss + 0.3 * alignment_loss

                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # Store losses
                task_losses.append(task_loss.item())
                domain_losses.append(domain_loss.item())
                alignment_losses.append(alignment_loss.item())
                total_losses.append(total_loss.item())

            except StopIteration:
                break

        # Update metrics
        self.training_history['task_loss'].append(np.mean(task_losses))
        self.training_history['domain_loss'].append(np.mean(domain_losses))
        self.training_history['alignment_loss'].append(np.mean(alignment_losses))
        self.training_history['total_loss'].append(np.mean(total_losses))

        # Update scheduler
        self.scheduler.step()

        print(f"Epoch {epoch}: Task Loss: {np.mean(task_losses):.4f}, "
              f"Domain Loss: {np.mean(domain_losses):.4f}, "
              f"Alignment Loss: {np.mean(alignment_losses):.4f}")

    def evaluate_transfer(self, real_test_loader):
        """Evaluate model performance on real test data"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, labels in real_test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                task_preds, _ = self.model(data)
                _, predicted = torch.max(task_preds.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Transfer accuracy on real test set: {accuracy:.2f}%")
        return accuracy

def train_with_domain_adaptation(synth_dataset, real_dataset, model, epochs=50):
    """Train model with domain adaptation for sim-to-real transfer"""

    # Create data loaders
    synth_loader = DataLoader(synth_dataset, batch_size=32, shuffle=True)
    real_loader = DataLoader(real_dataset, batch_size=32, shuffle=True)
    real_test_loader = DataLoader(real_dataset, batch_size=32, shuffle=False)

    # Initialize trainer
    trainer = Sim2RealTrainer(model)

    # Train with domain adaptation
    for epoch in range(epochs):
        trainer.train_epoch(synth_loader, real_loader, epoch)

    # Evaluate transfer performance
    final_accuracy = trainer.evaluate_transfer(real_test_loader)

    return final_accuracy
```

## Validation and Testing

### Transfer Validation Framework

```python
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import seaborn as sns

class TransferValidator:
    """Validate sim-to-real transfer performance"""

    def __init__(self):
        self.sim_metrics = {}
        self.real_metrics = {}
        self.transfer_gap_analysis = {}

    def validate_behavior_transfer(self,
                                 sim_behavior_data: Dict,
                                 real_behavior_data: Dict,
                                 behavior_type: str = "locomotion") -> Dict:
        """Validate that behaviors transfer correctly from sim to real"""

        if behavior_type == "locomotion":
            return self._validate_locomotion_transfer(sim_behavior_data, real_behavior_data)
        elif behavior_type == "manipulation":
            return self._validate_manipulation_transfer(sim_behavior_data, real_behavior_data)
        elif behavior_type == "navigation":
            return self._validate_navigation_transfer(sim_behavior_data, real_behavior_data)
        else:
            raise ValueError(f"Unknown behavior type: {behavior_type}")

    def _validate_locomotion_transfer(self, sim_data, real_data) -> Dict:
        """Validate locomotion behavior transfer"""
        metrics = {}

        # Compare gait patterns
        sim_stride_lengths = sim_data.get('stride_lengths', [])
        real_stride_lengths = real_data.get('stride_lengths', [])

        if len(sim_stride_lengths) > 0 and len(real_stride_lengths) > 0:
            mean_sim_stride = np.mean(sim_stride_lengths)
            mean_real_stride = np.mean(real_stride_lengths)
            stride_error = abs(mean_sim_stride - mean_real_stride) / mean_sim_stride
            metrics['stride_length_error'] = stride_error

        # Compare step frequencies
        sim_step_freqs = sim_data.get('step_frequencies', [])
        real_step_freqs = real_data.get('step_frequencies', [])

        if len(sim_step_freqs) > 0 and len(real_step_freqs) > 0:
            mean_sim_freq = np.mean(sim_step_freqs)
            mean_real_freq = np.mean(real_step_freqs)
            freq_error = abs(mean_sim_freq - mean_real_freq) / mean_sim_freq
            metrics['step_frequency_error'] = freq_error

        # Compare stability metrics
        sim_com_variances = sim_data.get('com_variances', [])
        real_com_variances = real_data.get('com_variances', [])

        if len(sim_com_variances) > 0 and len(real_com_variances) > 0:
            mean_sim_com_var = np.mean(sim_com_variances)
            mean_real_com_var = np.mean(real_com_variances)
            stability_error = abs(mean_sim_com_var - mean_real_com_var) / mean_sim_com_var
            metrics['stability_error'] = stability_error

        # Overall transfer score (weighted combination of metrics)
        weights = {
            'stride_length_error': 0.3,
            'step_frequency_error': 0.4,
            'stability_error': 0.3
        }

        overall_score = 0
        for metric, weight in weights.items():
            if metric in metrics:
                # Convert error to score (lower error = higher score)
                error = metrics[metric]
                score = max(0, 1 - error)  # Score between 0 and 1
                overall_score += score * weight

        metrics['transfer_score'] = overall_score

        return metrics

    def _validate_manipulation_transfer(self, sim_data, real_data) -> Dict:
        """Validate manipulation behavior transfer"""
        metrics = {}

        # Compare reaching accuracy
        sim_reach_errors = sim_data.get('reaching_errors', [])
        real_reach_errors = real_data.get('reaching_errors', [])

        if len(sim_reach_errors) > 0 and len(real_reach_errors) > 0:
            mean_sim_error = np.mean(sim_reach_errors)
            mean_real_error = np.mean(real_reach_errors)
            reach_accuracy_error = abs(mean_sim_error - mean_real_error) / (mean_sim_error + 1e-6)
            metrics['reaching_accuracy_error'] = reach_accuracy_error

        # Compare grasp success rates
        sim_grasp_success = sim_data.get('grasp_success_rates', [])
        real_grasp_success = real_data.get('grasp_success_rates', [])

        if len(sim_grasp_success) > 0 and len(real_grasp_success) > 0:
            mean_sim_success = np.mean(sim_grasp_success)
            mean_real_success = np.mean(real_grasp_success)
            grasp_success_error = abs(mean_sim_success - mean_real_success) / (mean_sim_success + 1e-6)
            metrics['grasp_success_error'] = grasp_success_error

        # Compare execution time
        sim_execution_times = sim_data.get('execution_times', [])
        real_execution_times = real_data.get('execution_times', [])

        if len(sim_execution_times) > 0 and len(real_execution_times) > 0:
            mean_sim_time = np.mean(sim_execution_times)
            mean_real_time = np.mean(real_execution_times)
            time_error = abs(mean_real_time - mean_sim_time) / (mean_sim_time + 1e-6)
            metrics['execution_time_error'] = time_error

        # Overall transfer score
        weights = {
            'reaching_accuracy_error': 0.4,
            'grasp_success_error': 0.4,
            'execution_time_error': 0.2
        }

        overall_score = 0
        for metric, weight in weights.items():
            if metric in metrics:
                error = metrics[metric]
                score = max(0, 1 - error)
                overall_score += score * weight

        metrics['transfer_score'] = overall_score

        return metrics

    def _validate_navigation_transfer(self, sim_data, real_data) -> Dict:
        """Validate navigation behavior transfer"""
        metrics = {}

        # Compare path efficiency
        sim_path_efficiencies = sim_data.get('path_efficiencies', [])  # Actual path / optimal path
        real_path_efficiencies = real_data.get('path_efficiencies', [])

        if len(sim_path_efficiencies) > 0 and len(real_path_efficiencies) > 0:
            mean_sim_eff = np.mean(sim_path_efficiencies)
            mean_real_eff = np.mean(real_path_efficiencies)
            path_eff_error = abs(mean_real_eff - mean_sim_eff) / (mean_sim_eff + 1e-6)
            metrics['path_efficiency_error'] = path_eff_error

        # Compare success rates
        sim_success_rates = sim_data.get('success_rates', [])
        real_success_rates = real_data.get('success_rates', [])

        if len(sim_success_rates) > 0 and len(real_success_rates) > 0:
            mean_sim_success = np.mean(sim_success_rates)
            mean_real_success = np.mean(real_success_rates)
            success_error = abs(mean_real_success - mean_sim_success) / (mean_sim_success + 1e-6)
            metrics['success_rate_error'] = success_error

        # Compare computation times
        sim_compute_times = sim_data.get('compute_times', [])
        real_compute_times = real_data.get('compute_times', [])

        if len(sim_compute_times) > 0 and len(real_compute_times) > 0:
            mean_sim_compute = np.mean(sim_compute_times)
            mean_real_compute = np.mean(real_compute_times)
            compute_error = abs(mean_real_compute - mean_sim_compute) / (mean_sim_compute + 1e-6)
            metrics['computation_time_error'] = compute_error

        # Overall transfer score
        weights = {
            'path_efficiency_error': 0.3,
            'success_rate_error': 0.4,
            'computation_time_error': 0.3
        }

        overall_score = 0
        for metric, weight in weights.items():
            if metric in metrics:
                error = metrics[metric]
                score = max(0, 1 - error)
                overall_score += score * weight

        metrics['transfer_score'] = overall_score

        return metrics

    def validate_perception_transfer(self, sim_model, real_model,
                                   sim_test_data, real_test_data) -> Dict:
        """Validate that perception models transfer from sim to real"""

        # Get predictions from both models
        sim_predictions = sim_model.predict(sim_test_data)
        real_predictions = real_model.predict(real_test_data)

        # Calculate accuracy on respective test sets
        sim_accuracy = accuracy_score(sim_test_data.labels, sim_predictions)
        real_accuracy = accuracy_score(real_test_data.labels, real_predictions)

        # Calculate domain adaptation metrics
        metrics = {
            'sim_accuracy': sim_accuracy,
            'real_accuracy': real_accuracy,
            'accuracy_gap': abs(sim_accuracy - real_accuracy),
            'transfer_ratio': real_accuracy / (sim_accuracy + 1e-6)
        }

        # Additional metrics for perception
        if hasattr(sim_test_data, 'features') and hasattr(real_test_data, 'features'):
            # Calculate feature distribution similarity
            from scipy.stats import wasserstein_distance

            # Flatten features if they're multi-dimensional
            sim_features_flat = sim_test_data.features.reshape(sim_test_data.features.shape[0], -1)
            real_features_flat = real_test_data.features.reshape(real_test_data.features.shape[0], -1)

            # Calculate Wasserstein distance between feature distributions
            w_distances = []
            for i in range(min(sim_features_flat.shape[1], 10)):  # Limit to first 10 dimensions
                w_dist = wasserstein_distance(
                    sim_features_flat[:, i],
                    real_features_flat[:, i]
                )
                w_distances.append(w_dist)

            metrics['feature_distribution_similarity'] = 1.0 / (1.0 + np.mean(w_distances))

        return metrics

    def analyze_reality_gap(self, sim_data: Dict, real_data: Dict) -> Dict:
        """Analyze the reality gap between simulation and real data"""

        gap_analysis = {}

        # Sensor data comparison
        if 'sensor_data' in sim_data and 'sensor_data' in real_data:
            sim_sensors = sim_data['sensor_data']
            real_sensors = real_data['sensor_data']

            gap_analysis['sensor_gaps'] = {}

            for sensor_type in set(sim_sensors.keys()).intersection(set(real_sensors.keys())):
                sim_values = np.array(sim_sensors[sensor_type])
                real_values = np.array(real_sensors[sensor_type])

                # Calculate mean absolute error
                mae = np.mean(np.abs(sim_values - real_values))

                # Calculate correlation
                if len(sim_values) > 1 and len(real_values) > 1:
                    correlation = np.corrcoef(sim_values.flatten(), real_values.flatten())[0, 1]
                else:
                    correlation = 0

                gap_analysis['sensor_gaps'][sensor_type] = {
                    'mae': mae,
                    'correlation': correlation,
                    'sim_mean': np.mean(sim_values),
                    'real_mean': np.mean(real_values),
                    'relative_error': mae / (np.mean(np.abs(sim_values)) + 1e-6)
                }

        # Control response comparison
        if 'control_responses' in sim_data and 'control_responses' in real_data:
            sim_ctrl = sim_data['control_responses']
            real_ctrl = real_data['control_responses']

            # Calculate control similarity
            ctrl_similarity = self._calculate_control_similarity(sim_ctrl, real_ctrl)
            gap_analysis['control_similarity'] = ctrl_similarity

        # Environmental interaction comparison
        if 'environmental_data' in sim_data and 'environmental_data' in real_data:
            sim_env = sim_data['environmental_data']
            real_env = real_data['environmental_data']

            # Analyze environmental interaction differences
            env_gap = self._analyze_environmental_gap(sim_env, real_env)
            gap_analysis['environmental_gap'] = env_gap

        # Calculate overall reality gap score
        gap_scores = []
        for category, data in gap_analysis.items():
            if isinstance(data, dict) and 'relative_error' in data:
                gap_scores.append(data['relative_error'])
            elif isinstance(data, list):
                # If it's a list of metrics, take the average relative error
                rel_errors = [item.get('relative_error', 0) for item in data if isinstance(item, dict)]
                if rel_errors:
                    gap_scores.extend(rel_errors)

        gap_analysis['overall_reality_gap'] = np.mean(gap_scores) if gap_scores else 0.0
        gap_analysis['transfer_feasibility'] = max(0, 1 - gap_analysis['overall_reality_gap'])

        return gap_analysis

    def _calculate_control_similarity(self, sim_ctrl, real_ctrl) -> Dict:
        """Calculate similarity between control responses"""
        similarities = {}

        # Compare control signals
        if 'command_signals' in sim_ctrl and 'command_signals' in real_ctrl:
            sim_cmds = np.array(sim_ctrl['command_signals'])
            real_cmds = np.array(real_ctrl['command_signals'])

            # Calculate RMSE
            rmse = np.sqrt(np.mean((sim_cmds - real_cmds) ** 2))

            # Calculate correlation
            if len(sim_cmds) > 1 and len(real_cmds) > 1:
                correlation = np.corrcoef(
                    sim_cmds.flatten(),
                    real_cmds.flatten()
                )[0, 1]
            else:
                correlation = 0

            similarities['command_similarity'] = {
                'rmse': rmse,
                'correlation': correlation,
                'relative_error': rmse / (np.mean(np.abs(sim_cmds)) + 1e-6)
            }

        # Compare response times
        if 'response_times' in sim_ctrl and 'response_times' in real_ctrl:
            sim_times = np.array(sim_ctrl['response_times'])
            real_times = np.array(real_ctrl['response_times'])

            time_diff = np.abs(np.mean(real_times) - np.mean(sim_times))
            similarities['timing_similarity'] = {
                'mean_time_diff': time_diff,
                'sim_mean_time': np.mean(sim_times),
                'real_mean_time': np.mean(real_times)
            }

        return similarities

    def _analyze_environmental_gap(self, sim_env, real_env) -> Dict:
        """Analyze differences in environmental interactions"""
        env_gap = {}

        # Compare contact forces
        if 'contact_forces' in sim_env and 'contact_forces' in real_env:
            sim_forces = np.array(sim_env['contact_forces'])
            real_forces = np.array(real_env['contact_forces'])

            force_error = np.mean(np.abs(sim_forces - real_forces))
            env_gap['contact_force_difference'] = {
                'mean_absolute_error': force_error,
                'sim_mean': np.mean(sim_forces),
                'real_mean': np.mean(real_forces),
                'relative_error': force_error / (np.mean(np.abs(sim_forces)) + 1e-6)
            }

        # Compare friction effects
        if 'friction_data' in sim_env and 'friction_data' in real_env:
            sim_friction = np.array(sim_env['friction_data'])
            real_friction = np.array(real_env['friction_data'])

            friction_error = np.mean(np.abs(sim_friction - real_friction))
            env_gap['friction_difference'] = {
                'mean_absolute_error': friction_error,
                'sim_mean': np.mean(sim_friction),
                'real_mean': np.mean(real_friction),
                'relative_error': friction_error / (np.mean(np.abs(sim_friction)) + 1e-6)
            }

        return env_gap

    def generate_validation_report(self, validation_results: Dict) -> str:
        """Generate a comprehensive validation report"""

        report = []
        report.append("# Sim-to-Real Transfer Validation Report\n")
        report.append(f"**Generated**: {np.datetime64('now')}\n")

        # Overall transfer score
        if 'transfer_score' in validation_results:
            score = validation_results['transfer_score']
            report.append(f"## Overall Transfer Score: {score:.3f}")
            report.append(f"**Transfer Quality**: {self._interpret_transfer_score(score)}\n")

        # Detailed metrics
        report.append("## Detailed Analysis\n")

        for key, value in validation_results.items():
            if isinstance(value, dict):
                report.append(f"### {key.replace('_', ' ').title()}")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        report.append(f"- {sub_key.replace('_', ' ').title()}: {sub_value:.4f}")
                    else:
                        report.append(f"- {sub_key.replace('_', ' ').title()}: {sub_value}")
                report.append("")
            elif isinstance(value, (int, float)):
                report.append(f"- {key.replace('_', ' ').title()}: {value:.4f}")

        # Recommendations
        report.append("## Recommendations\n")
        score = validation_results.get('transfer_score', 0)

        if score > 0.8:
            report.append("- Transfer quality is excellent, ready for deployment")
        elif score > 0.6:
            report.append("- Transfer quality is good, minor adjustments may be needed")
        elif score > 0.4:
            report.append("- Transfer quality is moderate, significant calibration required")
        else:
            report.append("- Transfer quality is poor, extensive domain adaptation needed")

        # Improvement suggestions
        reality_gap = validation_results.get('reality_gap_analysis', {}).get('overall_reality_gap', 1.0)
        if reality_gap > 0.5:
            report.append("\n### Suggested Improvements:")
            report.append("- Increase domain randomization in simulation")
            report.append("- Improve physics simulation accuracy")
            report.append("- Calibrate sensor models to match real hardware")
            report.append("- Implement online adaptation techniques")

        return "\n".join(report)

    def _interpret_transfer_score(self, score: float) -> str:
        """Interpret transfer score quality"""
        if score > 0.8:
            return "Excellent"
        elif score > 0.6:
            return "Good"
        elif score > 0.4:
            return "Fair"
        else:
            return "Poor"

class RealityGapAnalyzer:
    """Analyze and quantify the reality gap"""

    def __init__(self):
        self.gap_metrics = {}

    def calculate_reality_gap(self, sim_data: np.ndarray, real_data: np.ndarray) -> Dict:
        """Calculate various metrics for reality gap quantification"""

        gap_metrics = {}

        # Statistical similarity measures
        gap_metrics['mean_difference'] = np.mean(np.abs(sim_data - real_data))
        gap_metrics['std_ratio'] = np.std(real_data) / (np.std(sim_data) + 1e-6)
        gap_metrics['correlation'] = np.corrcoef(sim_data.flatten(), real_data.flatten())[0, 1]

        # Distribution similarity (using KL divergence approximation)
        sim_hist, _ = np.histogram(sim_data, bins=50, density=True)
        real_hist, _ = np.histogram(real_data, bins=50, density=True)

        # Normalize histograms
        sim_hist = sim_hist / (np.sum(sim_hist) + 1e-6)
        real_hist = real_hist / (np.sum(real_hist) + 1e-6)

        # Calculate Jensen-Shannon divergence (symmetric version of KL)
        m = 0.5 * (sim_hist + real_hist)
        js_div = 0.5 * np.sum(sim_hist * np.log((sim_hist + 1e-6) / (m + 1e-6))) + \
                 0.5 * np.sum(real_hist * np.log((real_hist + 1e-6) / (m + 1e-6)))

        gap_metrics['js_divergence'] = js_div

        # Frequency domain analysis
        sim_fft = np.fft.fft(sim_data)
        real_fft = np.fft.fft(real_data)
        freq_similarity = np.mean(np.abs(sim_fft - real_fft)) / (np.mean(np.abs(sim_fft)) + 1e-6)
        gap_metrics['frequency_difference'] = freq_similarity

        # Overall gap score (weighted combination)
        weights = {
            'mean_difference': 0.3,
            'std_ratio_deviation': 0.2,  # How much std ratio deviates from 1.0
            'correlation_inverse': 0.3,   # 1 - correlation
            'js_divergence': 0.2
        }

        std_ratio_dev = abs(gap_metrics['std_ratio'] - 1.0)
        correlation_inverse = 1.0 - abs(gap_metrics['correlation'])  # Inverse of correlation

        overall_gap = (
            weights['mean_difference'] * (gap_metrics['mean_difference'] / (np.mean(np.abs(sim_data)) + 1e-6)) +
            weights['std_ratio_deviation'] * std_ratio_dev +
            weights['correlation_inverse'] * correlation_inverse +
            weights['js_divergence'] * gap_metrics['js_divergence']
        )

        gap_metrics['overall_reality_gap'] = overall_gap
        gap_metrics['transfer_feasibility'] = max(0, 1 - overall_gap)

        return gap_metrics

    def visualize_reality_gap(self, sim_data: np.ndarray, real_data: np.ndarray,
                            title: str = "Reality Gap Analysis"):
        """Create visualization of reality gap"""

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=16)

        # Time series comparison
        axes[0, 0].plot(sim_data[:1000], label='Simulation', alpha=0.7)
        axes[0, 0].plot(real_data[:1000], label='Real', alpha=0.7)
        axes[0, 0].set_title('Time Series Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Histogram comparison
        axes[0, 1].hist(sim_data, bins=50, alpha=0.5, label='Simulation', density=True)
        axes[0, 1].hist(real_data, bins=50, alpha=0.5, label='Real', density=True)
        axes[0, 1].set_title('Distribution Comparison')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Scatter plot
        min_len = min(len(sim_data), len(real_data))
        axes[1, 0].scatter(sim_data[:min_len], real_data[:min_len], alpha=0.5)
        axes[1, 0].plot([sim_data.min(), sim_data.max()], [sim_data.min(), sim_data.max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('Simulation Values')
        axes[1, 0].set_ylabel('Real Values')
        axes[1, 0].set_title('Scatter Plot (Perfect = Diagonal)')
        axes[1, 0].grid(True, alpha=0.3)

        # Power spectral density comparison
        f_sim, psd_sim = plt.mlab.psd(sim_data)
        f_real, psd_real = plt.mlab.psd(real_data)
        axes[1, 1].semilogy(f_sim, psd_sim, label='Simulation', alpha=0.7)
        axes[1, 1].semilogy(f_real, psd_real, label='Real', alpha=0.7)
        axes[1, 1].set_title('Power Spectral Density')
        axes[1, 1].set_xlabel('Frequency')
        axes[1, 1].set_ylabel('PSD')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

def main_validation_example():
    """Example of how to use the validation framework"""

    # Initialize validators
    transfer_validator = TransferValidator()
    reality_analyzer = RealityGapAnalyzer()

    # Example: Validate locomotion behavior transfer
    sim_locomotion_data = {
        'stride_lengths': np.random.normal(0.6, 0.1, 100),
        'step_frequencies': np.random.normal(1.8, 0.2, 100),
        'com_variances': np.random.normal(0.02, 0.005, 100)
    }

    real_locomotion_data = {
        'stride_lengths': np.random.normal(0.58, 0.12, 100),  # Slightly different
        'step_frequencies': np.random.normal(1.75, 0.25, 100),  # Slightly different
        'com_variances': np.random.normal(0.025, 0.008, 100)   # Slightly different
    }

    # Validate behavior transfer
    locomotion_metrics = transfer_validator.validate_behavior_transfer(
        sim_locomotion_data, real_locomotion_data, 'locomotion'
    )

    print("Locomotion Transfer Metrics:")
    for key, value in locomotion_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Example: Analyze reality gap
    sim_sensor_data = np.random.normal(0, 1, 1000)
    real_sensor_data = np.random.normal(0.1, 1.1, 1000)  # Slightly shifted

    gap_metrics = reality_analyzer.calculate_reality_gap(sim_sensor_data, real_sensor_data)

    print("\nReality Gap Metrics:")
    for key, value in gap_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")

    # Generate validation report
    validation_results = {
        'locomotion_transfer': locomotion_metrics,
        'reality_gap_analysis': gap_metrics
    }

    report = transfer_validator.generate_validation_report(validation_results)
    print("\nValidation Report:")
    print(report)

if __name__ == "__main__":
    main_validation_example()
```

## Performance Optimization

### Adaptive Parameter Tuning

```python
import numpy as np
from scipy.optimize import minimize
from typing import Dict, Callable, Any
import time

class AdaptiveParameterTuner:
    """Adaptively tune parameters for sim-to-real transfer"""

    def __init__(self, parameter_bounds: Dict[str, tuple],
                 performance_metric: Callable):
        self.parameter_bounds = parameter_bounds
        self.performance_metric = performance_metric
        self.history = {
            'parameters': [],
            'performance': [],
            'timestamps': []
        }

    def bayesian_optimization_tune(self, initial_params: Dict[str, float],
                                  max_evaluations: int = 50) -> Dict[str, float]:
        """Use Bayesian optimization for parameter tuning"""

        def objective(params_array):
            """Objective function for optimization (negative performance for minimization)"""
            # Convert array back to parameter dictionary
            params = {}
            param_names = list(self.parameter_bounds.keys())
            for i, name in enumerate(param_names):
                params[name] = params_array[i]

            # Evaluate performance
            performance = self.performance_metric(params)

            # Store history
            self.history['parameters'].append(params.copy())
            self.history['performance'].append(performance)
            self.history['timestamps'].append(time.time())

            # Return negative performance (since optimizer minimizes)
            return -performance

        # Initial parameters as array
        initial_array = np.array([initial_params[name] for name in self.parameter_bounds.keys()])

        # Bounds as list of tuples
        bounds_list = [self.parameter_bounds[name] for name in self.parameter_bounds.keys()]

        # Perform optimization
        result = minimize(
            objective,
            initial_array,
            method='L-BFGS-B',
            bounds=bounds_list,
            options={'maxiter': max_evaluations}
        )

        # Convert result back to parameter dictionary
        optimal_params = {}
        param_names = list(self.parameter_bounds.keys())
        for i, name in enumerate(param_names):
            optimal_params[name] = result.x[i]

        return optimal_params

    def evolutionary_tune(self, population_size: int = 20, generations: int = 30) -> Dict[str, float]:
        """Use evolutionary algorithm for parameter tuning"""

        # Initialize population
        param_names = list(self.parameter_bounds.keys())
        population = []

        for _ in range(population_size):
            individual = {}
            for name in param_names:
                min_val, max_val = self.parameter_bounds[name]
                individual[name] = np.random.uniform(min_val, max_val)
            population.append(individual)

        # Evolution loop
        for gen in range(generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                performance = self.performance_metric(individual)
                fitness_scores.append(performance)

                # Store in history
                self.history['parameters'].append(individual.copy())
                self.history['performance'].append(performance)
                self.history['timestamps'].append(time.time())

            # Select parents (tournament selection)
            selected_indices = []
            for _ in range(population_size):
                tournament_size = 3
                tournament_indices = np.random.choice(len(population), tournament_size)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                selected_indices.append(winner_idx)

            # Create new population through crossover and mutation
            new_population = []
            for i in range(0, population_size, 2):
                parent1 = population[selected_indices[i]]
                parent2 = population[selected_indices[i + 1]] if i + 1 < len(selected_indices) else population[selected_indices[0]]

                # Crossover
                child1, child2 = self._crossover(parent1, parent2)

                # Mutation
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)

                new_population.extend([child1, child2])

            population = new_population[:population_size]

        # Return best individual
        final_fitness = []
        for individual in population:
            final_fitness.append(self.performance_metric(individual))

        best_idx = np.argmax(final_fitness)
        return population[best_idx]

    def _crossover(self, parent1: Dict[str, float], parent2: Dict[str, float]) -> tuple:
        """Perform crossover between two parents"""
        child1, child2 = {}, {}
        param_names = list(self.parameter_bounds.keys())

        for name in param_names:
            # Blend values
            alpha = np.random.uniform(0.3, 0.7)
            child1[name] = alpha * parent1[name] + (1 - alpha) * parent2[name]
            child2[name] = (1 - alpha) * parent1[name] + alpha * parent2[name]

            # Ensure bounds
            min_val, max_val = self.parameter_bounds[name]
            child1[name] = np.clip(child1[name], min_val, max_val)
            child2[name] = np.clip(child2[name], min_val, max_val)

        return child1, child2

    def _mutate(self, individual: Dict[str, float], mutation_rate: float = 0.1) -> Dict[str, float]:
        """Apply mutation to an individual"""
        mutated = individual.copy()
        param_names = list(self.parameter_bounds.keys())

        for name in param_names:
            if np.random.random() < mutation_rate:
                min_val, max_val = self.parameter_bounds[name]
                # Add Gaussian noise
                noise = np.random.normal(0, (max_val - min_val) * 0.1)
                mutated[name] = np.clip(individual[name] + noise, min_val, max_val)

        return mutated

    def get_tuning_insights(self) -> Dict[str, Any]:
        """Get insights from the tuning process"""

        if not self.history['performance']:
            return {}

        insights = {
            'best_performance': max(self.history['performance']),
            'worst_performance': min(self.history['performance']),
            'average_performance': np.mean(self.history['performance']),
            'performance_std': np.std(self.history['performance']),
            'tuning_stability': self._calculate_tuning_stability(),
            'convergence_rate': self._calculate_convergence_rate()
        }

        return insights

    def _calculate_tuning_stability(self) -> float:
        """Calculate how stable the tuning process was"""
        if len(self.history['performance']) < 2:
            return 0.0

        # Measure stability as inverse of performance volatility
        performance_changes = np.diff(self.history['performance'])
        volatility = np.std(performance_changes)

        # Normalize to [0, 1] where 1 is most stable
        return max(0, 1 - volatility)

    def _calculate_convergence_rate(self) -> float:
        """Calculate how quickly tuning converged to good parameters"""
        if len(self.history['performance']) < 10:
            return 0.0

        # Calculate improvement in first half vs second half
        mid_point = len(self.history['performance']) // 2
        first_half_best = max(self.history['performance'][:mid_point])
        second_half_best = max(self.history['performance'][mid_point:])

        # If second half has better performance, it's converging
        if second_half_best > first_half_best:
            improvement = (second_half_best - first_half_best) / (first_half_best + 1e-6)
            return min(1.0, improvement * 10)  # Scale and cap at 1.0
        else:
            return 0.0

class OnlineAdaptationSystem:
    """Online adaptation system for continuous parameter adjustment"""

    def __init__(self, base_params: Dict[str, float], adaptation_rate: float = 0.01):
        self.base_params = base_params.copy()
        self.current_params = base_params.copy()
        self.adaptation_rate = adaptation_rate
        self.performance_history = []
        self.param_history = []

    def update_parameters(self, current_performance: float,
                         reference_performance: float = 1.0) -> Dict[str, float]:
        """Update parameters based on performance feedback"""

        # Store performance
        self.performance_history.append(current_performance)

        # Calculate performance ratio
        perf_ratio = current_performance / (reference_performance + 1e-6)

        # Adaptive adjustment based on performance
        if perf_ratio < 0.8:  # Performance is poor
            # Aggressive adaptation
            adaptation_factor = min(1.0, (1.0 - perf_ratio) * 2.0)
        elif perf_ratio < 0.95:  # Performance is acceptable but not great
            # Moderate adaptation
            adaptation_factor = (1.0 - perf_ratio) * 1.0
        else:  # Performance is good
            # Conservative adaptation
            adaptation_factor = max(0.1, (1.0 - perf_ratio) * 0.5)

        # Adjust parameters based on performance
        for param_name, base_value in self.base_params.items():
            # Calculate gradient based on recent performance changes
            if len(self.performance_history) > 1:
                recent_perf_change = (
                    self.performance_history[-1] -
                    self.performance_history[-min(5, len(self.performance_history))]
                ) / max(1, len(self.performance_history) - 1)

                # Adjust parameter in direction that improves performance
                adjustment = self.adaptation_rate * adaptation_factor * recent_perf_change
                new_value = base_value + adjustment

                # Apply bounds (assume reasonable bounds based on parameter type)
                if param_name.endswith('_gain') or param_name.endswith('_weight'):
                    # For gains and weights, constrain to positive values
                    new_value = max(0.01, min(10.0, new_value))
                elif param_name.endswith('_threshold'):
                    # For thresholds, constrain to reasonable range
                    new_value = max(0.001, min(1.0, new_value))
                else:
                    # For other parameters, allow more flexibility
                    new_value = max(-10.0, min(10.0, new_value))

                self.current_params[param_name] = new_value

        # Store current parameters
        self.param_history.append(self.current_params.copy())

        return self.current_params.copy()

    def get_adaptation_metrics(self) -> Dict[str, float]:
        """Get metrics about the adaptation process"""

        if len(self.performance_history) < 2:
            return {}

        # Calculate performance trend
        recent_perf = self.performance_history[-min(10, len(self.performance_history)):]
        perf_trend = np.polyfit(range(len(recent_perf)), recent_perf, 1)[0] if len(recent_perf) > 1 else 0

        # Calculate parameter stability
        if len(self.param_history) > 1:
            param_changes = []
            for i in range(1, len(self.param_history)):
                for param_name in self.base_params.keys():
                    change = abs(
                        self.param_history[i][param_name] -
                        self.param_history[i-1][param_name]
                    ) / (abs(self.base_params[param_name]) + 1e-6)
                    param_changes.append(change)

            avg_param_change = np.mean(param_changes) if param_changes else 0
        else:
            avg_param_change = 0

        metrics = {
            'performance_trend': perf_trend,
            'average_param_change': avg_param_change,
            'adaptation_stability': max(0, 1 - avg_param_change * 10),  # Inverse of parameter change
            'current_performance': self.performance_history[-1] if self.performance_history else 0,
            'best_performance': max(self.performance_history) if self.performance_history else 0
        }

        return metrics

def example_performance_metric(params: Dict[str, float]) -> float:
    """Example performance metric function"""
    # Simulate a performance function that has an optimum
    # This is just an example - in reality, this would involve running the actual system
    x = params.get('param1', 0.5)
    y = params.get('param2', 0.3)

    # Example function with optimum at (0.5, 0.3)
    performance = 1.0 - ((x - 0.5)**2 + (y - 0.3)**2) * 10
    return max(0, min(1, performance))  # Clamp to [0, 1]

def main_optimization_example():
    """Example of parameter optimization"""

    # Define parameter bounds
    param_bounds = {
        'param1': (0.0, 1.0),
        'param2': (0.0, 1.0),
        'param3': (0.1, 2.0)
    }

    # Initialize tuner
    tuner = AdaptiveParameterTuner(param_bounds, example_performance_metric)

    # Initial parameters
    initial_params = {
        'param1': 0.3,
        'param2': 0.4,
        'param3': 1.0
    }

    # Bayesian optimization
    print("Starting Bayesian optimization...")
    optimal_params = tuner.bayesian_optimization_tune(initial_params, max_evaluations=20)

    print(f"Optimal parameters: {optimal_params}")

    # Get insights
    insights = tuner.get_tuning_insights()
    print(f"Tuning insights: {insights}")

    # Example of online adaptation
    print("\nStarting online adaptation...")
    online_adapter = OnlineAdaptationSystem(initial_params, adaptation_rate=0.01)

    # Simulate performance feedback loop
    for step in range(20):
        # Simulate current performance (would come from real system)
        current_performance = example_performance_metric(online_adapter.current_params)

        # Update parameters based on performance
        updated_params = online_adapter.update_parameters(current_performance)

        if step % 5 == 0:
            print(f"Step {step}: Performance = {current_performance:.4f}, Params = {updated_params}")

    # Get adaptation metrics
    metrics = online_adapter.get_adaptation_metrics()
    print(f"\nAdaptation metrics: {metrics}")

if __name__ == "__main__":
    main_optimization_example()