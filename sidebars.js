// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  textbook: {
    Welcome: [
      'welcome/index',
      'welcome/about',
      'welcome/contact',
      'welcome/assessment'
    ],
    'Introductory Content': [
      'introductory-content/week-1',
      'introductory-content/week-2',
      'introductory-content/week-3'
    ],
    'Module 1': [
      'module-1/intro',
      'module-1/ros2-architecture',
      'module-1/nodes-topics-services',
      'module-1/python-rclpy',
      'module-1/urdf-humanoids',
      'module-1/launch-files',
      'module-1/assessment'
    ],
    'Module 2': [
      'module-2/intro',
      'module-2/gazebo-setup',
      'module-2/physics-simulation',
      'module-2/urdf-sdf',
      'module-2/unity-rendering',
      'module-2/sensor-simulation',
      'module-2/assessment'
    ],
    'Module 3': [
      'module-3/intro',
      'module-3/nvidia-isaac-intro',
      'module-3/isaac-sim',
      'module-3/synthetic-data',
      'module-3/isaac-vslam',
      'module-3/nav2-path-planning',
      'module-3/sim-to-real'
    ]
  },
};

module.exports = sidebars;