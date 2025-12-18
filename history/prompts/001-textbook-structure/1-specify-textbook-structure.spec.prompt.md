---
id: 1
title: Specify Textbook Structure
stage: spec
date: 2025-12-18
surface: agent
model: claude-sonnet-4-5-20250929
feature: 001-textbook-structure
branch: 001-textbook-structure
user: Claude Code
command: /sp.specify
labels: [textbook,robotics,specification,docusaurus]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - specs/001-textbook-structure/spec.md
 - specs/001-textbook-structure/checklists/requirements.md
tests:
 - none
---

## Prompt

Specify the complete Physical AI & Humanoid Robotics textbook structure with all pages and components:

FOLDER STRUCTURE:
docs/
├── welcome/
│   ├── index.md (Welcome to Physical AI Textbook)
│   ├── about.md (About This Textbook)
│   ├── contact.md (Contact & Support)
│   └── assessment.md (Assessment Guidelines)
├── introductory-content/
│   ├── week-1.md (Foundations of Physical AI)
│   ├── week-2.md (Embodied Intelligence Principles)
│   └── week-3.md (Humanoid Robotics Landscape)
├── module-1/
│   ├── intro.md (Module 1 Overview)
│   ├── ros2-architecture.md
│   ├── nodes-topics-services.md
│   ├── python-rclpy.md
│   ├── urdf-humanoids.md
│   ├── launch-files.md
│   └── assessment.md
├── module-2/
│   ├── intro.md (Module 2 Overview)
│   ├── gazebo-setup.md
│   ├── physics-simulation.md
│   ├── urdf-sdf.md
│   ├── unity-rendering.md
│   ├── sensor-simulation.md
│   └── assessment.md
├── module-3/
│   ├── intro.md (Module 3 Overview)
│   ├── nvidia-isaac-intro.md
│   ├── isaes - Communication patterns
3. Python with rclpy - Building ROS 2 nodes
4. URDF for Humanoids - Robot description format
5. Launch Files - Parameter management
6. Assessment - Hands-on project

**Module 2: The Digital Twin (Gazebo & Unity)**
Chapters:
1. Gazebo Environment - Setup and basics
2. Physics Simulation - Gravity, collisions, constraints
3. URDF and SDF - Robot modeling
4. Unity Integration - High-fidelity rendering
5. Sensor Simulation - LiDAR, cameras, IMUs
6. Assessment - Simulation project

**Module 3: The AI-Robot Brain (NVIDIA Isaac)**
Chapters:
1. NVIDIA Isaac Intro - Platform overview
2. Isaac Sim - Photorealistic simulation
3. Synthetic Data Generation - Training datasets
4. Isaac ROS VSLAM - Visual SLAM
5. Nav2 Path Planning - Bipedal navigation
6. Sim-to-Real Transfer - Deployment techniques
7. Assessment - Perception pipeline

**Module 4: Vision-Language-Action (VLA)**
Chapters:
1. LLMs + Robotics - Convergence overview
2. Voice-to-Action - OpenAI Whisper integration
3. Cognitive Planning - LLMs for task planning
4. Multimodal Interaction - Speech, gesture, vision
5. Capstone Project - Complete autonomous system
6. Assessment - Final project

UI/UX SPECIFICATIONS:

**Chat Assistant Component:**
- Floating button: 60px circle, #2563eb background, white chat icon
- Position: bottom-right, 24px from edges
- Hover: scale 1.1, shadow increase
- Expanded panel: 400px × 600px, rounded corners
- Header: "Textbook Assistant" in bold, white text on #2563eb
- Close button (×) in top-right
- Chat area: white background, scrollable
- Welcome message: "Hello! I'm your AI-powered textbook assistant. Ask me anything about the current chapter or related robotics concepts."
- Input field: Gray border, rounded, placeholder text
- Send button: Blue, rounded, hover effect
- Smooth slide-in animation

**Color Theme:**
- Primary buttons: #2563eb
- Hover states: #1e40af
- Links: #2563eb, underline on hover
- Code blocks: Dark theme with #1e293b background
- Callouts: Light blue (#dbeafe) background
- Warnings: Orange accent
- Success: Green accent

**Typography:**
- Headings: Inter font, bold
- Body: System font stack
- Code: JetBrains Mono
- Line height: 1.6 for readability

**Responsive Breakpoints:**
- Mobile: < 768px
- Tablet: 768px - 1024px
- Desktop: > 1024px

DOCUSAURUS CONFIGURATION:
- Site title: "Physical AI & Humanoid Robotics"
- Tagline: "Bridging Digital Intelligence with Physical Reality"
- GitHub repo integration
- Algolia search (optional)
- Dark mode toggle
- Sidebar auto-generated from docs
- Blog disabled (focus on textbook)
- Plugin: @docusaurus/plugin-ideal-image

EACH MARKDOWN FILE MUST HAVE:
---
id: unique-id
title: Chapter Title
sidebar_position: number
description: Brief description for SEO
keywords: [keyword1, keyword2]
---

## Learning Objectives
- Objective 1
- Objective 2

## Content sections...

## Code Examples
```python
# Example code with comments
```

## Practical Exercise
Step-by-step tutorial

## Assessment Questions
1. Question 1
2. Question 2

## Further Reading
- Reference 1
- Reference 2

## Response snapshot

Created a comprehensive feature specification for the Physical AI & Humanoid Robotics textbook structure. The specification includes:

- Three prioritized user stories covering the student learning journey, interactive learning support, and content navigation
- 20 functional requirements covering all aspects of the textbook structure, content organization, UI components, accessibility, and responsive design
- Key entities identified (Chapter, Module, Student, Chat Assistant)
- 7 measurable success criteria focused on user experience and learning outcomes
- A quality checklist validating that all requirements are met

The specification organizes the textbook into welcome, introductory content, and 4 modules covering ROS 2, simulation environments, NVIDIA Isaac platform, and vision-language-action systems.

## Outcome

- ✅ Impact: Created complete specification for textbook structure with all required components
- 🧪 Tests: none
- 📁 Files: Created spec.md and requirements.md checklist
- 🔁 Next prompts: Ready for planning phase with /sp.plan
- 🧠 Reflection: Specification captures all user needs and technical requirements for the textbook structure

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A