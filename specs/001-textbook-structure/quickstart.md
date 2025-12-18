# Quickstart Guide: Physical AI & Humanoid Robotics Textbook

## Prerequisites

Before starting with the Physical AI & Humanoid Robotics textbook project, ensure you have:

- Node.js version 18 or higher
- npm or yarn package manager
- Git for version control
- A modern web browser (Chrome, Firefox, Safari, or Edge)
- Basic knowledge of Markdown and Git

## Getting Started

### 1. Clone the Repository

```bash
git clone <repository-url>
cd book-ai
```

### 2. Install Dependencies

```bash
npm install
```

This will install all necessary dependencies including Docusaurus, React, Tailwind CSS, and other development tools.

### 3. Run the Development Server

```bash
npm start
```

This command starts a local development server and opens the textbook in your default browser at `http://localhost:3000`. The site will automatically reload when you make changes to the content.

## Project Structure Overview

The textbook project follows this structure:

```
book-ai/
├── docs/                    # All textbook content
│   ├── welcome/            # Welcome section pages
│   ├── introductory-content/ # Introductory content
│   ├── module-1/           # Module 1: ROS 2 content
│   ├── module-2/           # Module 2: Simulation content
│   ├── module-3/           # Module 3: NVIDIA Isaac content
│   └── module-4/           # Module 4: Vision-Language-Action content
├── src/                    # Custom React components
│   ├── components/         # Reusable UI components
│   ├── css/               # Custom styles
│   └── pages/             # Custom pages
├── static/                 # Static assets (images, etc.)
├── docusaurus.config.js    # Docusaurus configuration
├── sidebars.js            # Navigation sidebar configuration
└── package.json           # Project dependencies and scripts
```

## Adding New Content

### Creating a New Chapter

1. Create a new Markdown file in the appropriate module directory:

```bash
# Example: Adding a new chapter to Module 1
touch docs/module-1/new-concept.md
```

2. Add the required frontmatter to your new chapter:

```markdown
---
id: new-concept
title: New Robotics Concept
sidebar_position: 5
description: Learn about new robotics concepts and implementations
keywords: [robotics, concept, ai]
---

## Learning Objectives

- Understand the fundamental principles of this new concept
- Apply the concept in practical scenarios

## Content Section

Your chapter content goes here...

## Code Examples

```python
# Python code example with ROS 2
import rclpy
from rclpy.node import Node

class NewConceptNode(Node):
    def __init__(self):
        super().__init__('new_concept_node')
        self.get_logger().info('New concept node started')

def main():
    rclpy.init()
    node = NewConceptNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Practical Exercise

Step-by-step exercise instructions...

## Assessment Questions

1. Question about the concept
2. Another question about the concept

## Further Reading

- Reference 1
- Reference 2
```

3. Update the sidebar configuration in `sidebars.js` to include your new chapter:

```javascript
module.exports = {
  textbook: {
    Welcome: ['welcome/index', 'welcome/about', 'welcome/assessment'],
    'Introductory Content': ['introductory-content/week-1', /* ... */],
    'Module 1': [
      'module-1/intro',
      'module-1/ros2-architecture',
      // ... other chapters
      'module-1/new-concept',  // Add your new chapter here
      'module-1/assessment'
    ],
    // ... other modules
  },
};
```

## Using Custom Components

### Chat Assistant Component

The floating chat assistant is available on all pages. It appears in the bottom-right corner and provides help about textbook content. The component is built with React and TypeScript, and styled with the blue theme colors.

### Module Cards

To display module cards on a page, you can use the ModuleCard component in MDX files:

```mdx
import ModuleCard from '@site/src/components/Modules/ModuleCard';

<ModuleCard
  title="Module 1: ROS 2 Fundamentals"
  description="Learn the basics of ROS 2 architecture, nodes, topics, and services"
  icon="🤖"
  link="/docs/module-1/intro"
/>
```

## Theming and Styling

### Color Theme

The textbook uses a blue color theme as specified:

- Primary: #2563eb (bright blue)
- Secondary: #1e40af (darker blue)
- Accent: #60a5fa (light blue)
- Background: #f8fafc (light gray)
- Text: #1e293b (dark gray)

### Custom Styles

Custom styles are defined in `src/css/custom.css` and extend the default Docusaurus theme. You can modify colors, fonts, and other styling by updating this file.

## Building for Production

To build the textbook for production deployment:

```bash
npm run build
```

This creates a `build/` directory with a production-ready version of the textbook that can be deployed to any static hosting service.

## Deployment

The textbook is configured for GitHub Pages deployment:

1. Ensure your repository is connected to GitHub
2. Run the deployment command:

```bash
npm run deploy
```

This builds the project and pushes the static files to the `gh-pages` branch for GitHub Pages hosting.

## Troubleshooting

### Common Issues

**Development server won't start**: Ensure Node.js and npm are properly installed and in your PATH.

**Changes not reflecting**: Make sure you're running `npm start` and not just opening the HTML files directly in the browser.

**Images not loading**: Place images in the `static/img/` directory and reference them with `/img/filename.jpg` path.

**Build fails**: Run `npm run build` to see detailed error messages and fix any syntax issues in your Markdown files.

## Next Steps

1. Explore the existing content in the `docs/` directory
2. Start reading from the welcome section to understand the textbook structure
3. Begin with Module 1 to learn ROS 2 fundamentals
4. Try the practical exercises in each chapter
5. Use the chat assistant for help when needed
6. Complete the assessments at the end of each module