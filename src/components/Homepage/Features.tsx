import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import styles from './Features.module.css';

type FeatureItem = {
  title: string;
  description: JSX.Element;
  link: string;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Module 1: ROS 2 Fundamentals',
    description: (
      <>
        Learn the foundations of ROS 2 architecture, nodes, topics, and services.
        Build your first robotic applications with Python and rclpy.
      </>
    ),
    link: '/docs/module-1/intro'
  },
  {
    title: 'Module 2: Simulation Environments',
    description: (
      <>
        Explore Gazebo and Unity for physics simulation and sensor modeling.
        Create digital twins of your robotic systems.
      </>
    ),
    link: '/docs/module-2/intro'
  },
  {
    title: 'Module 3: NVIDIA Isaac Platform',
    description: (
      <>
        Master the NVIDIA Isaac ecosystem for AI-powered robotics.
        Learn about VSLAM, path planning, and sim-to-real transfer.
      </>
    ),
    link: '/docs/module-3/intro'
  },
  {
    title: 'Module 4: Vision-Language-Action Systems',
    description: (
      <>
        Integrate LLMs with robotics for cognitive planning and multimodal interaction.
        Build next-generation autonomous systems.
      </>
    ),
    link: '/docs/welcome'  // Temporary link to welcome page
  }
];

function Feature({title, description, link}: FeatureItem) {
  return (
    <div className={clsx('col col--6 padding-horiz--md')}>
      <div className="card">
        <div className="card__body">
          <h3>{title}</h3>
          <p>{description}</p>
          <div className="card__footer">
            <Link to={link} className="button button--primary">
              Start Learning
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): JSX.Element {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}