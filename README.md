# Universal Robot Control

This Repo is a collection of the tools developed for my bachelor thesis "Robust Grasp Planning for Robots with Deep Learning Policies" which successfully implemented grasping arbitrary objects with a Universal Robot 10e utilizing an Intel RealSense L515. It contains the following tools:
- RealsenseInterface.py: A tool to interface with the Intel RealSense L515 depth-camera.
- PoseEstimator.py: A tool to estimate the pose of a camera and getting the transformation between it and the world coordinates.
- HandEyeCalibrator.py: A tool to calibrate the hand-eye transformation of a robot using Aruco markers.
- Server.py: A simple server to receive and send data to the robot.
- PolicyGUI.py: A GUI to test and evaluate the grasp policies.
- URPolicy.py: The backend for the PolicyGUI to control the robot and execute a cross entropy method based grasp policy.

The thesis and project are mainly based on [dex-net](https://berkeleyautomation.github.io/dex-net/) and [gqcnn](https://github.com/BerkeleyAutomation/gqcnn) by the [Automation Lab](https://autolab.berkeley.edu/) at UC Berkeley, so a huge shoutout to them for their amazing work and make sure to check out their research.

If you have any questions or need help with the tools, feel free to reach out to me.
