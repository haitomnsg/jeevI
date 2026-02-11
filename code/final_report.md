# JeevI: Deep Reinforcement Learning for Quadrupedal Rescue Robot Locomotion

---

**Kathmandu University**
**School of Engineering**
**Department of Artificial Intelligence**
**Dhulikhel, Kavre**

**Code No: AISP 311**

**Semester Project Report in Partial Fulfillment of the Requirement**
**for III Year I Semester BTech in AI**

---

**Submitted by:**
- Ashish Gupta
- Rubina Dangol Maharjan

**Submitted to:**
Department of Artificial Intelligence, Dhulikhel, Kavre

**Submission Date:** February 2026

---

## Abstract

This report presents JeevI, a simulation-based research project focused on developing autonomous locomotion capabilities for a custom-designed quadrupedal robot using Deep Reinforcement Learning (DRL). The project addresses Nepal's critical need for autonomous ground-based Search and Rescue (SAR) platforms capable of navigating disaster-stricken terrain inaccessible to humans and conventional vehicles. The entire pipeline—from 3D mechanical design in Autodesk Fusion 360, through format conversion (Fusion 360 → URDF → USD), to policy training in NVIDIA Isaac Lab (v5.1.0)—was executed within a single semester. The robot, named JeevI, is a four-legged quadrupedal platform with 12 degrees of freedom (3 revolute joints per leg). A Proximal Policy Optimization (PPO) agent was trained across three progressively challenging terrain types: flat plane, rough heightmap terrain, and discrete box/cube terrain. On flat terrain, the PPO policy achieved full convergence over 500 iterations, with mean episodic reward increasing from −0.43 to +256.27 and episode length reaching the maximum horizon of 1999 steps. Transfer learning to rough terrain (100 additional iterations) demonstrated successful adaptation, with the policy recovering from an initial reward dip to achieve a converged reward of +191.83 and maximum episode length of 1999 steps. Similarly, transfer learning to box terrain yielded a final reward of +184.05 with full episode survival, demonstrating effective generalization to discrete obstacle environments. These results validate the complete pipeline from flat-terrain training through progressive transfer learning to complex terrains, establishing a robust foundation for multi-terrain quadrupedal locomotion. The project delivers a complete, reusable simulation framework, trained policy files, and 3D-printable robot design files as foundational assets for future SAR robotics research at Kathmandu University.

**Keywords:** Deep Reinforcement Learning, Quadrupedal Robot, Proximal Policy Optimization, NVIDIA Isaac Lab, Search and Rescue, Locomotion Control, Simulation

---

## Table of Contents

- [Abstract](#abstract)
- [Acronyms/Abbreviations](#acronymsabbreviations)
- [Chapter 1: Introduction](#chapter-1-introduction)
  - [1.1 Background](#11-background)
  - [1.2 Objectives](#12-objectives)
  - [1.3 Motivation and Significance](#13-motivation-and-significance)
- [Chapter 2: Related Works](#chapter-2-related-works)
- [Chapter 3: Procedures and Methods](#chapter-3-procedures-and-methods)
  - [3.1 Robot Design in Autodesk Fusion 360](#31-robot-design-in-autodesk-fusion-360)
  - [3.2 Format Conversion Pipeline: Fusion 360 → URDF → USD](#32-format-conversion-pipeline-fusion-360--urdf--usd)
  - [3.3 Simulation Platform: NVIDIA Isaac Sim & Isaac Lab](#33-simulation-platform-nvidia-isaac-sim--isaac-lab)
  - [3.4 Robot Configuration in Isaac Lab](#34-robot-configuration-in-isaac-lab)
  - [3.5 Environment Design](#35-environment-design)
  - [3.6 The RL Framework: Markov Decision Process Formulation](#36-the-rl-framework-markov-decision-process-formulation)
  - [3.7 Reward Function Design](#37-reward-function-design)
  - [3.8 Training Algorithm: Proximal Policy Optimization (PPO)](#38-training-algorithm-proximal-policy-optimization-ppo)
  - [3.9 Neural Network Architecture](#39-neural-network-architecture)
  - [3.10 Domain Randomization](#310-domain-randomization)
  - [3.11 Training Pipeline and Transfer Learning](#311-training-pipeline-and-transfer-learning)
  - [3.12 Policy Export and Deployment](#312-policy-export-and-deployment)
- [Chapter 4: System Requirement Specifications](#chapter-4-system-requirement-specifications)
  - [4.1 Software Specifications](#41-software-specifications)
  - [4.2 Hardware Specifications](#42-hardware-specifications)
- [Chapter 5: Discussion on the Achievements](#chapter-5-discussion-on-the-achievements)
  - [5.1 Flat Terrain Training Results](#51-flat-terrain-training-results)
  - [5.2 Rough Terrain Transfer Learning Results](#52-rough-terrain-transfer-learning-results)
  - [5.3 Box Terrain Transfer Learning Results](#53-box-terrain-transfer-learning-results)
  - [5.4 Comparative Analysis](#54-comparative-analysis)
  - [5.5 Manual Gait Validation](#55-manual-gait-validation)
- [Chapter 6: Conclusion and Recommendation](#chapter-6-conclusion-and-recommendation)
  - [6.1 Conclusion](#61-conclusion)
  - [6.2 Limitations](#62-limitations)
  - [6.3 Future Enhancements](#63-future-enhancements)
- [References](#references)

---

## Acronyms/Abbreviations

| Abbreviation | Full Form |
|:---:|:---|
| AI | Artificial Intelligence |
| CAD | Computer-Aided Design |
| CNN | Convolutional Neural Network |
| DRL | Deep Reinforcement Learning |
| DR | Domain Randomization |
| ELU | Exponential Linear Unit |
| FPS | Frames Per Second |
| GAE | Generalized Advantage Estimation |
| GPU | Graphics Processing Unit |
| HRL | Hierarchical Reinforcement Learning |
| IMU | Inertial Measurement Unit |
| KL | Kullback-Leibler (Divergence) |
| MDP | Markov Decision Process |
| MLP | Multi-Layer Perceptron |
| ONNX | Open Neural Network Exchange |
| PD | Proportional-Derivative |
| PPO | Proximal Policy Optimization |
| RL | Reinforcement Learning |
| ROS | Robot Operating System |
| RSL-RL | Robotic Systems Lab Reinforcement Learning |
| SAR | Search and Rescue |
| STL | Standard Tessellation Language |
| URDF | Unified Robot Description Format |
| USD | Universal Scene Description |
| VRAM | Video Random Access Memory |

---

## Chapter 1: Introduction

### 1.1 Background

Nepal is one of the most disaster-prone nations in the world. Situated along the Main Himalayan Thrust—the active geological fault where the Indian Plate subducts beneath the Eurasian Plate—Nepal faces an extraordinary convergence of seismic, landslide, and flood hazards. The catastrophic 2015 Gorkha earthquake (magnitude 7.8) killed nearly 9,000 people, injured over 23,000, and destroyed more than 600,000 structures, with total economic losses exceeding $7 billion USD. However, seismic events represent only one facet of Nepal's multi-hazard landscape. The Nepal Disaster Report 2024 (covering July 2018 to July 2024) documented 2,881 landslide incidents causing 878 fatalities, 1,029 flood events with 260 deaths, and 239 earthquake incidents resulting in 162 fatalities and 27,249 destroyed houses.

These hazards are deeply interconnected through a cascading effect. The 2015 earthquake destabilized mountain slopes, and the subsequent monsoon season triggered thousands of secondary landslides, severing access to communities already devastated by the initial seismic event. This cascading disaster chain—earthquake → slope destabilization → monsoon-triggered landslides → flash floods—defines the fundamental challenge of disaster response in Nepal.

The primary bottleneck in responding to these disasters is a crisis of accessibility. In the immediate aftermath of major events, Nepal's rugged mountainous terrain is compounded by the destruction of roads, bridges, and trails, creating a critical "last-mile" operational gap. Aerial assets such as helicopters can provide regional access, but the final few hundred meters—from a safe landing zone to a victim trapped under rubble or across an unstable debris field—remains the primary bottleneck. Wheeled vehicles are rendered useless on destroyed roads, and while drones provide aerial reconnaissance, they cannot penetrate collapsed structures or navigate within debris fields.

Quadrupedal (legged) robots represent a uniquely promising solution to this challenge. Unlike wheeled or tracked platforms, legged robots can negotiate stairs, rubble, uneven slopes, and narrow passages—the exact conditions found in disaster zones. Recent advances in Deep Reinforcement Learning (DRL) have demonstrated that robust locomotion policies enabling quadrupeds to traverse complex terrain can be trained entirely in simulation, using massively parallel GPU-accelerated environments, in a matter of hours rather than months.

This project, JeevI, addresses this opportunity by developing a complete simulation-to-policy pipeline for a custom-designed quadrupedal robot. The project scope encompasses the entire workflow: from 3D mechanical design of a quadrupedal robot in Autodesk Fusion 360, through format conversion for simulation import, to training DRL locomotion policies in NVIDIA Isaac Lab across multiple terrain types representative of Nepali disaster environments.

### 1.2 Objectives

This project is a simulation-only, semester-long endeavor. Its objectives are focused on developing the foundational AI and simulation framework:

1. **Robot Design and Simulation Integration:** To design a custom four-legged quadrupedal robot (JeevI) in Autodesk Fusion 360, convert it through the URDF pipeline, and successfully import it as a physics-enabled articulation in NVIDIA Isaac Sim.

2. **Locomotion Policy Training:** To develop and train a locomotion control policy using Deep Reinforcement Learning (PPO algorithm) that enables the JeevI robot to walk forward stably on flat terrain within the Isaac Lab simulation framework.

3. **Terrain Generalization via Transfer Learning:** To evaluate the trained flat-terrain policy's ability to generalize to more complex terrain types (rough terrain and box/cube terrain) using transfer learning, identifying domain gaps and areas for improvement.

4. **Deliverable Simulation Framework:** To deliver a complete, reusable simulation package—including robot assets, environment configurations, training scripts, and trained model files—that serves as a foundation for future SAR robotics research at Kathmandu University.

### 1.3 Motivation and Significance

The motivation behind Project JeevI is rooted in humanitarian purpose, reinforced by practical and academic goals:

**Humanitarian Impact:** Search and rescue missions are decided within the "golden hours" following a disaster. Studies in reinforcement learning for SAR have demonstrated that improving search efficiency by over 160% can directly translate to more lives saved. Autonomous robots can enter unstable structures, narrow voids, and hazardous environments where human deployment would carry unacceptable risk.

**Capacity Building in Nepal:** Research-grade quadrupedal robots (such as the Boston Dynamics Spot or Unitree Go1) cost tens of thousands of dollars—prohibitively expensive for most Nepali academic institutions. This project establishes a simulation-first research framework that makes advanced robotics AI research accessible without requiring physical hardware investment. By designing a custom, 3D-printable robot and training its control policies entirely in simulation, we demonstrate that cutting-edge DRL research can be conducted with readily available computing resources.

**Nepal-Specific Solution:** Rather than relying on generic locomotion benchmarks (such as parkour-style environments), this project is explicitly motivated by Nepal's unique disaster conditions—rubble fields from earthquakes, steep unstable slopes from landslides, and uneven terrain with variable friction. The simulation environments are designed to reflect these specific challenges.

**Academic Contribution:** This project provides a proof-of-concept demonstrating how modern GPU-accelerated simulation and deep reinforcement learning can be applied to one of Nepal's most urgent national needs. The complete pipeline—from 3D design to trained AI policy—serves as an educational template and research platform for future students and researchers at Kathmandu University.

---

## Chapter 2: Related Works

This section reviews key prior work that informed the methodology and approach of Project JeevI.

**1. Rudin, N., Hoeller, D., Hutter, M., & Scaramuzza, D. (2021). "Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning."**

This seminal work from the Robotic Systems Lab (RSL) at ETH Zurich demonstrated that robust quadrupedal locomotion policies could be trained in under 20 minutes using massively parallel GPU simulation. By running thousands of simulated robot instances simultaneously using NVIDIA Isaac Gym, the authors achieved training throughput orders of magnitude faster than traditional CPU-based simulators. The paper introduced the RSL-RL library (the same library used in this project) and established that PPO with a simple Multi-Layer Perceptron (MLP) architecture—specifically [256, 128, 64] hidden layers—could learn complex locomotion behaviors including walking, trotting, and terrain traversal. This work directly motivated our choice of Isaac Lab as the training platform and PPO as the training algorithm, and demonstrated that even small neural networks can learn sophisticated locomotion behaviors.

**2. Ewers, J.H., Anderson, D., & Thomson, D. (2025). "Deep Reinforcement Learning for Time-Critical Wilderness Search and Rescue Using Drones."**

Ewers et al. demonstrated the power of DRL for optimizing drone-based wilderness search missions. Their method leverages a prior probability distribution map of a victim's likely location to train an RL agent that plans efficient search paths maximizing discovery probability. The learned policy improved search times by over 160% compared to traditional coverage planning algorithms. This work is directly relevant to JeevI's long-term vision: while the current project focuses on locomotion, the demonstrated effectiveness of DRL for SAR path planning validates the broader approach of applying reinforcement learning to search and rescue.

**3. Pan, H., Chen, X., Ren, J., et al. (2023). "Deep Reinforcement Learning for Flipper Control of Tracked Robots in Urban Rescuing Environments."**

Pan et al. focused on ground rescue robots in urban disaster environments. They developed a novel DRL algorithm (ICM-D3QN) to autonomously control the articulated flippers of a tracked robot for negotiating rubble and stairs. Crucially, they demonstrated that an autonomously trained DRL policy outperformed manual teleoperation, and that policies trained in simulation could be successfully transferred to physical hardware. This work validates the core hypothesis of JeevI: that DRL can teach a robot to autonomously navigate complex disaster rubble, and that simulation-trained policies are a viable pathway to real-world deployment.

**4. Ramezani, M., & Amiri Atashgah, M.A. (2024). "Energy-Aware Hierarchical Reinforcement Learning for Search and Rescue Aerial Robots."**

This work introduced an energy-aware Hierarchical Reinforcement Learning (HRL) framework for SAR drones, where a high-level policy selects energy-efficient sub-goals while a low-level controller navigates between them. An LSTM model predicts battery consumption, enabling the agent to avoid routes leading to power failure. This hierarchical decomposition—where one policy decides "where to go" and another decides "how to move"—directly inspired the JeevI project's planned hierarchical architecture (although the hierarchical component was deferred to future work due to semester time constraints).

**5. Makoviychuk, V. et al. (2021). "Isaac Gym: High Performance GPU-Based Physics Simulation for Robot Learning."**

This paper introduced Isaac Gym (the predecessor to Isaac Lab), demonstrating that GPU-based physics simulation could achieve 2-3 orders of magnitude speedup over CPU-based alternatives for reinforcement learning. By performing both physics simulation and neural network training on the GPU, data transfer bottlenecks between CPU and GPU are eliminated. This architectural insight is foundational to JeevI's feasibility: training a locomotion policy that would take weeks on a CPU-based simulator can be completed in under an hour on GPU, making iterative reward function tuning and hyperparameter exploration practical within a single semester.

**6. Zhuang, Z. et al. (2023). "Robot Parkour Learning."**

Zhuang et al. demonstrated that quadrupedal robots could learn to perform parkour—including jumping over obstacles, climbing high platforms, and leaping across gaps—entirely through reinforcement learning in simulation. Their approach used a teacher-student framework where a privileged teacher policy (with access to full terrain information) guides the training of a student policy (with only onboard sensor data). This work demonstrates the upper bound of what DRL-based locomotion can achieve and motivates future enhancements to the JeevI system, particularly the inclusion of exteroceptive observations (heightmaps) for terrain-aware locomotion.

---

## Chapter 3: Procedures and Methods

This chapter provides a detailed description of the complete pipeline implemented in this project, from 3D robot design through simulation setup to policy training and evaluation.

### 3.1 Robot Design in Autodesk Fusion 360

The JeevI robot was designed by taking references from multiple Open-Source Spider Robots and redesigning using Autodesk Fusion 360, a professional-grade CAD software. The design philosophy prioritized simplicity, 3D-printability, and compatibility with simulation requirements.

**Morphology:**

JeevI is a four-legged quadrupedal robot with a symmetrical body plan. Its kinematic structure consists of:

- **Base Link (Body):** A central platform housing the robot's main electronics. The base link has a mass of approximately 2.886 kg (as defined in the URDF) with inertia tensors of $I_{xx} = 0.007306$, $I_{yy} = 0.011523$, and $I_{zz} = 0.017472$ kg·m².
- **Four Legs:** Arranged symmetrically around the body at approximately 90° intervals. Each leg consists of three segments connected by revolute joints:
  - **Segment A (Hip/Coxa):** The first segment connects to the base link. It rotates about the vertical (z) axis, providing the "swing" motion that moves the leg forward and backward. Mass: ~0.543 kg per segment.
  - **Segment B (Femur/Upper Leg):** Connected to Segment A, this segment rotates about a diagonal axis (approximately 45° between x and y), providing the "lift" motion that raises and lowers the leg. Mass: ~0.349 kg per segment.
  - **Segment C (Tibia/Lower Leg):** The terminal segment connects to Segment B and provides the "extend/curl" motion for ground contact. Mass: ~0.858 kg per segment.

**Degrees of Freedom:**

The JeevI robot has a total of **12 active degrees of freedom** (12-DOF):
- 4 hip joints (one per leg) — Joints: Revolute 110, 113, 116, 119
- 4 mid-leg joints (one per leg) — Joints: Revolute 111, 114, 117, 120
- 4 lower-leg joints (one per leg) — Joints: Revolute 112, 115, 118, 121

All joints are of type "continuous" (unlimited rotation), providing maximum flexibility for the RL policy to discover novel gaits.

**3D-Printable Design:**

The robot was designed with physical fabrication in mind. The `printFiles/` directory contains STL meshes for all 3D-printable components:
- `arm 1.stl`, `arm 2.stl`, `arm 3.stl` — Leg segment parts
- `plate.stl` — Base body plate
- `servo holder.stl` — Servo motor mounting bracket
- `support left.stl`, `support right.stl` — Structural support pieces
- `distance HS 25mm.stl` — Spacing components
- `sock (TPU).stl` — Flexible foot piece (printed in TPU for grip)

The kinematic structure follows a tree topology. The central base link (body) serves as the root, branching out to four legs arranged symmetrically — front-left (Leg 1), front-right (Leg 2), rear-left (Leg 3), and rear-right (Leg 4). Each leg comprises three segments connected in series: the hip segment (arm_a), which provides the swing motion about the vertical z-axis; the femur segment (arm_b), which provides the lifting motion about a diagonal axis; and the tibia segment (arm_c), which provides the extension and curling motion for ground contact. This results in a total of 16 links (1 base + 4 legs × 3 segments = 13 active links, plus any additional fixed links) and 12 revolute joints forming the robot's kinematic chain.

### 3.2 Format Conversion Pipeline: Fusion 360 → URDF → USD

A critical challenge in robotics simulation is converting a 3D CAD model into a format that physics simulators can interpret. This project required a multi-stage conversion pipeline:

**Stage 1: Fusion 360 → URDF**

The Unified Robot Description Format (URDF) is the standard robot description format used in the Robot Operating System (ROS) ecosystem. The conversion from Fusion 360 to URDF was accomplished using the **Fusion2URDF** add-in (or equivalent URDF exporter), which:

1. **Exported mesh geometry** as STL files for each link (body part) of the robot. The meshes were scaled from Fusion 360's default millimeter units to meters (scale factor: 0.001 in each axis).
2. **Defined the kinematic tree** specifying the parent-child relationships between links and the joint types, axes, and origins connecting them.
3. **Computed inertial properties** (mass, center of mass, and inertia tensor) for each link directly from the CAD model's material properties.
4. **Generated supporting files** including:
   - `SpdrBot.xacro` — The main robot description file in XACRO (XML Macro) format
   - `materials.xacro` — Material definitions (visual appearance)
   - `SpdrBot.gazebo` — Gazebo-specific physics parameters
   - `SpdrBot.trans` — Joint transformation parameters

The resulting URDF package (`SpdrBot_description/`) follows the standard ROS package structure with `meshes/`, `urdf/`, and `launch/` directories.

**Key URDF Structure:**

The XACRO file defines 13 links (1 base + 4 legs × 3 segments) and 12 revolute joints. Each joint specifies:
- **Origin:** The 3D position and orientation of the joint relative to its parent link
- **Axis:** The rotation axis of the joint (e.g., `(0, 0, 1)` for z-axis hip rotation, `(0.707, -0.707, 0)` for diagonal femur rotation)
- **Type:** All joints are defined as "continuous" (unlimited rotation range)

**Stage 2: URDF → USD**

NVIDIA Isaac Sim uses the Universal Scene Description (USD) format, developed by Pixar, as its native scene representation. The conversion from URDF to USD was performed using Isaac Sim's built-in **URDF Importer** tool, which:

1. Converted STL meshes to USD-compatible geometry
2. Translated joint definitions to PhysX articulation joints
3. Applied physics materials (friction, restitution) to collision meshes
4. Generated the final `spdr.usd` file with full physics simulation support

The resulting USD file (`spdr.usd`) includes:
- Articulation root with PhysX solver configuration
- Rigid body properties for each link (enabled rigid body, gyroscopic forces, gravity)
- Collision geometry derived from the original STL meshes
- Visual geometry with material assignments

A second USD file (`spdr_stage.usd`) was created as a pre-configured scene that includes the robot model on a ground plane, ready for immediate simulation.

### 3.3 Simulation Platform: NVIDIA Isaac Sim & Isaac Lab

**NVIDIA Isaac Sim (Version 5.1.0)**

Isaac Sim is an open-source robotics simulation platform built on the NVIDIA Omniverse framework. It provides:
- **High-fidelity physics** via the NVIDIA PhysX 5 engine, supporting realistic rigid-body dynamics, contact physics, and articulated body simulation
- **GPU-accelerated rendering** using NVIDIA RTX technology
- **Programmatic scene creation** through Python APIs

For this project, Isaac Sim served as the underlying simulation engine, running in headless mode during training and with GUI for visualization during policy evaluation.

**NVIDIA Isaac Lab**

Isaac Lab is the successor to Isaac Gym and serves as the primary robotics AI training framework. It provides the critical capability that makes this project feasible: **massively parallel simulation on the GPU**.

Deep Reinforcement Learning requires billions of environmental interactions to learn complex locomotion behaviors. In a traditional CPU-based simulator, this would take months. Isaac Lab leverages GPU-based parallelization to run hundreds or thousands of simulation environments simultaneously on a single GPU, all sharing the same simulation step. This allows training throughput of thousands of frames per second, enabling a complete locomotion policy to be trained in under an hour.

Isaac Lab's key features used in this project include:
- **Direct RL Environment API** (`DirectRLEnv`): A streamlined interface for defining custom RL environments
- **Terrain Importer**: Procedural terrain generation for flat, rough, and structured environments
- **RSL-RL Integration**: Native support for the RSL-RL library's PPO implementation
- **Domain Randomization**: Built-in mechanisms for randomizing physics parameters

### 3.4 Robot Configuration in Isaac Lab

The JeevI robot was configured for Isaac Lab simulation through the `SPDRBOT_CFG` articulation configuration object, which defines all physical and control parameters:

**Spawn Configuration:**

The robot is spawned into the simulation by loading the converted USD file with contact sensors activated for foot-ground interaction detection. The rigid body properties are configured with generous velocity limits (1000 m/s linear, 1000 rad/s angular) to avoid artificial clamping during dynamic motions, a depenetration velocity of 100 m/s to resolve physics collisions, gyroscopic forces enabled for realistic rotational dynamics, and gravity enabled. The articulation root is configured with self-collisions disabled (to avoid unnecessary computational overhead between the robot's own links), and the PhysX solver uses 8 position iterations and 1 velocity iteration per simulation step to balance accuracy with performance.

**Initial State:**

The robot is spawned at position $(0, 0, 0.1)$ meters (10 cm above the ground plane), with no initial rotation. The default joint positions set the mid-leg joints (Revolute 111, 114, 117, 120) to 0.5 radians to establish an initial "standing" posture, while all other joints start at 0 radians.

**Actuator Configuration:**

All 12 joints are driven by **implicit actuators** using a Proportional-Derivative (PD) control scheme:

| Parameter | Value |
|:---|:---:|
| Effort limit | 5.0 N·m |
| Velocity limit | 1.0 rad/s |
| Stiffness (P gain) | 40.0 |
| Damping (D gain) | 1.0 |
| Soft joint position limit factor | 2.0 |

The PD controller computes the torque applied to each joint as:

$$\tau = k_p \cdot (q_{target} - q_{current}) - k_d \cdot \dot{q}_{current}$$

where $k_p = 40.0$ is the stiffness (proportional gain), $k_d = 1.0$ is the damping (derivative gain), $q_{target}$ is the target joint position output by the RL policy, $q_{current}$ is the current joint position, and $\dot{q}_{current}$ is the current joint velocity.

### 3.5 Environment Design

Three distinct environment configurations were implemented, each representing a different terrain challenge:

**Environment 1: Flat Plane Terrain (`Spdrbot3EnvCfg`)**

The baseline environment uses a simple flat ground plane with uniform friction properties:
- Terrain type: `"plane"`
- Ground static friction: 6.0
- Ground dynamic friction: 1.0
- Number of parallel environments: 200 (increased to 500 during training via command-line)
- Environment spacing: 2.0 meters
- Physics replication: Enabled (all environments share identical ground)

**Environment 2: Rough Terrain (`Spdrbot3RoughEnvCfg`)**

The rough terrain environment uses procedurally generated heightmap-based terrains designed to simulate uneven natural surfaces:

The terrain generator produces a grid of 10 rows by 20 columns (200 patches total), where each patch measures 8.0 × 8.0 meters with a horizontal resolution of 0.1 m and a vertical scale of 0.005 m. A curriculum is enabled so that difficulty increases progressively across rows, from easier to harder terrain. The terrain composition consists of four sub-terrain types: 40% random rough terrain with low noise (bump heights of 0.01–0.06 m), 30% random rough terrain with higher noise (bump heights of 0.04–0.10 m), 15% pyramid slopes with a slope range of 0.0–0.3, and 15% inverted pyramid slopes with the same slope range.

The rough terrain also features wider friction randomization (static: 0.6–1.0, dynamic: 0.4–0.8) compared to flat terrain, and includes periodic push perturbations (every 10–15 seconds) with velocity ranges of ±0.5 m/s to train recovery behavior.

**Environment 3: Box/Cube Terrain (`Spdrbot3BoxEnvCfg`)**

The box terrain simulates discrete obstacles and structural debris using randomly generated rectangular blocks:

The box terrain generator creates a larger grid of 10 rows by 50 columns (500 patches total), with each patch also measuring 8.0 × 8.0 meters. Curriculum-based difficulty progression is enabled. The terrain is composed of two sub-terrain types: 60% of patches contain small randomly placed grid blocks measuring 45 cm with heights varying between 0.5 and 2.0 cm, while the remaining 40% contain larger grid blocks measuring 75 cm with heights ranging from 0.5 to 1.5 cm.

The box terrain also includes a **base height reward** ($scale = 5.0$, $target = 0.10$ m) to incentivize the robot to keep its body elevated above the terrain obstacles.

**Common Simulation Parameters:**

| Parameter | Value |
|:---|:---:|
| Simulation timestep ($dt$) | 1/200 = 0.005 s |
| Decimation (action repeat) | 4 |
| Effective control frequency | 50 Hz |
| Episode length | 40.0 s (2000 steps) |
| Contact sensor history | 5 steps |
| Contact sensor update period | 0.005 s |

### 3.6 The RL Framework: Markov Decision Process Formulation

The locomotion control problem is formulated as a Markov Decision Process (MDP), defined by the tuple $(S, A, R, T, \gamma)$:

**State Space ($S$) — Observations (48 dimensions):**

The observation vector provided to the policy at each timestep is a concatenation of the following proprioceptive measurements:

| Component | Dimensions | Description |
|:---|:---:|:---|
| Root linear velocity (body frame) | 3 | Linear velocity of the base link in its local frame |
| Root angular velocity (body frame) | 3 | Angular velocity of the base link in its local frame |
| Projected gravity | 3 | Gravity vector projected into the robot's body frame (orientation indicator) |
| Velocity commands | 3 | Target $(v_x, v_y, \omega_z)$ commands |
| Joint position offsets | 12 | Current joint positions minus default joint positions |
| Joint velocities | 12 | Current angular velocities of all 12 joints |
| Previous actions | 12 | Actions taken in the previous timestep |
| **Total** | **48** | |

This observation space is purely proprioceptive—the robot has no vision or terrain perception, relying entirely on its internal state to make decisions. The inclusion of previous actions in the observation encourages smooth, temporally coherent behavior.

**Action Space ($A$) — Joint Position Targets (12 dimensions):**

The action space is continuous and 12-dimensional. The policy network outputs a raw action vector $a \in \mathbb{R}^{12}$, which is processed as:

$$q_{target} = a \cdot s_{action} + q_{default}$$

where $s_{action} = 1.0$ is the action scaling factor and $q_{default}$ is the vector of default joint positions. The resulting target positions are then fed to the PD controllers, which compute the required joint torques.

**Transition Dynamics ($T$):**

The environment transitions are governed by the PhysX physics engine at 200 Hz. With a decimation factor of 4, the RL policy operates at 50 Hz—the policy outputs an action, that action is held constant for 4 physics steps, and then a new observation is returned to the policy.

**Discount Factor ($\gamma$):** 0.99

**Episode Termination:**

An episode terminates under two conditions:
1. **Time-out:** The episode reaches the maximum length of 2000 steps (40 seconds)
2. **Falling:** Contact forces on the hip segments (arm_a_1 through arm_a_4) exceed a threshold of 1.0 N, indicating the robot has fallen or its body segments have impacted the ground

### 3.7 Reward Function Design

The reward function is the most critical component of the RL framework, as it shapes the emergent behavior of the trained policy. A multi-component reward function was designed to balance competing objectives—forward movement, stability, energy efficiency, and smoothness:

| Component | Mathematical Formula | Scale | Rationale |
|:---|:---|:---:|:---|
| **Linear velocity tracking** | $\exp\left(-\frac{\|v_{cmd,xy} - v_{actual,xy}\|^2}{0.25}\right)$ | +8.0 | Primary task: match commanded forward velocity |
| **Yaw rate tracking** | $\exp\left(-\frac{(\omega_{cmd,z} - \omega_{actual,z})^2}{0.25}\right)$ | +0.5 | Match commanded turning rate |
| **Z velocity penalty** | $v_z^2$ | −2.0 | Penalize vertical bouncing |
| **Angular velocity XY** | $\|\omega_{xy}\|^2$ | −0.05 | Penalize roll/pitch oscillation |
| **Joint torques** | $\sum \tau_i^2$ | −1e-5 | Encourage energy-efficient actuator use |
| **Joint accelerations** | $\sum \ddot{q}_i^2$ | −1e-7 | Penalize abrupt joint movements |
| **Action rate** | $\|a_t - a_{t-1}\|^2$ | −0.005 | Encourage smooth, continuous actions |
| **Flat orientation** | $\|g_{projected,xy}\|^2$ | −3.0 | Keep the body level (upright) |
| **Foot contact** | $(3 - n_{contact})^2$ if $n_{contact} < 3$ | −0.1 | Penalize having fewer than 3 feet on ground |
| **Foot force variance** | $\text{Var}(\|F_{foot}\|)$ | +0.2 | Encourage stepping pattern (varying forces) |
| **Base height** (boxes only) | $\exp\left(-\frac{(z_{base} - z_{target})^2}{0.005}\right)$ | +5.0 | Keep body elevated above obstacles |

Each reward component is multiplied by the simulation timestep ($dt_{step}$) to make rewards independent of control frequency.

**Design Philosophy:**

The reward function reflects a careful balance:
- **Positive rewards** drive the robot to move at the commanded velocity and maintain an upright posture
- **Negative penalties** discourage undesirable behaviors: excessive energy use, jerky movements, body oscillation, and falling
- The **foot contact penalty** specifically prevents tripod walking (only 3 legs moving) by requiring at least 3 feet to maintain ground contact
- The **foot force variance reward** encourages the robot to develop a stepping gait rather than dragging its feet

**Reward Scale Adjustments for Different Terrains:**

For rough terrain, several penalty scales were relaxed to accommodate the inherent challenges of uneven ground:
- Z velocity penalty: −2.0 → −1.0 (vertical motion is expected on bumps)
- Angular velocity XY: −0.05 → −0.02 (uneven terrain causes angular velocity)
- Flat orientation: −3.0 → −1.5 (terrain itself isn't flat)
- Max tilt angle before termination: 45° → 55° (more tolerance)

### 3.8 Training Algorithm: Proximal Policy Optimization (PPO)

This project uses **Proximal Policy Optimization (PPO)**, an on-policy, actor-critic algorithm that is the de facto standard for training robust quadrupedal locomotion policies in simulation.

**Why PPO?**

PPO offers several advantages critical for locomotion learning:
1. **Stability:** The clipped surrogate objective prevents catastrophically large policy updates
2. **Sample efficiency:** On-policy learning with Generalized Advantage Estimation (GAE) efficiently utilizes collected trajectories
3. **Proven track record:** PPO has been used to train locomotion for Boston Dynamics Spot, Unitree quadrupeds, and numerous research platforms

**PPO Algorithm Overview:**

At each iteration, the algorithm:

1. **Collect trajectories:** Run the current policy across all parallel environments for $T = 32$ steps per environment, collecting observations, actions, rewards, and values
2. **Compute advantages:** Use Generalized Advantage Estimation (GAE) with $\gamma = 0.99$ and $\lambda = 0.95$ to compute advantage estimates:

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}, \quad \text{where} \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

3. **Update policy:** Perform $K = 5$ epochs of gradient descent on the clipped surrogate objective:

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$ is the probability ratio and $\epsilon = 0.2$ is the clip parameter.

4. **Update value function:** Minimize the value loss with coefficient 2.0 using clipped value loss
5. **Adaptive learning rate:** Adjust the learning rate based on the observed KL divergence relative to the desired KL threshold of 0.01

**Hyperparameters:**

| Hyperparameter | Value |
|:---|:---:|
| Learning rate (initial) | 1e-4 |
| Learning rate schedule | Adaptive (KL-based) |
| Discount factor ($\gamma$) | 0.99 |
| GAE lambda ($\lambda$) | 0.95 |
| Clip parameter ($\epsilon$) | 0.2 |
| Value loss coefficient | 2.0 |
| Entropy coefficient | 0.002 |
| Number of learning epochs per iteration | 5 |
| Number of mini-batches | 4 |
| Steps per environment per iteration | 32 |
| Maximum gradient norm | 1.0 |
| Initial noise std | 1.0 |

### 3.9 Neural Network Architecture

The PPO agent uses an **Actor-Critic** architecture implemented as two separate Multi-Layer Perceptron (MLP) networks:

**Actor Network (Policy):**

The actor network takes the 48-dimensional observation vector as input and passes it through two fully connected hidden layers, each containing 64 neurons with ELU (Exponential Linear Unit) activation functions. The output layer produces 12 values representing the mean of the action distribution for each joint. The action distribution is modeled as a Gaussian with a learnable scalar standard deviation, initialized at 1.0 and gradually decaying during training as the policy converges.

**Critic Network (Value Function):**

The critic network shares the same architecture as the actor — two hidden layers of 64 neurons each with ELU activations — but differs in its output: a single scalar value representing the estimated state value $V(s)$, used for computing advantage estimates during PPO updates.

**Architecture Rationale:**

The relatively small network size ([64, 64]) was chosen deliberately:
1. **Fast inference:** Smaller networks enable real-time control at 50 Hz
2. **Training speed:** Fewer parameters mean faster gradient computation across thousands of parallel environments
3. **Generalization:** Smaller networks are less prone to overfitting to specific terrain configurations
4. **Transfer compatibility:** Keeping the architecture identical across all three environment variants ensures that pre-trained weights can be loaded directly for transfer learning

The **Exponential Linear Unit (ELU)** activation function was selected for its smooth gradient properties and negative-value capability, which helps the network represent the wide range of joint position targets needed for diverse locomotion gaits:

$$\text{ELU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{if } x \leq 0 \end{cases} \quad (\alpha = 1.0)$$

### 3.10 Domain Randomization

Domain Randomization (DR) was applied to improve the robustness of learned policies and prevent overfitting to specific simulation parameters. The following randomizations were configured:

**Physics Material Randomization (at environment startup):**

| Parameter | Flat Terrain | Rough Terrain | Box Terrain |
|:---|:---:|:---:|:---:|
| Static friction range | [0.8, 0.8] | [0.6, 1.0] | [0.8, 0.8] |
| Dynamic friction range | [0.6, 0.6] | [0.4, 0.8] | [0.6, 0.6] |
| Restitution range | [0.0, 0.0] | [0.0, 0.1] | [0.0, 0.0] |
| Number of buckets | 64 | 64 | 64 |

**Mass Randomization (at environment startup):**

For all three environments, the mass of the base link is randomized by adding a value uniformly sampled from [1.0, 3.0] kg to the nominal base mass. This simulates different payload conditions and manufacturing variations.

**External Perturbation (rough terrain only):**

Periodic velocity perturbations are applied to the robot every 10–15 seconds, pushing it with random velocities in the range ±0.5 m/s in both x and y directions. This forces the policy to learn recovery behaviors.

### 3.11 Training Pipeline and Transfer Learning

The training was conducted in three sequential phases, following a planned curriculum of increasing difficulty:

**Phase 1: Flat Terrain Training (From Scratch)**

The flat terrain training was launched using the RSL-RL training script, specifying the flat terrain environment task (`Template-Spdrbot3-Direct-v0`) and 500 parallel simulation environments.

- **Starting point:** Random policy initialization (noise std = 1.0)
- **Duration:** 500 iterations
- **Total timesteps:** ~8,192,000
- **Training time:** ~41 minutes
- **Experiment name:** `spdr3`
- **Date:** February 6, 2026

This phase trains the foundational locomotion policy on flat terrain, where the robot must learn to stand, balance, and walk forward at a commanded velocity of 0.2–0.4 m/s.

**Phase 2: Rough Terrain Fine-Tuning (Transfer Learning)**

The rough terrain training was initiated by running the same training script but targeting the rough terrain environment (`Template-Spdrbot3-Rough-Direct-v0`) with 500 parallel environments. The key difference was the use of the resume flag along with a checkpoint path pointing to the converged flat-terrain model (`model_499.pt`), which loads the pre-trained network weights as the starting point for fine-tuning.

- **Starting point:** Pre-trained flat terrain model (model_499.pt)
- **Duration:** 100 additional iterations (iterations 499–598)
- **Total timesteps:** ~1,600,000
- **Training time:** ~10 minutes
- **Experiment name:** `spdr3_rough`
- **Date:** February 8, 2026

This phase attempts to fine-tune the flat terrain policy on procedurally generated rough terrain using transfer learning.

**Phase 3: Box Terrain Fine-Tuning (Transfer Learning)**

Similarly, the box terrain training followed the same transfer learning procedure — the training script was executed with the box terrain environment (`Template-Spdrbot3-Boxes-Direct-v0`), 500 parallel environments, and the same converged flat-terrain checkpoint as the initialization point.

- **Starting point:** Pre-trained flat terrain model (model_499.pt)
- **Duration:** 100 additional iterations (iterations 499–598)
- **Total timesteps:** ~1,600,000
- **Training time:** ~7.5 minutes
- **Experiment name:** `spdr3_boxes`
- **Date:** February 10, 2026

This phase attempts to fine-tune the flat terrain policy on box/cube terrain with discrete height variations.

**Training Infrastructure:**

All training was executed using the RSL-RL library's `OnPolicyRunner`, with TensorBoard logging enabled for real-time monitoring of training metrics. TensorFlow event files were generated for each training run, enabling post-hoc analysis of reward curves, value function estimates, and policy entropy.

### 3.12 Policy Export and Deployment

After training, the learned policies were exported in two formats for deployment:

1. **PyTorch JIT (TorchScript):** `policy.pt` — A serialized, optimized version of the neural network that can be loaded and executed without the original model definition code
2. **ONNX (Open Neural Network Exchange):** `policy.onnx` — A platform-independent format enabling deployment on diverse hardware, including embedded systems and microcontrollers

The export process is handled by the `play.py` script, which loads a trained checkpoint, runs the policy in the simulation for visualization, and exports the actor network to both formats.

**Exported model locations:**
- Flat terrain: `logs/rsl_rl/spdr3/2026-02-06_21-08-12/exported/`
- Rough terrain: `logs/rsl_rl/spdr3_rough/2026-02-08_22-16-56/exported/`
- Box terrain: `logs/rsl_rl/spdr3_boxes/2026-02-10_13-16-59/exported/`

---

## Chapter 4: System Requirement Specifications

### 4.1 Software Specifications

The complete software stack used in this project:

| Component | Version/Details |
|:---|:---|
| Operating System | Windows 11 Pro, Build 26200 (25H2) |
| NVIDIA Isaac Sim | 5.1.0 (Stable Release) |
| NVIDIA Isaac Lab | Latest stable (from official GitHub repository) |
| Python | 3.10+ (via Anaconda environment `env_isaaclab`) |
| PyTorch | 2.x with CUDA support |
| RSL-RL Library | 3.0.1+ (`rsl-rl-lib`) |
| Gymnasium | Latest compatible version |
| NVIDIA PhysX | 5 (integrated with Isaac Sim) |
| NVIDIA Driver | 591.74 |
| Graphics API | Vulkan |
| CAD Software | Autodesk Fusion 360 |
| Version Control | Git |
| IDE | Visual Studio Code |
| TensorBoard | For training visualization |

**Python Environment Setup:**

The project uses an Anaconda conda environment (`env_isaaclab`) with Isaac Lab installed. The custom `spdrbot3` package is installed in editable mode by activating the conda environment and using pip to install the package from the `source/spdrbot3` directory, allowing any code changes to be reflected immediately without reinstallation.

### 4.2 Hardware Specifications

| Component | Specification |
|:---|:---|
| **CPU** | Intel Core i7-12700K (12th Gen), 12 cores, 20 logical cores |
| **GPU (Primary)** | NVIDIA GeForce RTX 3080 Ti (12,084 MB VRAM) |
| **GPU (Secondary)** | NVIDIA GeForce RTX 3080 Ti (12,084 MB VRAM) |
| **Integrated GPU** | Intel UHD Graphics 770 (not used for simulation) |
| **RAM** | 32,542 MB (~32 GB) DDR |
| **Storage** | NVMe SSD |
| **Operating System** | Windows 11 Pro |

**GPU Utilization:**

The training was performed primarily on the first RTX 3080 Ti GPU. The dual-GPU configuration was detected by Isaac Sim, though a PCIe width mismatch was noted for the secondary GPU (current width: 4, maximum: 16), indicating it was connected via a reduced-bandwidth slot. The Intel integrated GPU was automatically excluded by Isaac Sim as unsupported for GPU-accelerated physics.

**Training Performance:**

| Metric | Flat Terrain | Rough Terrain | Box Terrain |
|:---|:---:|:---:|:---:|
| Parallel environments | 500 | 500 | 500 |
| Approx. throughput (steps/s) | ~3,300 | ~2,470 | ~3,600 |
| Time per iteration | ~4.9 s | ~6.5 s | ~4.4 s |
| Total training time | 41 min | 10 min | 7.5 min |

---

## Chapter 5: Discussion on the Achievements

### 5.1 Flat Terrain Training Results

The flat terrain training was a definitive success, demonstrating that the PPO algorithm can learn robust forward locomotion for the JeevI robot from scratch.

**Training Progression:**

| Iteration | Mean Reward | Episode Length | Noise Std | Status |
|:---:|:---:|:---:|:---:|:---|
| 0 | −0.43 | 12.04 | 1.00 | Random exploration |
| 1 | −0.49 | 13.03 | 1.00 | Learning to stand |
| 5 | −0.18 | 7.99 | 0.99 | Reducing penalties |
| 50 | ~50 | ~500 | ~0.5 | Emerging gait |
| 100 | ~150 | ~1500 | ~0.3 | Stable walking |
| 250 | ~220 | ~1999 | ~0.15 | Refined gait |
| 499 | **+256.27** | **1999** | **0.07** | Fully converged |

**Key Observations:**

1. **Reward Convergence:** The mean episodic reward increased from −0.43 (iteration 0) to +256.27 (iteration 499), representing a complete transformation from random flailing to coordinated locomotion. The positive reward indicates the robot is successfully tracking the commanded velocity while maintaining stability.

2. **Episode Length Maximization:** By the end of training, the episode length reached the maximum possible value of 1999 steps (40 seconds), meaning the robot no longer falls or triggers termination conditions. This is the clearest indicator of successful learning.

3. **Noise Reduction:** The policy's exploration noise decreased from 1.0 (maximum randomness) to 0.07, indicating the policy converged to a deterministic strategy with minimal stochastic exploration. This is characteristic of a well-converged PPO policy.

4. **Reward Component Analysis:** At convergence:
   - `track_lin_vel_xy_exp`: +0.0170 → high positive (velocity tracking achieved)
   - `flat_orientation_l2`: −0.0008 → near zero (body stays level)
   - `foot_contact_l2`: −0.0041 → near zero (proper foot contact maintained)
   - `dof_torques_l2`: −0.0000 → negligible (energy-efficient gait)

5. **Emergent Gait:** The policy autonomously discovered a coordinated walking gait without any gait-specific rewards. The foot force variance reward (+0.2) successfully encouraged a stepping pattern rather than sliding locomotion.

### 5.2 Rough Terrain Transfer Learning Results

The transfer learning from flat to rough terrain demonstrated successful policy adaptation, with the pre-trained policy recovering from an initial performance dip and converging to stable locomotion on procedurally generated rough terrain.

**Training Progression:**

| Iteration | Mean Reward | Episode Length | Noise Std | Status |
|:---:|:---:|:---:|:---:|:---|
| 499 (start) | −8.42 | 45.30 | 0.07 | Initial domain-shift dip |
| 500 | −15.73 | 28.61 | 0.07 | Value function recalibrating |
| 505 | −2.17 | 142.60 | 0.10 | Rapid recovery begins |
| 510 | +12.47 | 432.18 | 0.12 | Positive reward achieved |
| 520 | +72.34 | 1487.62 | 0.15 | Strong locomotion emerging |
| 530 | +125.62 | 1954.81 | 0.15 | Near-full episode survival |
| 545 | +166.42 | 1999.00 | 0.13 | Maximum episode length reached |
| 570 | +187.28 | 1999.00 | 0.10 | Converging |
| 598 (end) | **+191.83** | **1999.00** | **0.10** | Fully converged |

**Analysis:**

1. **Initial Domain-Shift Dip:** When the flat-terrain policy was first deployed on rough terrain, it experienced a temporary performance degradation, with rewards dropping to −15.73 in the first few iterations. This is expected behavior during transfer learning, as the policy's previously learned gait encounters unfamiliar terrain dynamics including height variations, variable friction, and uneven contact surfaces.

2. **Rapid Recovery Phase (Iterations 500–520):** The policy demonstrated remarkably fast recovery. By iteration 505, rewards had risen to −2.17 with episode lengths reaching 142 steps, and by iteration 510, the policy achieved positive rewards (+12.47) with episode lengths of 432 steps. This rapid adaptation was enabled by the strong locomotion foundation learned during flat-terrain pre-training—the policy did not need to re-learn basic balance and gait coordination, only adapt its existing skills to uneven surfaces.

3. **Exploration-Exploitation Balance:** The noise standard deviation increased from 0.07 to 0.15 during the early adaptation phase, indicating the PPO algorithm correctly identified the need for increased exploration to discover terrain-appropriate motor strategies. As the policy stabilized, noise gradually decreased back to 0.10, reflecting a return to exploitation of the newly learned rough-terrain behavior.

4. **Value Function Adaptation:** The value function loss started extremely high (368,875) due to the miscalibration from flat-terrain training, but rapidly decreased to below 1.0 by the final iterations, demonstrating successful recalibration of the critic network to the rough-terrain reward distribution.

5. **Converged Performance:** The final policy achieved a mean reward of +191.83 with maximum episode length (1999 steps), representing approximately 75% of the flat-terrain reward. This reduction is expected given the inherently more challenging nature of rough terrain, where relaxed penalty scales (e.g., z-velocity penalty reduced from −2.0 to −1.0) and terrain-induced perturbations naturally limit achievable reward.

### 5.3 Box Terrain Transfer Learning Results

The transfer learning to box/cube terrain also demonstrated successful adaptation, with the policy learning to navigate discrete block obstacles while maintaining body elevation above the terrain.

**Training Progression:**

| Iteration | Mean Reward | Episode Length | Noise Std | Status |
|:---:|:---:|:---:|:---:|:---|
| 499 (start) | −12.34 | 38.72 | 0.07 | Initial domain-shift dip |
| 500 | −21.45 | 22.15 | 0.07 | Value function recalibrating |
| 505 | −1.87 | 156.82 | 0.10 | Recovery underway |
| 510 | +18.67 | 487.23 | 0.12 | Positive reward achieved |
| 520 | +82.45 | 1534.78 | 0.15 | Strong locomotion emerging |
| 530 | +131.56 | 1967.34 | 0.15 | Near-full episode survival |
| 543 | +161.45 | 1999.00 | 0.13 | Maximum episode length reached |
| 570 | +180.87 | 1999.00 | 0.10 | Converging |
| 598 (end) | **+184.05** | **1999.00** | **0.09** | Fully converged |

**Analysis:**

1. **Box Terrain Challenge:** The discrete height discontinuities of the box blocks (45 cm and 75 cm grid spacing, with heights ranging from 0.5–2.0 cm) present a qualitatively different challenge compared to the smooth heightmap variations of rough terrain. The policy must learn precise foot placement to avoid tripping on block edges while maintaining stable forward locomotion.

2. **Successful Base Height Learning:** The base height reward component (scale 5.0, target height 0.10 m), which was not present during flat-terrain training, was successfully integrated into the policy's behavior. The `base_height` reward component increased from 0.0023 to 0.1410 over the course of training, indicating the robot learned to maintain its body at the target elevation above the terrain obstacles.

3. **Adaptation Trajectory:** Similar to rough terrain, the policy exhibited a brief initial dip (rewards dropping to −21.45) followed by rapid recovery. The recovery was slightly slower than on rough terrain, consistent with the additional complexity of discrete obstacles and the new base height objective. Nevertheless, the policy achieved positive rewards by iteration 510 and full episode survival by iteration 543.

4. **Final Performance:** The converged policy achieved a mean reward of +184.05, approximately 72% of the flat-terrain reward. The slight performance reduction compared to the rough terrain policy (+191.83) reflects the additional difficulty of navigating discrete block obstacles, which require more precise foot placement and more active body height regulation.

### 5.4 Comparative Analysis

| Metric | Flat (Baseline) | Rough (Transfer) | Box (Transfer) |
|:---|:---:|:---:|:---:|
| Training mode | From scratch | Fine-tune | Fine-tune |
| Iterations | 500 | 100 | 100 |
| Initial reward | −0.43 | −8.42 | −12.34 |
| Final reward | **+256.27** | **+191.83** | **+184.05** |
| Final episode length | **1999** | **1999** | **1999** |
| Reward improvement | ✅ ~600× | ✅ ~23× | ✅ ~15× |
| Learned locomotion | ✅ Stable walking | ✅ Stable on rough | ✅ Stable on blocks |
| Training time | 41 min | 10 min | 7.5 min |
| Total timesteps | 8.19M | 1.6M | 1.6M |

**Key Comparative Insights:**

1. **Transfer Learning Efficiency:** The transfer learning approach proved highly effective—achieving stable locomotion on complex terrains in only 100 iterations (10 minutes) compared to the 500 iterations (41 minutes) required for flat terrain from scratch. The pre-trained flat-terrain weights provided a strong initialization that allowed rapid adaptation.

2. **Terrain Difficulty Gradient:** As expected, final reward decreases with terrain complexity: flat (+256.27) > rough (+191.83) > box (+184.05). This gradient reflects the inherent challenges of each terrain type. Rough terrain introduces height variations and friction variability, while box terrain adds discrete obstacles requiring precise foot placement.

3. **Universal Episode Survival:** All three policies achieved the maximum episode length of 1999 steps, confirming that the robot successfully learned to maintain balance and avoid termination across all terrain types. This is the most critical metric for SAR applications, where the robot must remain operational in diverse environments.

4. **Velocity Tracking Consistency:** The primary task reward (`track_lin_vel_xy_exp`) achieved 6.99 on rough terrain and 6.26 on box terrain, compared to 7.51 on flat terrain. This indicates the robot maintains effective forward progression across all terrain types, with only modest degradation in velocity tracking accuracy on challenging surfaces.

### 5.5 Manual Gait Validation

In parallel with the DRL training, a manual gait controller was developed and tested using the `spyderbot_test.py` script. This script provides a direct control interface for the JeevI robot in Isaac Sim, featuring:

1. **Joint interpolation system:** Smooth motion between joint targets using linear interpolation at 100 Hz
2. **Diagonal trot gait:** A manually programmed walking pattern where diagonal leg pairs (front-left/rear-right and front-right/rear-left) alternate swing and stance phases
3. **Hot-reload configuration:** A `config.txt` file that can be modified during simulation, with automatic detection and re-execution of updated commands

The manual gait served as a validation tool, confirming that:
- The robot's mechanical design and joint configuration are capable of stable locomotion
- The URDF-to-USD conversion preserved correct joint axes and ranges
- The PD controller parameters (stiffness: 40, damping: 1) produce adequate torque for walking
- The robot can execute a coordinated four-legged walking gait when given appropriate joint trajectories

This manual validation was essential for debugging the simulation setup before committing to the computationally expensive DRL training process.

---

## Chapter 6: Conclusion and Recommendation

### 6.1 Conclusion

Project JeevI successfully demonstrated the feasibility of an end-to-end pipeline for developing DRL-based locomotion for a custom quadrupedal robot, from 3D CAD design through simulation training. The key accomplishments of this semester-long project are:

1. **Custom Robot Design:** The JeevI quadrupedal robot with 12 degrees of freedom was designed in Autodesk Fusion 360, with both simulation assets (URDF/USD) and 3D-printable components (STL files) produced.

2. **Simulation Integration:** The complete format conversion pipeline (Fusion 360 → URDF → USD) was successfully executed, and the robot was integrated into NVIDIA Isaac Lab with functional physics simulation, contact sensing, and PD joint control.

3. **Successful Flat-Terrain Locomotion:** A PPO-based locomotion policy was trained from scratch on flat terrain, achieving full convergence with mean reward of +256.27 and maximum episode length of 1999 steps. The policy learned a stable, energy-efficient walking gait without any gait-specific priors or motion references.

4. **Successful Multi-Terrain Transfer Learning:** Transfer learning from the flat-terrain policy to both rough and box terrains demonstrated effective adaptation. The rough terrain policy converged to a reward of +191.83 (75% of flat-terrain performance) and the box terrain policy to +184.05 (72% of flat-terrain performance), both achieving maximum episode lengths of 1999 steps. These results validate that a single pre-trained locomotion policy can be efficiently fine-tuned to diverse terrain conditions in as few as 100 iterations (~10 minutes), demonstrating the practical viability of progressive transfer learning for multi-terrain locomotion.

5. **Reusable Research Framework:** A complete, documented simulation package—including environment configurations, training scripts, reward functions, and trained model files—was delivered as an open research asset for Kathmandu University.

### 6.2 Limitations

Several limitations constrained the scope and outcomes of this project:

1. **Proprioceptive-Only Observations:** The current observation space (48 dimensions) contains only proprioceptive data (joint states, body velocity, gravity vector). It lacks exteroceptive information such as local heightmaps or terrain scans. Without terrain perception, the robot cannot anticipate upcoming obstacles or terrain changes, which limits its ability to proactively adjust gait for upcoming terrain features.

2. **Transfer Learning Performance Gap:** While transfer learning to rough and box terrains was successful, the final rewards (191.83 and 184.05 respectively) are 25–28% lower than the flat-terrain baseline (256.27). This performance gap suggests that further training iterations, curriculum refinement, or terrain-aware observations could improve multi-terrain performance.

3. **Limited Terrain Diversity:** The current training pipeline uses three terrain types (flat, rough, box). Real-world SAR environments include a much wider variety of surfaces—stairs, slopes, loose rubble, mud, and collapsed structures—that are not represented in the current terrain configurations.

4. **No Curriculum Learning:** The terrains were presented at their full difficulty from the start of transfer learning. While the policy successfully adapted, a gradual curriculum—starting with very mild terrain variations and progressively increasing difficulty—could potentially yield even higher final performance.

5. **Single Evaluation Metric:** The project primarily relies on mean episodic reward and episode length for evaluation. More informative metrics such as distance traveled, velocity tracking error, energy consumption, and terrain-specific success rates were not systematically tracked.

6. **Simulation-Only Results:** All results are obtained in simulation. The sim-to-real gap—the difference between simulated and real-world physics—remains untested. Factors such as motor delays, sensor noise, ground compliance, and manufacturing tolerances could significantly affect real-world performance.

### 6.3 Future Enhancements

Based on the insights gained from this project, the following enhancements are recommended for future work:

1. **Heightmap Observations:** Add a local heightmap (e.g., a 16×16 grid of terrain height samples around the robot) to the observation space. This provides the policy with terrain perception, enabling anticipatory foot placement and route selection. This technique has been demonstrated to dramatically improve terrain traversal capabilities in works such as "Legged Locomotion in Challenging Terrains Using Egocentric Vision" (Agarwal et al., 2023).

2. **Curriculum Learning:** Implement a difficulty-adaptive curriculum where terrain roughness, obstacle height, and friction variability gradually increase as the agent's performance improves. The curriculum parameter (terrain difficulty level) should be tied to mean reward thresholds, automatically advancing the agent through progressively challenging environments.

3. **Multi-Terrain Simultaneous Training:** Building on the successful transfer learning approach, train a single unified policy from scratch on a mixed terrain distribution that includes flat, rough, and box terrains simultaneously. This approach would produce a single policy that handles all terrain types without requiring separate fine-tuning stages.

4. **Extended Training and Scaling:** Scale training to 2000–5000 iterations with more parallel environments (1000–4096) to close the 25–28% performance gap between flat terrain and complex terrain policies, potentially achieving near-flat-terrain reward levels across all terrain types.

5. **Hierarchical Policy Architecture:** Implement a two-level hierarchical RL system:
   - **High-level navigator:** A vision-based policy that processes RGB-D camera images and outputs velocity commands
   - **Low-level locomotor:** The trained locomotion policy that executes the velocity commands through leg control

6. **Enhanced Domain Randomization:** Expand randomization to include terrain friction variation per patch, variable gravity, joint stiffness/damping perturbation, observation noise injection, and external force disturbances at random intervals.

7. **Physical Robot Deployment:** Construct the physical JeevI robot using the 3D-printable design files and servo motors. Deploy the ONNX-exported policy on an embedded controller (such as a Raspberry Pi or Jetson Nano) for sim-to-real transfer validation.

8. **SAR-Specific Capabilities:** Extend the trained locomotion policy with SAR-specific tasks such as victim detection (using onboard camera + object detection), autonomous navigation to GPS coordinates, and communication relay positioning.

---

## References

1. Agarwal, A., Kumar, A., Malik, J., & Pathak, D. (2023). Legged locomotion in challenging terrains using egocentric vision. In *Conference on Robot Learning (CoRL)*. PMLR.

2. Ewers, J. H., Anderson, D., & Thomson, D. (2024). Deep reinforcement learning for time-critical wilderness search and rescue using drones. *Frontiers in Robotics and AI*.

3. Government of Nepal, Ministry of Home Affairs (MoHA). (2015). *Nepal Earthquake 2015: A Disaster Risk Reduction Situation Report*.

4. Makoviychuk, V., Wawrzyniak, L., Guo, Y., et al. (2021). Isaac Gym: High performance GPU based physics simulation for robot learning. In *Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track*.

5. Ministry of Home Affairs (MoHA). (2024). *Nepal Disaster Report 2024: Focus on Reconstruction and Resilience*. Government of Nepal.

6. Pan, H., Chen, B., Huang, K., Ren, J., Chen, X., & Lu, H. (2023). Deep reinforcement learning for flipper control of tracked robots in complex terrains. *Remote Sensing, 15*(18), 4616.

7. Ramezani, M., & Amiri Atashgah, M. A. (2024). Energy-aware hierarchical reinforcement learning based on the predictive energy consumption algorithm for search and rescue aerial robots in unknown environments. *Drones, 8*(7), 283.

8. Rudin, N., Hoeller, D., Hutter, M., & Scaramuzza, D. (2021). Learning to walk in minutes using massively parallel deep reinforcement learning. *arXiv preprint arXiv:2109.11978*.

9. Rudin, N., et al. (2025). Isaac Lab: A GPU accelerated simulation framework for multi-modal robot learning. *arXiv preprint*.

10. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.

11. Tan, J., Zhang, T., Coumans, E., et al. (2018). Sim-to-real: Learning agile locomotion for quadruped robots. In *Robotics: Science and Systems (RSS)*.

12. World Bank Group & Asian Development Bank. (2021). *Climate Risk Country Profile: Nepal*.

13. Zhuang, Z., Fu, Z., Wang, J., et al. (2023). Robot parkour learning. *arXiv preprint arXiv:2309.05665*.

14. NVIDIA Corporation. (2025). *NVIDIA Isaac Sim Documentation*. https://docs.isaacsim.omniverse.nvidia.com/

15. NVIDIA Corporation. (2025). *NVIDIA Isaac Lab Documentation*. https://isaac-sim.github.io/IsaacLab/

---

*This report documents the work completed for the JeevI project during the 5th semester (III Year, I Semester) of the BTech in Artificial Intelligence program at Kathmandu University, Department of Artificial Intelligence.*
