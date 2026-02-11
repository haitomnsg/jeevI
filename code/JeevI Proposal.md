**JeevI: Deep Reinforcement Learning Rescue Robot** 



![ISMS::Kathmandu University](Aspose.Words.584fa2df-b1e5-4e96-9a59-511742e11dcc.001.jpeg)


**Kathmandu University**

**School of Engineering**

**Department of Artificial Intelligence**

**Dhulikhel, Kavre**


**Code No: AISP 311**


**Semester Project Proposal in Partial Fulfillment of the requirement** 

**for III Year I Semester BTech in AI** 







**Submitted by,**

**Ashish Gupta**

**Rubina Dangol Maharjan**

**Submitted to,**

**Department of Artificial Intelligence, Dhulikhel, Kavre**

<b>Submission Date:  14<sup>th</sup> of Nov 2025</b>







#


# <a name="page2"></a><a name="_toc214131468"></a>**Abstract**
Nepal, a nation of profound geopolitical and environmental significance, is uniquely vulnerable to a spectrum of multi-hazard disasters, including catastrophic earthquakes, frequent, deadly landslides, and severe flooding. These events consistently overwhelm national Search and Rescue (SAR) capabilities, primarily due to the nation's rugged topography and the immediate destruction of critical infrastructure, which renders disaster zones inaccessible to human rescuers and conventional vehicles. This inaccessibility creates a critical time gap between incident and rescue, resulting in preventable loss of life. This project, "JeevI," proposes a simulation-based framework to address this operational gap. It focuses on the development of an autonomous quadrupedal (four-legged) robot, a platform uniquely suited for navigating the complex, unstructured terrain of a Nepali disaster zone. The project's core methodology is the application of Deep Reinforcement Learning (DRL) within the high-fidelity, GPU-accelerated NVIDIA Isaac Lab simulation platform. This project will develop a progressive training curriculum, starting with basic locomotion on varied Nepali terrains (e.g., gravel roads, steep slopes, forests) before advancing to complex, procedurally generated disaster environments (e.g., earthquake rubble, landslide debris, and flooded urban areas). The expected outcome is not a physical robot, but a robust, trained AI "policy" (the robot's brain) and a comprehensive simulation framework. This deliverable will serve as a high-impact, low-cost proof-of-concept, demonstrating the agent's capability to autonomously navigate these hazardous zones and providing a foundational tool for future SAR robotics research at Kathmandu University and within Nepal.

**Keywords:** *Deep Reinforcement Learning (DRL), Quadrupedal Robot, Search and Rescue (SAR), NVIDIA Isaac Lab..*

8

# <a name="page5"></a>**Table of Contents**
[Abstract	i](#_toc214131468)

[List of Figures	iv](#_toc214131469)

[Acronyms/Abbreviations	v](#_toc214131470)

[Chapter 1	Introduction	1](#_toc214131471)

[1.1	Background	1](#_toc214131472)

[1.2	Problem Statement	3](#_toc214131473)

[1.3	Objectives	4](#_toc214131474)

[1.4	Motivation and Significance	5](#_toc214131475)

[1.5	Expected Outcome	6](#_toc214131476)

[Chapter 2	Related Works/ Existing Works	7](#_toc214131477)

[Chapter 3	Procedure and Methods	9](#_toc214131478)

[3.1	Simulation Platform: NVIDIA Isaac Sim & Isaac Lab	9](#_toc214131479)

[3.2	Robotic Agent: The "JeevI" Quadruped	10](#_toc214131480)

[3.3	Environment Design and Domain Randomization (DR)	11](#_toc214131481)

[3.4	The RL Framework: A Markov Decision Process (MDP)	12](#_toc214131482)

[3.5	Training Methodology: Proximal Policy Optimization (PPO)	14](#_toc214131483)

[3.6	Curriculum Learning for Progressive Skill Acquisition	15](#_toc214131484)

[3.7	Evaluation and Benchmarking	17](#_toc214131485)

[Chapter 4	System Requirement Specifications	19](#_toc214131486)

[4.1	Software Specifications	19](#_toc214131487)

[4.2	Hardware Specifications	19](#_toc214131488)

[4.3	Equipment Required	20](#_toc214131489)

[Chapter 5	Project Planning and Scheduling	22](#_toc214131490)

[5.1	Tasks:	22](#_toc214131491)

[Chapter 6	Expected Outcome	23](#_toc214131492)

[References	25](#_toc214131493)


#
#











# <a name="page6"></a><a name="_toc214131469"></a>**List of Figures**
[Figure 1.1 A Nepal Police rescue team searches for victims in the rubble of a collapsed building in Bhaktapur, following the 2015 Gorkha earthquake.	3](#_toc91578187)

[Figure 1.2 A rescue team from the Armed Police Force, Nepal, conducts a search operation at a landslide site in Dhading District.	4](#_toc91578187)

[Figure 3.1 A simulation environment displaying multiple quadrupedal robots navigating a grid-based terrain, likely for studies in robotics, multi-agent systems, or locomotion control.	1](#_toc91578187)0

[Figure 3.3 High-level architecture of the Actor-Critic Reinforcement Learning framework (PPO), illustrating the agent-environment interaction loop.	15](#_toc91578187)

[Figure 5.1 Gantt Chart	22](#_toc91578187)


# <a name="_toc214131470"></a>**Acronyms/Abbreviations**

CNNConvolution Neural Network

AI

Artificial Intelligence

YOLO

DR

DRL

DRRM

DQN

HRL

IMU

LSTM

MDP

PPO

ROS

SAR

USD

You Only Look Once	

Domain Randomization

Deep Reinforcement Learning

Disaster Risk Reduction and Management

Deep Q-Network

Hierarchical Reinforcement Learning

Inertial Measurement Unit

Long Short-Term Memory

Markov Decision Process

Proximal Policy Optimization

Robotic Operating System

Search and Rescue

Universal Scene Description 





1  # <a name="page7"></a><a name="_toc214131471"></a>**Introduction**
   JeevI, a simulation-based research initiative focused on developing an autonomous robotic solution for Search and Rescue (SAR) operations within Nepal's uniquely challenging multi-hazard environments. The core problem this project addresses is the critical inaccessibility of disaster zones following major events like earthquakes, landslides, and floods, where rugged terrain and destroyed infrastructure make rapid human-led rescue impossible. Our proposed solution is the "JeevI" agent, a simulated quadrupedal (legged) robot, whose mobility is ideally suited for such unstructured environments. The project's methodology is centered on Deep Reinforcement Learning (DRL), which will be used to train the agent's "brain." This training will occur within a high-fidelity, GPU-accelerated simulation framework built using NVIDIA Isaac Lab featuring procedurally generated environments that replicate Nepali terrains, earthquake rubble, and landslide debris. The final outcome will be a validated AI policy capable of autonomously navigating these complex virtual disaster zones, providing a foundational proof-of-concept for next-generation, low-cost SAR robotics in Nepal.
   1. ## <a name="_toc214131472"></a>**Background**
      Nepal's geography is both a source of national identity and a catalyst for profound, recurrent natural disasters. The nation is situated directly above the Main Himalayan Thrust, the active geological fault where the Indian Plate is subducting beneath the Eurasian Plate. This tectonic collision, which formed the Himalayas, also makes Nepal one of the most seismically hazardous regions on Earth.

      The 2015 Gorkha earthquake provided a devastating demonstration of this vulnerability. The 7.8-magnitude event and its subsequent aftershocks resulted in the deaths of nearly 9,000 people, injured over 23,000, and destroyed or severely damaged more than 600,000 buildings. The total economic losses were estimated to exceed $7 billion USD, representing a catastrophic blow to the nation's development.

      However, this seismic threat does not exist in isolation. Nepal faces a "multi-hazard" environment, where disasters often compound one another in a devastating cascade. The 2024 Nepal Disaster Report (NDR), which analyzed data from July 2018 to July 2024, provides a stark overview of this reality. Furthermore, climate change and glacial melt are accelerating the risk of severe flooding, with models projecting that the economic impacts of riverine floods in Nepal could triple by 2030.

      These hazards are deeply interconnected. The 2015 Gorkha earthquake, for example, did not just destroy buildings; it critically destabilized mountain slopes. The subsequent 2015 monsoon season then triggered thousands of landslides in these pre-weakened areas, severing access to entire regions that were already reeling from the earthquake. This cascading effect earthquake creating the condition for landslides, which are then triggered by monsoonal floods defines the true challenge of disaster response in Nepal. Any effective SAR solution must be a generalist, capable of navigating the sequential and combined hazards of structural rubble, mud, debris fields, and water.

      |**Hazard Type**|**Number of Incidents**|**Fatalities**|**Houses Destroyed (Partial/Full)**|**Economic Loss (Approx. NPR)**|
      | :-: | :-: | :-: | :-: | :-: |
      |Landslide|2,881|878|4,217|1\.83 Billion|
      |Fire|19,534|619|19,307|12\.18 Billion|
      |Thunderbolt|2,642|477|425|0\.16 Billion|
      |Flood|1,029|260|4,964|6\.55 Billion|
      |Earthquake|239|162|27,249|2\.50 Billion|

      Table 1.1: Multi-Hazard Disaster Impact in Nepal (17 July 2018 – 16 July 2024)


      ![Nepal Police Rescue Team Clearing Debris Stock Photo 374188561 |  Shutterstock](Aspose.Words.584fa2df-b1e5-4e96-9a59-511742e11dcc.002.jpeg)

      **Figure 1.1: <a name="_hlk214132026"></a>A Nepal Police rescue team searches for victims in the rubble of a collapsed building in Bhaktapur, following the 2015 Gorkha earthquake.**
   1. ## <a name="_toc214131473"></a>**Problem Statement**
      The primary challenge in responding to these disasters is a crisis of inaccessibility. In the immediate aftermath of a major earthquake or landslide, Nepal's rugged terrain a key challenge in its own right is compounded by the "non-recoverable" damage to critical infrastructure. Roads, bridges, and trails are not merely blocked; they are often completely destroyed.

      This reality severely constrains current SAR solutions:

      1\. **Human Rescue Teams:** While brave and essential, national responders from the Armed Police Force (APF) and the Nepal Army face extreme operational limitations. They are deployed into highly unstable environments with significant personal risk. Their deployment is often slow, contingent on navigating the same "rugged" and inaccessible terrain that has cut off the victims. Furthermore, they often lack the specialized equipment for complex technical search and rescue in confined spaces.

      2\. **Aerial Assets:** Helicopters are a vital tool for traversing Nepal's mountainous topography and are used for rescue operations. However, their effectiveness is limited by the region's notoriously dynamic and unpredictable weather, their small payload and fuel constraints, and the sheer scale of need, which can involve evacuating hundreds or thousands of people. Most critically, while a helicopter can deliver a rescuer to a disaster zone, it cannot penetrate a collapsed building or search within a debris field.

      This analysis reveals a critical, unresolved "last-mile" operational gap. Aerial assets can provide regional access, but the final, most dangerous 100 meters—from a safe landing zone to a victim trapped under a rubble pile, or across an unstable landslide field remains the primary bottleneck. Wheeled vehicles are useless. Drones can provide an "eye in the sky" but cannot look under rubble.

      ![Rescuers retrieve ten more bodies from Jhyaple Khola landslide](Aspose.Words.584fa2df-b1e5-4e96-9a59-511742e11dcc.003.jpeg)Therefore, the specific problem this project addresses is: the lack of a rapidly deployable, all-terrain autonomous system that can penetrate, traverse, and search complex, unstructured 3D disaster zones (rubble, landslide debris, and flooded areas) that are inaccessible to and/or too dangerous for human rescuers. This project posits that a quadrupedal (legged) robot, which combines all-terrain mobility with the expendability of a machine, is the ideal platform to fill this gap.

      **Figure 1.2: <a name="_hlk214132060"></a>A rescue team from the Armed Police Force, Nepal, conducts a search operation at a landslide site in Dhading District.**
   1. ## <a name="_toc214131474"></a>**Objectives**
      This project is a simulation-only, semester-long endeavor. Its objectives are therefore focused on developing the foundational AI and simulation framework. The four primary objectives are:

1. To design and implement a high-fidelity simulation framework within NVIDIA Isaac Sim, named JeevI-Sim, featuring procedurally generated environments that realistically model characteristic Nepali terrains (e.g., gravel, slopes, forests) and multi-hazard disaster zones (e.g., earthquake rubble, landslide debris, flooded areas).
1. To develop and train a robust, adaptive locomotion control policy for a simulated quadrupedal robot ("JeevI") using Deep Reinforcement Learning (PPO), enabling stable and efficient traversal across all defined unstructured terrains.
1. To implement and train a secondary hierarchical policy for an autonomous, vision-based SAR-specific task, enabling the agent to identify and navigate toward a designated goal (i.e., a "victim" beacon) within the complex disaster environments.
1. To rigorously evaluate the trained "JeevI" agent's performance on a series of "held-out" (unseen) simulation maps, benchmarking its traversal success rate, stability, and time-to-goal against baseline policies to quantify the effectiveness of the specialized training.
   1. ## <a name="_toc214131475"></a>**Motivation and Significance**
      The motivation behind Project JeevI is rooted in a humanitarian purpose, strengthened by practical and scientific goals. Search and rescue missions are often decided within the golden hours, and studies in reinforcement learning for SAR have shown that improving search efficiency by even 160 percent can directly translate to more lives saved. The project also aims to safeguard Nepal’s rescuers, who routinely face dangerous and sometimes fatal conditions. Robots can enter unstable structures, narrow voids, or hazardous environments where sending human teams would be risky. Along with this, Nepal’s disaster reports repeatedly highlight shortages of trained SAR personnel and coordination challenges. Introducing autonomous robots as force multipliers can help a single operator oversee multiple agents, expanding the effective search area and easing the burden on limited human resources.

      The significance of this work extends beyond the technical achievement. It contributes to capacity building in Nepal by establishing a low-cost, simulation-first research framework at Kathmandu University. Since research-grade quadrupedal robots are extremely expensive, this approach makes advanced robotics research accessible without heavy investment. More importantly, the solution is tailored for Nepal’s own disaster conditions rubble fields, steep terrain, mud, and water rather than relying on generic parkour-style training environments that fail to capture the country’s unique challenges. Finally, the project serves as a proof of concept for the university and national agencies like the NDRRMA, showing how modern GPU-accelerated simulation and deep reinforcement learning can be used to create affordable, high-impact tools for one of Nepal’s most urgent national needs.
   1. ## <a name="_toc214131476"></a>**Expected Outcome**
      The project is expected to produce three key outcomes. First, it will deliver a trained AI policy, packaged as a model file, that can handle robust and generalized locomotion across a variety of terrain types. Second, it will include a collection of high-fidelity validation environments along with video demonstrations showing the JeevI robot successfully navigating these challenging scenarios. Finally, the project will culminate in a comprehensive proposal that also functions as a final report, documenting the entire process from methodology and experimentation to the results achieved.
1  # <a name="page8"></a><a name="_toc214131477"></a>**Related Works/ Existing Works**
   **1. Ewers, Jan-Hendrik; Anderson, David; Thomson, Douglas (2025). *Deep reinforcement learning for time-critical wilderness search and rescue using drones.* Frontiers in Robotics and AI**

   Ewers et al. (2025): This highly relevant study demonstrates the power of Deep RL for optimizing drone-based wilderness search missions. Their method leverages a prior probability distribution map of a victim's likely location to train an RL agent. This agent learns to plan an efficient search path that maximizes the probability of a quick discovery. The results are compelling: the learned policy improved search times by over 160% compared to traditional coverage planning algorithms. This quantifiably demonstrates that RL is a superior strategy for time-critical SAR planning.

   **2. Ramezani, M.; Amiri Atashgah, M. A. (2024). *Energy-Aware Hierarchical Reinforcement Learning Based on the Predictive Energy Consumption Algorithm for Search and Rescue Aerial Robots in Unknown Environments.* Drones**

   Ramezani & Amiri Atashgah (2024): This work addresses a critical practical limitation of aerial SAR: the limited battery life of drones. The authors introduced an energy-aware Hierarchical Reinforcement Learning (HRL) framework. In this system, a high-level policy chooses energy-efficient sub-goals, while a low-level controller navigates between them. This is augmented by an LSTM model that predicts battery consumption, allowing the agent to avoid routes that would lead to power failure mid-mission.

   **3. Pan, Hainan; Chen, Xieyuanli; Ren, Junkai; Chen, Bailiang; Huang, Kaihong; Zhang, Hui; Lu, Huimin (2023).** *Deep Reinforcement Learning for Flipper Control of Tracked Robots in Urban Rescuing Environments.* Remote Sensing

   Pan et al. (2023): This work is directly relevant to the application domain of Project JeevI. The authors focused on ground rescue robots, specifically a tracked robot with articulated flippers, designed for negotiating rubble and stairs in urban disaster environments. They developed a novel DRL algorithm (ICM-D3QN) to autonomously control the flippers, using the robot's state and local terrain information as input. This is a crucial finding, as it proves that DRL is a viable method for teaching a robot to autonomously navigate complex disaster rubble. The policy was trained in simulation and validated on a real robot, demonstrating that this autonomous control outperformed manual teleoperation.
1  # <a name="page9"></a><a name="_toc214131478"></a>**Procedure and Methods**
   1. ## <a name="_toc214131479"></a>**Simulation Platform: NVIDIA Isaac Sim & Isaac Lab** 
      The choice of simulation platform is the most critical technical decision for this project. The selected platform must support high-fidelity physics, realistic sensor simulation, and, most importantly, computationally efficient training. For these reasons, this project will be built on the NVIDIA Isaac ecosystem.

- NVIDIA Isaac Sim: This will serve as the core simulation engine.41 Built on the NVIDIA Omniverse™ platform, Isaac Sim is an open-source, extensible framework that provides two essential features: 
  - (1) High-Fidelity Physics via the NVIDIA PhysX® 5 engine, allowing for realistic simulation of rigid-body dynamics, contact physics, and even fluid dynamics (for flood scenarios).
  - (2) Photorealistic Rendering using NVIDIA RTX™, which is essential for training the vision-based policies in Objective 3.
- NVIDIA Isaac Lab: This open-source framework, the successor to Isaac Gym, will be the project's robotics AI training application. Isaac Lab is specifically designed for robot learning and provides the single most important technology for making this semester-long project feasible: massively parallel simulation on the GPU.

Deep Reinforcement Learning requires an agent to collect billions of environmental interactions to learn a complex skill like walking. In a traditional, CPU-based simulator, this would take months or years. Isaac Lab, by contrast, leverages GPU-based parallelization to run thousands of simulation environments in parallel, on the GPU itself.18 This massive data collection throughput allows for state-of-the-art locomotion policies to be trained in a matter of hours, not months. This computational efficiency is the central pillar that makes this ambitious project achievable within a single semester.

![](Aspose.Words.584fa2df-b1e5-4e96-9a59-511742e11dcc.004.png)

**Figure 3.1: <a name="_hlk214132089"></a>A simulation environment displaying multiple quadrupedal robots navigating a grid-based terrain, likely for studies in robotics, multi-agent systems, or locomotion control.**
1. ## <a name="_toc214131480"></a>**Robotic Agent: The "JeevI" Quadruped**
   To accelerate the research and focus on AI development rather than 3D modeling, this project will adopt a pre-existing, validated quadrupedal robot model. Isaac Sim provides a rich library of pre-built robot assets in the Universal Scene Description (OpenUSD) format.

- Base Model: The "JeevI" agent will be based on the Unitree Go1 model. This platform is widely used in quadrupedal research, is well-supported in Isaac Lab, and possesses the dynamic capabilities required for navigating complex terrain.
- Simulated Sensor Suite: The agent will be equipped with a sensor suite that mirrors real-world, state-of-the-art quadruped hardware:
  - Proprioceptive Sensors: These sensors provide information about the robot's own state. This includes: (1) Joint Encoders for the position and velocity of all 12 leg joints, and (2) an Inertial Measurement Unit (IMU) mounted on the robot's base, providing its 3D orientation (roll, pitch, yaw) and 3D linear/angular velocities.
  - Exteroceptive Sensors: These sensors provide information about the external environment. This includes: (1) a Forward-Facing RGB-D (Depth) Camera, which provides both color (for Objective 3) and per-pixel depth information (for terrain-aware locomotion), and (2) Foot Contact Sensors to detect when each foot is touching the ground.
  1. ## ` `**<a name="_toc214131481"></a>Environment Design and Domain Randomization (DR)**
     This section details the creation of the virtual training environments—the JeevI-Sim—which is central to the project's Nepal-specific focus. We will create a procedural generation script in Python, ensuring that the agent never sees the exact same terrain twice. This forces the agent to learn a general, adaptive policy rather than "memorizing" a specific map.

     Terrain Types:

1. Basic Terrains: A set of procedurally generated heightmaps to simulate Nepali landscapes, including: flat ground, gravel/particulate surfaces (simulated with variable friction), steep, variable slopes (emulating mountains), and "forests" (simulated as randomly placed pillar obstacles).
1. Disaster Terrains: These are the key environments for this project.
- Earthquake Rubble: Inspired by research on rubble simulation, a procedural rubble pile generator will be scripted. This script will spawn a randomized collection of primitive shapes (cubes, beams, cylinders, plates) and drop them from a height to create complex, non-planar, and unstable piles of debris.
- Landslide Debris: This will be modeled as a large, steep, uneven heightmap (to simulate the "flow") with high-variance friction coefficients (to simulate mud and loose soil) and populated with large, scattered "boulder" obstacles.
- Flooded Areas: A plane of "water" with simulated physics (e.g., buoyancy and drag) will be overlaid on other terrains (e.g., a "flooded urban street" or "flooded forest"). The depth will be variable and unseen by the agent, forcing it to adapt its gait.

Domain Randomization (DR): To ensure the learned policy is robust and not "overfit" to the simulation's specific physics, we will use Isaac Lab's built-in DR tools. This is a standard technique for improving sim-to-real transfer. During training, the following parameters will be randomized at the start of each simulation:

- Physics Parameters: Robot mass, center of mass, motor strength, joint friction, ground friction coefficients, and the mass of debris objects.
- Visual Parameters: (For Objective 3) The position, intensity, and color of lighting; the textures applied to the ground and debris; and the density of atmospheric effects like fog or smoke.
  1. ## <a name="_toc214131482"></a>**The RL Framework: A Markov Decision Process (MDP)**
     The locomotion challenge is formally defined as a Markov Decision Process (MDP), which provides the mathematical framework for an RL agent. An MDP consists of a state space, an action space, and a reward function.

- State Space ($S$) (The "Observation"): This is the set of information the AI agent "sees" at each timestep to make a decision.
  - Proprioceptive State: The robot's base linear velocity, base angular velocity, and gravity vector (all from the IMU); the position and velocity of all 12 leg joints; and the action taken in the previous timestep (to encourage smoothness).
  - Exteroceptive State: A local heightmap sampled in a grid around the robot. This heightmap is derived from the simulated depth camera and provides the agent with "vision" of the terrain immediately ahead, allowing it to be proactive rather than reactive to obstacles.
- Action Space ($A$) (The "Control"): This defines what the agent can do.
  - The action space will be continuous. The AI policy's neural network will output a target position for each of the 12 leg joints.
  - A simulated low-level Proportional-Derivative (PD) controller will then take these target positions and calculate the motor torques required to reach them. This is a stable and common control structure.
- Reward Function ($R$) (The "Goal"): This is the most critical component of the AI's design. The reward function "shapes" the agent's behavior by providing positive or negative feedback for its actions. A complex, multi-component reward function is required to produce stable, efficient, and robust locomotion, as detailed in Table 3.1.

|**Component**|**Description**|**Type & Weight**|**Rationale**|
| - | - | - | - |
|**Task: Linear Velocity**|Matches a target forward velocity command.|$+$ Positive (High)|The primary task: move forward at the commanded speed.|
|**Task: Lateral Velocity**|Penalizes unwanted sideways "crabbing" or drift.|$-$ Negative|Encourages straight-line tracking.|
|**Task: Angular Velocity**|Matches a target turning velocity command.|$+$ Positive|The primary task: turn at the commanded rate.|
|**Stability: Upright**|Keeps the robot's "up" vector (from IMU) aligned with the gravity vector.|$+$ Positive (High)|The primary stability goal: do not fall over.|
|**Stability: Base Motion**|Penalizes high linear or angular velocity of the *robot's base*.|$-$ Negative|Encourages a smooth, stable gait, minimizing "wobble".|
|**Energy: Torque**|Penalizes the squared sum of all 12 motor torques.|$-$ Negative (Low)|Encourages energy-efficient gaits that don't fight the physics.|
|**Smoothness: Action Rate**|Penalizes large differences in action (target joint) from one timestep to the next.|$-$ Negative|Prevents "jerky" leg movements, which are unstable.|
|**Safety: Termination**|A large fixed penalty if the robot enters a "termination state" (e.g., base hits ground, robot flips over).|$-$ Negative (Large)|The strongest signal to avoid catastrophic failure.|

\
Table 3.1: Reward Function Components for "JeevI" Locomotion Policy

This balanced reward function is key. It does not just reward speed; it rewards stable, smooth, and efficient speed. This balance is what will allow the agent to successfully navigate the chaos of a rubble field, where reckless speed would lead to immediate failure.
1. ## <a name="_toc214131483"></a>**Training Methodology: Proximal Policy Optimization (PPO)**
- Algorithm Choice: This project will use Proximal Policy Optimization (PPO). PPO is an on-policy, actor-critic algorithm that is the de facto industry and research standard for training robust quadrupedal locomotion policies. It is known for its stability, sample efficiency, and reliable convergence.
- Architecture: The PPO agent will be implemented using an Actor-Critic neural network architecture.
  - The Actor (Policy): A Multi-Layer Perceptron (MLP) network that takes the full State (S) as input and outputs the Action (A) (the 12 target joint positions).
  - The Critic (Value): A separate MLP network that takes the State (S) as input and outputs a single value: its prediction of the total future reward it expects to get from that state. The critic's prediction is used to train the actor.
- ![The Actor-Critic Architecture | Download Scientific Diagram](Aspose.Words.584fa2df-b1e5-4e96-9a59-511742e11dcc.005.png)Training Process: The PPO algorithm will run, collecting experience (trajectories of states, actions, and rewards) from the thousands of parallel simulation environments in Isaac Lab. This data is collected into a buffer and then used to update the weights of the Actor and Critic networks via gradient descent.

**Figure 3.3: <a name="_hlk214132878"></a>High-level architecture of the Actor-Critic Reinforcement Learning framework (PPO), illustrating the agent-environment interaction loop.**
1. ## <a name="_toc214131484"></a>**Curriculum Learning for Progressive Skill Acquisition**
   The agent will not be trained on complex rubble from scratch; this would be too difficult and would likely fail to converge. Instead, this project will implement a Curriculum Learning strategy, which is a key technique for training robust policies. The difficulty of the environment will be progressively increased as the agent's performance improves.

   Phase 1: Basic Locomotion.

- Environment: Flat ground only.
- Goal: Train the basic PPO policy to walk, run, and turn by varying the commanded target velocity. The agent first learns to stand, then to walk, then to run.

Phase 2: Terrain Adaptation.

- Environment: The procedural generator for basic terrains is activated. The curriculum will start with 90% flat ground and 10% gravel/slopes. As the agent's performance (e.g., average reward) increases, this ratio will shift (e.g., 80/20, 70/30) until the agent is robustly traversing all basic Nepali terrains.

Phase 3: Disaster Navigation (Core Locomotion Policy).

- Environment: The disaster assets (rubble, landslides, floods) are introduced. The curriculum will start by adding low-density, low-height rubble piles. The complexity, height, and instability of the rubble will be progressively "dialed up" as the agent learns to master them.
- Goal: The final output of this phase is a single, robust locomotion policy that can handle all environments, from flat ground to complex rubble.

Phase 4: SAR Task (Hierarchical RL).

- Concept: To achieve Objective 3, a second, high-level policy will be trained, creating a Hierarchical RL (HRL) structure.
- High-Level Policy (The "Navigator"): This policy will be vision-based. Its State (S) will be the forward-facing camera image and a goal location (the "victim beacon"). Its Action (A) will be to output a target velocity command (e.g., "move forward at 0.5 m/s, turn 10 degrees").
- Low-Level Policy (The "Driver"): This will be the robust locomotion policy trained in Phase 3. It takes the "Navigator's" command as its input and executes the physical leg movements required to achieve that velocity.
- Goal: This two-level system separates the problem. One brain (Navigator) figures out where to go, while another (Driver) figures out how to move the legs.
  1. ## <a name="_toc214131485"></a>**Evaluation and Benchmarking**
     To rigorously validate the success of the project (Objective 4), the final trained policies will be evaluated in a controlled setting.

- Held-Out Test Environments: A set of 5-10 test maps will be created. These maps will be procedurally generated but held-out (i.e., the agent will never have seen them during training). This ensures the test is a true measure of generalization.
- Baseline for Comparison: The performance of the final "JeevI" agent (from Phase 4) will be compared against a baseline agent (e.g., the "Phase 1" policy trained only on flat ground). This will quantify the improvement and demonstrate why the disaster-specific training curriculum was necessary.

Key Performance Metrics:

1. Traversal Success Rate (%): In $N$ test runs, the percentage of runs where the agent successfully reaches the goal location without falling or timing out.
1. Mean Time to Goal (s): The average time taken to complete the successful runs.
1. Stability Metrics: The average angular velocity of the robot's base (a quantitative measure of "wobble") and the average motor torque (a measure of energy efficiency).

This comprehensive methodology, from platform selection to curriculum design, provides a clear and feasible path to achieving the project's objectives, grounded in state-of-the-art robotics research.

1  # <a name="page10"></a><a name="_toc214131486"></a>**System Requirement Specifications** 
   This chapter details the specific software, hardware, and equipment specifications required for the successful execution of Project JeevI. As this is a simulation-only project, these requirements pertain to the development workstation that will be used for designing the environments, running the high-fidelity physics simulations, and—most critically training the Deep Reinforcement Learning models.
   1. ## <a name="_toc214131487"></a>**Software Specifications** 
      The project will be developed on a Linux-based operating system, as this is the standard for the NVIDIA robotics ecosystem and most academic robotics research.

- Operating System: Ubuntu 22.04 LTS (Ubuntu 20.04 LTS is also supported).
- Core Simulator: NVIDIA Isaac Sim (e.g., Version 4.0 or the latest stable release at project commencement).
- RL Framework: NVIDIA Isaac Lab (latest stable release from the official GitHub repository).
- Programming Language: Python 3.10 or higher.
- Machine Learning Library: PyTorch (e.g., Version 2.0 or higher, with the correct CUDA toolkit version), as required by the Isaac Lab installation.
- Version Control: Git, for managing the project's codebase.
  1. ## <a name="_toc214131488"></a>**Hardware Specifications**
     The hardware, specifically the Graphics Processing Unit (GPU), is the single most important factor for this project's feasibility. The entire methodology outlined in Chapter 3 is predicated on GPU-accelerated parallel simulation. A workstation that does not meet these specifications will be unable to run the simulation or, more likely, will be unable to complete the DRL training.

     |**Component**|**Minimum Requirement**|**Recommended Requirement**|
     | - | - | - |
     |**Operating System**|Ubuntu 20.04 LTS|Ubuntu 22.04 LTS|
     |**CPU**|Intel Core i7 (10th Gen) / AMD Ryzen 7 (3000 series)|Intel Core i9 (13th Gen+) / AMD Ryzen 9 (7000 series+)|
     |**GPU (NVIDIA only)**|NVIDIA GeForce RTX 3070 (8 GB VRAM)|**NVIDIA GeForce RTX 4080 (16 GB VRAM) or RTX 4090 (24 GB VRAM)**|
     |**RAM**|32 GB DDR4|64 GB DDR5|
     |**Storage**|1 TB NVMe SSD|2 TB NVMe SSD|
     |**Core Software**|Isaac Sim, Isaac Lab, PyTorch, Python 3.10|Isaac Sim (Latest), Isaac Lab (Latest), PyTorch (Latest)|

     Table 4.1: System Hardware and Software Requirements

     The emphasis on a high-end NVIDIA RTX 40-series GPU (or an equivalent professional-grade A-series card) is not arbitrary. Training state-of-the-art locomotion policies has been demonstrated to take as little as one hour on an RTX 4090. A lesser card, such as the minimum-spec RTX 3070, could extend this training time by a factor of 5-10x, turning a one-hour experiment into a full day's work. This compressed iteration time is vital for debugging the reward function and successfully completing the training curriculum.
  1. ## <a name="_toc214131489"></a>**Equipment Required**
     Based on the specifications above, the project requires the following equipment:

- **Primary Development Workstation:** A single desktop or high-performance laptop workstation that meets or exceeds the *Recommended* hardware specifications listed in Table 4.1. This will be the primary machine for all development, simulation, and training.
- **Access to University HPC:** Kathmandu University possesses a High-Performance Computing (HPC) access to this cluster would be a significant asset for running multiple training experiments in parallel.

1  # <a name="page11"></a><a name="_toc214131490"></a>**Project Planning and Scheduling**
   ![](Aspose.Words.584fa2df-b1e5-4e96-9a59-511742e11dcc.006.png)We aim to complete this project within a four-month timeframe, ensuring that tasks are equitably distributed among the four team members based on their expertise. Weekly meetings will be held on ever project day to discuss progress, troubleshoot issues, and compile code collectively.

   <a name="_toc91578187"></a>**Figure 5.1 Gantt Chart**
   1. ## <a name="_toc214131491"></a>**Tasks:**
1. Setup & Literature Review
1. Basic Terrain Training
1. Begin PPO Training
1. Disaster Terrain Training
1. Evaluation & Analysis

These are the task that we will perform in order to complete our project.
1  # <a name="_toc214131492"></a>**Expected Outcome**
   This chapter provides a detailed description of the concrete deliverables that will be produced upon the successful completion of Project JeevI. These outcomes are aligned with the project's simulation-only scope and are designed to provide a comprehensive and valuable contribution to the field of SAR robotics at Kathmandu University.

   The project's deliverables are categorized into three areas: the AI model, the simulation environment, and the final demonstration package.

   1\. Primary Deliverable: The "JeevI" AI Policy

   The principal outcome of this project is the artificial intelligence itself. This will be delivered as a set of trained neural network policy files (i.e., the saved model weights). This "brain" is the core innovation of the project and will include:

- **The Robust Locomotion Policy:** A policy (from Phase 3) trained on the full disaster curriculum, capable of receiving a target velocity command and executing stable, adaptive locomotion across all simulated environments (flat, gravel, slopes, rubble, landslide, and floods).
- **The Hierarchical SAR Policy:** A two-level policy (from Phase 4) where a high-level, vision-based "Navigator" directs the low-level "Locomotion" policy to autonomously find and move to a designated "victim" beacon within a complex, unseen disaster map.

2\. Secondary Deliverable: The JeevI-Sim Environment

The second major outcome is the complete, reusable NVIDIA Isaac Sim simulation package. This JeevI-Sim environment is a valuable asset for future research, allowing other students to build upon this work. This package will contain:

- All OpenUSD assets for the robot and environment components.
- All Python scripts for the procedural generation of the Nepali terrains (mountains, forests, etc.).
- All Python scripts for the procedural generation of the multi-hazard disaster zones (earthquake rubble, landslide debris, and flooded areas).
- The complete Python codebase for the PPO agent, including the reward function implementations and training configurations.

This set of deliverables represents a complete and successful fulfillment of all four project objectives, providing a robust, data-driven proof-of-concept and a powerful foundational tool for future research.


# <a name="_toc214131493"></a>**References**
Amatya, S. C. (2020). Landslide Disaster Management in Nepal. *Journal of Development Innovations, 4*(1), 1–17.

Agarwal, A., Kumar, A., Malik, J., & Pathak, D. (2023). Legged locomotion in challenging terrains using egocentric vision. In *Conference on Robot Learning (CoRL)*. PMLR.

Ewers, J. H., Anderson, D., & Thomson, D. (2024). Deep Reinforcement Learning for Time-Critical Wilderness Search And Rescue Using Drones. *arXiv preprint arXiv:2405.12800*.

Government of Nepal, Ministry of Home Affairs (MoHA). (2015). *Nepal Earthquake 2015: A Disaster Risk Reduction Situation Report*.

Li, Z., Zhang, J., He, J., Li, Y., & Li, G. (2024). A Dual-Layer Reinforcement Learning-Based Adaptive Control for Quadruped Robots on Complex Terrains. *Applied Sciences, 14*(19), 8697.

Makoviychuk, V., Wawrzyniak, L., Guo, Y., Lu, Y., Handa, A., State, G.,... & Macklin, M. (2021). Isaac Gym: High performance GPU based physics simulation for robot learning. In *Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track*.

Ministry of Home Affairs (MoHA). (2024). *Nepal Disaster Report 2024: Focus on Reconstruction and Resilience*. Government of Nepal.

Pan, H., Chen, B., Huang, K., Ren, J., Chen, X., & Lu, H. (2023). Deep Reinforcement Learning for Flipper Control of Tracked Robots in Complex Terrains. *Remote Sensing, 15*(18), 4616.

Ramezani, M., & Amiri Atashgah, M. A. (2024). Energy-Aware Hierarchical Reinforcement Learning Based on the Predictive Energy Consumption Algorithm for Search and Rescue Aerial Robots in Unknown Environments. *Drones, 8*(7), 283.

Rudin, N., Hoeller, D., Hutter, M., & Scaramuzza, D. (2021). Learning to walk in minutes using massively parallel deep reinforcement learning. *arXiv preprint arXiv:2109.11978*.

Rudin, N., et al. (2025). Isaac Lab: A GPU Accelerated Simulation Framework For Multi-Modal Robot Learning. *arXiv preprint*.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.

Tan, J., Zhang, T., Coumans, E., Iscen, A., Bai, Y., Wawrzyniak, L.,... & Bojovschi, A. (2018). Sim-to-real: Learning agile locomotion for quadruped robots. In *Robotics: Science and Systems (RSS)*.

World Bank Group & Asian Development Bank. (2021). *Climate Risk Country Profile: Nepal*.

Zhuang, Z., Fu, Z., Wang, J., Atkeson, C. G., Schwertfeger, S., Finn, C., & Zhao, H. (2023). Robot parkour learning. *arXiv preprint arXiv:2309.05665*.

