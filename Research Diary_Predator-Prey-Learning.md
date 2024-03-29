## To-do | Check list
### What are the sophisticated or complex tasks for robot decision-making in my focus?
1. moving target navigation
2. one predator (faster or UAV) chasing one prey (slower or ground robot)
2. one predator chasing one prey, where the prey has a navigation goal
3. multiple predator chasing one prey
### What is the performance of end-to-end RL on complex robot decision-making tasks?
### Is Ring-action RL framework sufficient for complex robot decision-making?
### What is the performance of Context-Aware-Navigation baseline?
### What kind of graph is more powerful for graph-based neural navigation, uniform graph or RRT?
### Bonus question: Will RT-RRT* approach enough to solve the moving target navigation problem?  
RT-RRT* + short-term trajectory prediction for moving target navigation (no escaping), see the performance

## Procedure of graph-based neural navigation
1. Sensor
2. Navigation Graph: Graph, State of each node S
    State of each node includes (node_abs_coords, indicator, utility, target_heuristic)
    Graph should include (node_index, nodes (the object), adjacent matrix)
3. Attention-based encoder-decoder
    takes (Nav-Graph, robot_position) as input
    output a distribution for node selection
4. path planning to the selected node
    global: A-star, RRT, etc
    local: cmu-planner
5. iteration with a certain fixed frequency

## Research contents
### 4 Tasks
### 3 Navigation frameworks 
### 2 Robot platforms (2D & 3D)
### 1.5 Simulators
- ir-sim for testing ideas
- Gazebo for formal experiments
## Research progress:
### 1. Issues of lidar in ir-sim:
* Issues:
  * cannot observe some polygon obstacles (like rectangles, triangles)
  * runs slowly with high resolution and large ray numbers
  * all robots can only be equipped with same lidar settings  
* Considered Solutions:
    1) give up ir-sim, use Gazebo directly
    2) only use built-in obstacles
    3) use my way of detecting obstacles  
* Final decision:  
Use Gazebo as the experiment environment;  
Use ir-sim as an early-stage testing environment for my development, and possibly a visualization terminal for Gazebo simulation
### 2. Gazebo environment: develop the Pursuit-Evasion Platform for Robot Learning (Pep4RL)
* Base: [Aerial-Navigation-Development-Environment](https://github.com/caochao39/aerial_navigation_development_environment?tab=readme-ov-file) & [Autonomous-Exploration-Development-Environment](https://www.cmu-exploration.com/)
* Reference: [XTDrone](https://github.com/robin-shaun/XTDrone?tab=readme-ov-file), [3DMR](https://github.com/luigifreda/3dmr?tab=readme-ov-file), [SDDPG](https://github.com/combra-lab/spiking-ddpg-mapless-navigation)
## Next To-dos
### Construct the Pursuit-Evasion Platform for Robot Learning (Pep4RL)
- [x] Reproduce _CMU-ANDE_
- [ ] Store useful world files to my platform directory
- [ ] Can the local planner of _CMU-ANDE_ be set as a 2D planner?  
  - What is the "indoor" setting of the local planner in _CMU-ANDE_? 
- [ ] What is the complete process of local planning and path following?
- [ ] Use the drone in _CMU-ANDE_ or use Hector quadrotor?   
**Features:**
- [ ] Automated collision check
- [ ] Integrated with [ir-sim](https://github.com/hanruihua/ir_sim/tree/main) (an easy-to-run python robot simulator) and Gazebo
- [ ] 4 kinds of complex robot decision-making tasks
- [ ] Integrated with ground mobile robot (Turtlebot3) for 2D decision-making and quadrotor (Hector quadrotor) for 3D decision-making
- [ ] A variety of world files for model training and evaluation
- [ ] Multi-robot interaction & learning in Gazebo simulator

### Train an end-to-end RL agent for moving target navigation (2D)
- [ ] Reproduce MADDPG for simple_tag in MPE
- [ ] Train MADDPG for end-to-end simple_tag in ir-sim
- [ ] Train MADDPG for end-to-end simple_tag in Gazebo (training_worlds-ENV5, turtlebot3)

### Train a ring-action RL agent for moving target navigation (2D)
### Priority 1 - Test the idea of graph-based neural navigation in ir-sim
(Reference: [Context-Aware-Navigation](https://github.com/marmotlab/Context_Aware_Navigation))
- [ ] Incrementally construct the grid-like graph from sensor inputs
- [ ] Compute the utility of each node

## Intallation
If you have issues in importing "ir-sim" package in python scripts, try running the following command to install ir-sim in your virtual python environment:
```bash
$ pip install -e path/to/pep4rl/learning/ir_sim_learning
```
This will excute a local installation of ir-sim in "editable" mode, which allows you to install the local project without copying any files.