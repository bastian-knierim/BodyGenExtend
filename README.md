# Manipulation Tasks in Embodiment Co-Design - BodyGen Extended

This work is an extended version of <a href="https://github.com/GenesisOrigin/BodyGen">BodyGen</a>. Modifications were made in the mujoco environments, agent configurations as well as optimization environments. The goal of this work was to extend the existing framework by adding manipulation tasks and comparing ablations. 

## Tasks

Four different Tasks were added, each tested with different abblations. The task is defined by the reward function in combination with the environment. And the ablations are about, size and weight of the cube, friction of the floor and additional objects in the environment, like a wall. Also an additional reward term was tested, rewarding the overall movement in all directions.

### Pushing a Cube

This task was the initial manipulation task, it was used to decide on the environment of further experiments. It is about pushing a cube as far as possible. The reward function is defined as:
$$     r_{push,t} = \frac{p_{cube,t+1}^x - p_{cube,t}^x}{\Delta t}
    + \frac{d_{agent,cube,t} - d_{agent,cube,t+1}}{\Delta t} $$

with
$$    d_{agent,cube,t} = \lVert p_{agent,t} - p_{cube,t} \rVert_2.
$$

Here three different environments were tested with each ten seeds. The following videos show multiple seeds of each environment.

The **Swimmer**:

<video controls src="illustrations/Swimmer.mp4" title="Title"></video>

The **Walker**:

<video controls src="illustrations/Walker.mp4" title="Title"></video>

The **Crawler**

<video controls src="illustrations/Crawler.mp4" title="Title"></video>

### Pushing a Cube towards a goal Position

Adding a goal position for the cube was the next incremental step in complexity, where the reward function was designed as:
$$    r_{pushgoal,t} =
    \frac{d_{cube,goal,t} - d_{cube,goal,t+1}}{\Delta t}
    + \frac{d_{agent,cube,t} - d_{agent,cube,t+1}}{\Delta t} $$

For this task five seeds per ablation were used and in total two ablations. The video shows on the top no ablation and on the bottom a small cube.

<video controls src="illustrations/CrawlerGoal.mp4" title="Title"></video>

### Flipping a Cube

In this task the agent is trained to flip the cube, the reward function is formulated as:
$$   r_{flip,t} =
  \frac{\theta_{cube,goal,t} - \theta_{cube,goal,t+1}}{\Delta t}
  + \frac{d_{agent,cube,t} - d_{agent,cube,t+1}}{\Delta t}$$
with 
$$    \theta = 2 \arccos \left( \left| \langle \mathbf{p}, \mathbf{q} \rangle \right| \right) $$
and in some ablations an additional control reward

$$     r_{cont,t} = ||\mathbf{k}_{cube,t} - \mathbf{k}_{cube,t+1}||_2
$$
with $\mathbf{k}_{cube}$ containing the position and orientation as a $7\times1$ vector. 

Because multiple testing phases did not produce successful results, the testing was constrained to one seed and mutliple environmental modifications at once.
In the following illustration on the
- top left: a tiny light cube and a higher friction were used.
- top right: a tilted initial position, a small light cube and an addtional control reward were used.
- bottom left: a tilted initial position and a tiny cube were used.
- bottom right: a tilted initial position, a tiny light cube and a higher friction were used.

<video controls src="illustrations/Flip.mp4" title="Title"></video>

To ensure a valid evaluation further seeding would be needed. 

### Lifting a Cube

The last task is about lifting a cube, the reward is defined as:
$$     r_{lift,t} =
    \frac{p_{cube,t+1}^z - h_{cube}}{\Delta t}
    + \frac{d_{agent,cube,t} - d_{agent,cube,t+1}}{\Delta t}$$

Here again multiple ablations were used each with three seeds. In the illustration each video represents one ablation. 
On the
- top left: no ablations were used.
- top right: a light cube was used.
- bottom left: a small cube was used.
- bottom right: a small light cube was used.

<video controls src="illustrations/Lift.mp4" title="Title"></video>

## Setup / Training

Because BodyGen was used, the setup and training is similar, only addtional requirements were added, which can be seen in requirements.txt.