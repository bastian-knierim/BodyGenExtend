# Manipulation Tasks in Embodiment Co-Design – BodyGen Extended

This repository provides an extension of the BodyGen framework by introducing
manipulation tasks for embodiment co-design. The focus is on task design and
systematic ablations in MuJoCo environments.

## Contribution

This work extends BodyGen by:
- adding manipulation tasks,
- introducing task-specific environment ablations,
- evaluating embodiment performance across different manipulation settings.

## Tasks

Four manipulation tasks were added. Tasks differ in reward functions, environment configuration. Additionaly tasks were executed with different ablations such as cube size, cube mass, floor friction, additional obstacles, and optional reward terms.




## Pushing a Cube

Initial manipulation task used to select environments for further experiments.
Three environments were tested with ten seeds each. All videos represent different strategies found in each environment.

- **Swimmer**: ▶️ [Video](illustrations/Swimmer.mp4)



https://github.com/user-attachments/assets/a5760f3c-b8a8-4d3c-a28a-5a37e6d05a24



- **Walker**: ▶️ [Video](illustrations/Walker.mp4)



https://github.com/user-attachments/assets/8e2b8dce-313b-4201-863e-6fb55663f085



- **Crawler**: ▶️ [Video](illustrations/Crawler.mp4)

https://github.com/user-attachments/assets/10019e65-758a-4ecc-b197-bbf4e5fcde53


## Pushing a Cube towards a Goal Position

Incremental increase in task complexity by adding a goal position.
Two cube sizes were tested with five seeds each.

- **Crawler (Goal Task)**: ▶️ [Video](illustrations/CrawlerGoal.mp4)


https://github.com/user-attachments/assets/7253d792-9db6-442f-a78f-4ffecda44724


## Flipping a Cube

The agent is trained to flip a cube. Due to unstable results, evaluation was
limited to a single seed with multiple simultaneous environment modifications.

- **Flip task (multiple configurations)**: ▶️ [Video](illustrations/Flip.mp4)


https://github.com/user-attachments/assets/0f9902a3-5638-4d89-9430-93101cb32677


Further seeding would be required for a statistically valid evaluation.

## Lifting a Cube

The agent is trained to lift a cube vertically.
Four ablations were tested, each with three seeds.

- **Lift task (all ablations)**: ▶️ [Video](illustrations/Lift.mp4)


https://github.com/user-attachments/assets/1d1b5d69-d22c-4d0f-8257-95951e80ca3d


## Installation and Training

Installation and training follow the original BodyGen framework.
Please refer to the official BodyGen repository for setup and usage instructions:

https://github.com/GenesisOrigin/BodyGen

## Reproducibility

All experiments were executed with fixed random seeds.
The number of seeds varies per task and is stated in the corresponding sections.

## Citation

If you use this repository, please cite this work in addition to the original
BodyGen paper:

```bibtex
@misc{bodygen_extended_2025,
  title  = {Manipulation Tasks in Embodiment Co-Design: BodyGen Extended},
  author = {Bastian Knierim},
  year   = {2025},
  note   = {Extension of the BodyGen framework with manipulation tasks}
}
