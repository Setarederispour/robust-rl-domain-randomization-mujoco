# Robust Reinforcement Learning via Domain Randomization in MuJoCo

This repository contains a cleaned version of my final project for the Robot Learning course at Politecnico di Torino.

The project investigates the effectiveness of domain randomization for improving the robustness and generalization of reinforcement learning policies in simulated locomotion tasks.

## Project Overview

The goal of the project was to compare different training strategies for robust reinforcement learning under controlled source-target dynamics mismatch.

We considered:

- No Domain Randomization
- Uniform Domain Randomization
- Adversarial Domain Randomization

The experiments were performed on MuJoCo locomotion environments, mainly Hopper and Walker2d, using PPO.

## Main Topics

- Reinforcement Learning
- Proximal Policy Optimization
- MuJoCo locomotion environments
- Domain randomization
- Robust policy learning
- Sim-to-sim / sim-to-real generalization
- Training stability under task complexity

## Repository Structure

```text
scripts/     Main experiment and training scripts
utils/       Helper functions
env/         Custom environment and domain-randomization code
images/      Selected plots and result figures
models/      Optional trained model files, if included
