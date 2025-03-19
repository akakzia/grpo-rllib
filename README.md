# GRPO Implementation in RLlib for Classic Control Environments

## Overview
This repository provides an implementation of **Group Relative Policy Optimization (GRPO)** in **RLlib** for **classic control environments**. GRPO, originally introduced in the DeepSeek framework for **large language models (LLMs)**, is a variant of **Proximal Policy Optimization (PPO)** that enhances training efficiency by estimating baselines from group scores instead of using a critic model.

## Features
- **GRPO Algorithm:** Implements GRPO within RLlib, leveraging its efficiency in (sparse) reinforcement learning.
- **Classic Control Environments:** Currently, experiments are to be conducted on Pendulum-v1.

## Next Steps
- Implement `GRPOTorchTrainer` loss. 
- Implement `GRPORLModule`.
- Fix termination and truncation flags. 
- Incorporate lambda in the baseline computation. 
- Fine tune GRPO on Pendulum-v1.
- Benchmark GRPO against PPO. 
- Extend GRPO to other environments.

## References
- [DeepSeek: GRPO Introduction](https://github.com/deepseek-ai/DeepSeek-Math)
- [RLlib Documentation](https://docs.ray.io/en/latest/rllib/index.html)

