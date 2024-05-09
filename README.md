# IntelliGin

This repository contains our source code for our cs2951f project investigating applications of DQN to Gin Rummy. Our group members were Michelle Liu, Evan Lu, and Peter Curtin.

## Files

`src.py` contains the source for our learned agents, as well as a human agent that can be used to interface with the environment on Google Colab.
The following scripts can be used to start experiments with a variety of models and should be considered example scripts for various configurations.
All of them are designed to be run as Weights and Biases agents, for example: `wandb agent team-name/project-name/sweep-id`, where the sweep would specify the executable used.
1. `run.py` is a training script for our first attempt at solving the environment with DQN, before using DoubleDQN.
2. `run-2.py` is a training script for training our $\text{Simple}_{\text{RULE}}$ model.
3. `run-3.py` is a training script for training our $\text{Rich}_{\text{MIX}}$ model.
4. `run-lstm.py` is a training script for training our `SeqDoubleDQNAgent` on the simplified observation environment.
5. `run-self.py` is a training script for training our $\text{Simple}_{\text{SELF}}$ model.

Finally, the `tests.ipynb` file was used to generate several tables in our final report, and requires checkpoints from trained models to function. 
