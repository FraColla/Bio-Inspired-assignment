Cessna 620 RL Controller

This repository contains a DDPG agent for altitude control of a linearized model of a Cessna Model 620. The code is organized into clear, modular files so it can be easily understood and extended.

Repository layout

* mainCessna620.m
  orchestrates environment setup, training and simulation
* funcReset.m
  initializes the environment, trim conditions and random initial state
* funcStep.m
  advances one simulation step, computes dynamics and reward
* f_rewards.m
  defines the quadratic reward terms and terminal bonuses/penalties
* config.m
  hyperparameter and training configuration
* models
  saved actor and critic networks plus training logs
* results
  reward curves and performance summaries
* README.md
  this file

Requirements
• MATLAB R2021a or later
• Reinforcement Learning Toolbox
• Deep Learning Toolbox

How to run

1. Edit config/params.m to set learning rates, discount factor, soft-update coefficient, buffer size, batch size, noise parameters and training limits.
2. In MATLAB, type
   main
   This will linearize the Simulink plant, create and train the DDPG agent, then run a final simulation and save plots.
3. Examine the results folder for reward-versus-episode plots, saved network files in models/, and altitude-tracking traces in figures/.

Notes
The implementation separates environment definition, agent architecture and reward logic into distinct files. To extend to 6-DOF or lateral dynamics, update funcReset.m and funcStep.m with the additional states and adjust the neural-network dimensions accordingly.

Francesco Collatina
AE4350: Bio-Inspired Intelligence and Learning for Aerospace Applications
Delft University of Technology
