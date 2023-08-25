# Adversarial Causal Bayesian Optimization
Companion code for the paper [Adversarial Causal Bayesian Optimization](https://arxiv.org/abs/2307.16625)

## Credit
The starting point for the code in this repository was
https://github.com/ssethz/mcbo. 

## To get the conda environment setup
Starting with a fresh conda environment, 
```conda install botorch -c pytorch -c gpytorch -c conda-forge```
Then in the base directory of this repository:
```pip install -e .``` 
This conda environment should be loaded whenever you run experiments.

## Running
You can launch experiments by running `scripts/runner.py` for the synthetic function networks and `scripts/bikes_runner.py` for the shared mobility system simulator. Experiment parameters are controlled by the command line inputs.
All experimental results are logged to the Weights and Bias service. Before running you should 
set the variables such as `WANDB_ENTITY` in the runner file to include your WANDB settings. 

## Naming
`MCBO` is the algorithm studied in the Model-based Causal Bayesian Optimization paper.
The algorithm in this repo named `MCBO` is designed for just near-noiseless environments
(like Function Networks). The algorithm named `NMCBO` implements `MCBO` for potentially noisy
environments. 

In the synthetic function network experiments the following algorithms are supported: `Random`, `UCB`, `GP-MW`, `MCBO`, `CBO-MW`. 

In the bikes experiments, the following algorithms are supported: `Random`, `D-GP-MW`, `D-CBO-MW`. 

## File Structure
`mcbo` provides the core functionality of model-based causal bayesian optimization. 
In this folder, 
`mcbo_trial.py` implements the environment interaction loop. 
`models/gp_network.py` contains the class for fitting GPs for EIFN and MCBO
`models/eta_network.py` contains the training loop for the custom optimizer used for
optimizing the acquisition function in `NMCBO`. All other methods use default BOTorch
optimizers. 

`scripts` provides the key functionality for running experiments performed in the paper. 
`scripts/runner.py` can be used to run the experiments from the paper that use synthetic function networks. 
`scripts/bikes_runner.py` can be used to run the experiments from the paper on rebalancing in a Shared Mobility System. 

