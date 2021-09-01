# Counterfactual-Explanation via Prototype Learning
This repository is the implementation of ["Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations"](https://dl.acm.org/doi/pdf/10.1145/3351095.3372850) and ["Interpretable Counterfactual Explanations Guided by Prototypes"](https://arxiv.org/pdf/1907.02584.pdf)
## Requirements
To install requirements:

```sh
conda env create -f env.yml
conda activate CFE
```

## CoutnerFactual Explanation (CFE) 
To run the data collection with MCTS for initial iteration, run this command with :

```
python test.py
```
All experiments in the paper were conducted with Google Cloud Platform(GCP) and HTCondor to run in parallel. If you want to run the experiment in parallel, we reccomend to set up the HTCondor and environment on your GCP account.

## Merge planning trajectories and experience replays
To merge the data for learning the Q-network and policy, run this command:

python merge_plans.py
## Train the Q-network and policy
To run the Q-network and policy with collected data, run this command:

(train Q-network) python run_learning.py --mode=0
(train policy) python run_learning.py --mode 1
## Monte-Carlo Planning with Language Action Value Estimates (MC-LAVE)
To run the MC-LAVE planning with trained Q-network and policy, run this command with :

python run_planning.py --seed=$SEED --trial=1
(For each iteration, use --trial option with iteration number)

## Acknowledgement
This code is adapted and modified upon the code [github](https://github.com/interpretml/DiCE) of FAT 2020 paper "Explaining Machine Learning Classifiers through Diverse
Counterfactual Explanations". We appreciate their released dataset and code which are very helpful to our research.
