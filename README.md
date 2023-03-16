# Welfare and Fairness in Multi-objective Reinforcement Learning (AAMAS 2023)
[![arxiv badge](https://arxiv.org/pdf/2212.01382.pdf)

This repository contains our implementation of experiments in the paper, primarily using NumPy.

_Abstract_:  We study fair multi-objective reinforcement learning in which an agent must learn a policy that simultaneously achieves high reward on multiple dimensions of a vector-valued reward. Motivated by the fair resource allocation literature, we model this as an expected welfare maximization problem, for some non-linear fair welfare function of the vector of long-term cumulative rewards. One canonical example of such a function is the Nash Social Welfare, or geometric mean, the log transform of which is also known as the Proportional Fairness objective. We show that even approximately optimal optimization of the expected Nash Social Welfare is computationally intractable even in the tabular case. Nevertheless, we provide a novel adaptation of Q-learning that combines non-linear scalarized learning updates and non-stationary action selection to learn effective policies for optimizing nonlinear welfare functions. We show that our algorithm is provably convergent, and we demonstrate experimentally that our approach outperforms techniques based on linear scalarization, mixtures of optimal linear scalarizations, or stationary action selection for the Nash Social Welfare Objective.

## Requirements
To install the necessary packages, run 
```bash
pip install -r requirements.txt
```

## Code Structure
The code has two major parts:
* Simulated enviroments: `Fair_Taxi_MDP_Penalty_V2.py` and `resource_gathering.py`
* Algorithm implementation: `linear_scalarization.py`, `mixture_policy.py`, `nsw_ql.py`, `other_welfare.py`, and `ql.py`

### Experiments
* To reproduce our results for Welfare Q-learning with NSW, mixture policy, and linear scalarization baselines, run those files with same hyper-parameter as reported in the paper, respectively: `nsw_ql.py`, `mixture_policy.py`, `linear_scalarization.py`
* To reproduce our results in the supplementary material for other welfare functions (egalitarian, p-welfare), select the welfare function of your choice in the file, and run `other_welfare.py` with same hyper-parameters.

### See My Agent
I personally find it very satisfying to see the performance of trained agents. Thus, I have provided `evaluate.py` in this repository, which can be used to render the performance of trained agents using PyGame. To use `evaluate.py`, load the trained Q-table saved from training, and pass it into `eval_nsw()` function. An example is shown below.

### Visualization of agent performance trained with *Welfare Q-learning* under *Nash Social Welfare* objective

Our agent is a "taxi driver" who is trying to deliver passengers from multiple different locations (red, blue square represent destination and origin, respectively). Green means there is a passenger currently in taxi, with the number indicating where the passenger is from. Orange means the taxi is currently empty. The goal of the agent is to perform delivery service fairly over the locations, which is defined by *Nash Social Welfare* function in this specific case.

<img src="img/2.gif" width="250" height="250"/> <img src="img/3.gif" width="250" height="250"/> <img src="img/4.gif" width="250" height="250"/>


## Citation
```latex
@article{fan2022welfare,
  title={Welfare and Fairness in Multi-objective Reinforcement Learning},
  author={Fan, Zimeng and Peng, Nianli and Tian, Muhang and Fain, Brandon},
  journal={arXiv preprint arXiv:2212.01382},
  year={2022}
}
```
