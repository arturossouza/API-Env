# API Management Reinforcement Learning Environment  

* 1. [Introduction](#1-introduction)
* 2. [States, Actions and Rewards/Penalties](#2-states-actions-and-rewardspenalties)
  * 2.1. [States](#21-states)
  * 2.2. [Actions](#22-actions)
  * 2.3. [Rewards/Penalties](#23-rewardspenalties)
* 3. [Experiments](#3-experiments)

## 1. Introduction

This repository implements and environment for Reinforcement Learning (RL) algorithms based on [Gymnasuym](https://gymnasium.farama.org/index.html) libraries.

The implemented environment represent a Cloud architecture im which the RL agent has to manage the health of a certain API.

## 2. States, Actions and Rewards/Penalties

<!-- ```mermaid
    %% flowchart LR
    %%     state1([Available_Fast_Healthy_Low])
    %%     state2([Available_Fast_Healthy_High])
    %%     state3([Available_Slow_Overloaded_High])
    %%     state4([Available_Slow_Healthy_High])
    %%     state5([Available_Slow_Healthy_High])
    %%     state6([Available_Slow_Healthy_Low])
    %%     state7([Available_Medium_Error_High])
    %%     state8([Available_Medium_Healthy_High])
    %%     state9([Available_Medium_Healthy_Low])
    %%     state10([Offline_Slow_Overloaded_Low])
    %%     state11([Offline_Slow_Healthy_Medium])
    %%     state12([Offline_Fast_Error_Medium])
``` -->

## 2.1. States

The environment models the API in basically 4 features and their respective states:

| Feature     	| States                     	|
|-------------	|----------------------------	|
| Availability 	| Offline, Online            	|
| Speed       	| Slow, Medium, Fast         	|
| Health      	| Healthy, Overloaded, Error 	|
| Capacity    	| Low, Medium High           	|

Therefore, the model can take actions to transit between each of those states.

Is is also worth to mention that a state is composed by a combination of each feature, hence, a valid state $k$ would be:

$$
S_{k} = Availability(k)\underline{}Speed(k)\underline{}Health(k)\underline{}Capacity(k)
$$

For instance, $S_{k}$ could be

$$
S_{k} = Offline\underline{}Slow\underline{}Healthy\underline{}Low
$$

## 2.2. Actions

A complete list of all actions that can be taken is available on ```src/apienv.py``` or in the beginning of the ```main.ipynb```, where also have all the experiments with the API Env.

All the actions defined on this environment are stochastic, and the probability associated with each action also can be seen on the ```src/apienv.py``` file.

## 2.3. Rewards/Penalties

On the notebook with all the experiments we also define the rewards and the penalties.

The rewards are given to the agent based on the state that it was able to achieve and the penalties basically are a measure of bad actions that to take.

For instance, it the API is overloaded, increase CPU usually will work to solve this problem, but have a high penalty on this action because we want the agent to explore better and more efficient actions to take in each situation.

This reflects on the algorithms equations as follows:

$$
Reward(k) \leftarrow{} Reward(k) + Penalty(k)
$$


# 3. Experiments

On the ```main.ipynb``` we have all the experiments that were tried with this new environment.

The current algorithms tested was:

* Dynamic Programming
  * Policy Evaluation
  * Policy Iteration
* Monte Carlo
  * Epsilon Greedy Control
* Temporal Difference
  * Q Learning
  * SARSA
  * Expected SARSA

Each of those algorithms implementations can be found on ```src/algorithms``` folder, and the results of each algorithms such as the environment setup can be found in the main notebook.
