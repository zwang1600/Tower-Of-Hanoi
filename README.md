## What is in this repo?
This project is an implementation of the solved version of the game [Tower of Hanoi](https://en.wikipedia.org/wiki/Tower_of_Hanoi). The interactions between a problem-solving agent and its problem-domain environment are modeled probabilistically using the technique of [Markov Decision Processes](https://en.wikipedia.org/wiki/Markov_decision_process). Two approaches are used to maximize the expected utility: [Value Iteration](https://en.wikipedia.org/wiki/Markov_decision_process#Value_iteration) and [Q-learning](https://en.wikipedia.org/wiki/Q-learning).


## Value Iteration
<img src="https://latex.codecogs.com/svg.image?\large&space;\bg{white}&space;V_{i&plus;1}(s)=\max_{a}\sum&space;_{s'}T_{a}(s,s')[R_{a}(s,s')&plus;\gamma&space;V_{i}(s')]"/>
In value iteration, the agent knows the parameters of the MDP (especially the transition model $T$ and the reward function $R$).

## Q-learning
Q-learning finds an optimal policy in the sense of maximizing the expected value of the total reward. It is a [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) algorithm. In Q-learning, the agent does not know the MDP parameters. 

## How to run the game?

**With GUI**:

In a command line, run: 
```bash
python gui.py
```
If you run into an error: No module named 'tzdata', you can resolve it by running
```bash
pip install tzdata
```
You should be able to see the GUI after this. You can play around with the menu buttons on top. For instance, you can change the noise, rewards, and discount using the "MDP Noise", "MDP Awards" and "Discount" buttons. And you can see value changes and perform steps using the "Value Iteration" and "Q-Learning" buttons.

## Acknowledgements
This is a project from the class [CSE 415: Introduction to Artificial Intelligence](https://courses.cs.washington.edu/courses/cse415/23wi/) from the University of Washington.
