# TDHL_6PA
Training classical agents and agents with a single hidden layer to learn 6 paired associations in a navigationt task with sparse rewards
Agents are trained using Temporal Difference Error modulated Hebbian Plascticity

Run 6pa_classic.py or 6pa_hidden.py to start the 6PA navigation task with the relevant agents.

Other scripts include:
  - 1pa_{} trains the agent to learn single reward locations
  - 6pa_{} trains the agent to learn 6PAs with cues presented throughout the trial
  - wkm_{} trains the agent with a bump attractor to learn 6PAs with transient cues presented
  - 16pa_{} trains the agent with hidden layer to learn 16 PAs with different hyperparameter conditions
  - 6pa_hidden_a2c.py is the agent with a single nonlinear hidden layer and discrete actions trained by Advantage Actor Critic (A2C)
